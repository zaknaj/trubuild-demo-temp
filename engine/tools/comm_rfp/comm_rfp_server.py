"""
FastAPI server for commercial evaluation using the real LLM extraction pipeline.

Run with:
  cd engine/tools/comm_rfp
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json uvicorn mock_server:app --reload --port 8100

Or copy .env.example to .env and set one of: GOOGLE_APPLICATION_CREDENTIALS,
GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY.
"""

import os
import sys
import json
import tempfile
import asyncio
import traceback
import time
from pathlib import Path

# Load .env from this directory so credentials can be set without exporting
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _setup_credentials():
    """Resolve GOOGLE_APPLICATION_CREDENTIALS path and support API-key JSON."""
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not creds_path.strip():
        return
    creds_path = Path(creds_path.strip().strip('"').strip("'"))
    if not creds_path.is_absolute():
        creds_path = Path(os.path.expanduser(str(creds_path))).resolve()
    else:
        creds_path = creds_path.resolve()
    if not creds_path.exists():
        return
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path)
    try:
        with open(creds_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if isinstance(data, dict) and "api_key" in data and "private_key" not in data:
        key = data.get("api_key") or data.get("API_KEY")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
            if not os.environ.get("LLM_PROVIDER"):
                os.environ["LLM_PROVIDER"] = "google"
    elif isinstance(data, dict) and data.get("type") == "service_account":
        if not os.environ.get("LLM_PROVIDER"):
            os.environ["LLM_PROVIDER"] = "vertex"


_setup_credentials()

from typing import Optional, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from tools.comm_rfp.comm_rfp_parse_excel import ExcelParser
from tools.comm_rfp.comm_rfp_extract import (
    extract_boq_parallel,
    extract_divisions_streaming,
    DivisionResult,
    split_content_by_divisions,
    reconcile_boq,
    build_reference_items_from_boq,
    _extract_summary_totals_fast,
    _is_summary_sheet,
    _is_content_sheet,
)
from tools.comm_rfp.comm_rfp_models import BOQ
from tools.comm_rfp.comm_rfp_compare import _compare_boqs, ComparisonReport

app = FastAPI(title="Commercial Evaluation Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_llm_credentials():
    """Raise a clear HTTP error if no LLM credentials are configured."""
    has_vertex = bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    has_google = bool(os.environ.get("GOOGLE_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))

    if not any([has_vertex, has_google, has_openai, has_anthropic]):
        raise HTTPException(
            500,
            "No LLM credentials configured. Set one of: "
            "GOOGLE_APPLICATION_CREDENTIALS (Vertex AI), "
            "GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY",
        )


def _sheet_to_text(sheet) -> str:
    """Convert an ExcelSheet to pipe-delimited text for LLM processing."""
    lines = [
        "# Columns: ITEM_ID | DESCRIPTION | QTY | UNIT | RATE | AMOUNT",
        "# Note: AMOUNT should equal QTY x RATE",
        "",
    ]
    for row in sheet.rows:
        values = [str(v) if v is not None else "" for v in row.raw_values[:6]]
        row_text = " | ".join(v for v in values if v.strip())
        if row_text.strip():
            lines.append(row_text)
    return "\n".join(lines)


def _format_size(n_bytes: int) -> str:
    """Format byte count as human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.1f} KB"
    else:
        return f"{n_bytes / (1024 * 1024):.1f} MB"


def _parse_excel_to_sheets(file_path: str) -> list[tuple[str, str]]:
    """Parse an Excel file and return (sheet_name, sheet_text) tuples."""
    parser = ExcelParser(file_path)
    result = parser.extract()

    sheets_data = []
    for sheet in result.sheets:
        text = _sheet_to_text(sheet)
        if text.strip():
            sheets_data.append((sheet.name, text))
    return sheets_data


def _count_boq_stats(boq: dict) -> tuple[int, int, int]:
    """Return (divisions, groupings, line_items) counts from a BOQ dict."""
    n_div = 0
    n_grp = 0
    n_items = 0
    for d in boq.get("divisions", []):
        n_div += 1
        for g in d.get("grouping", []):
            n_grp += 1
            for sg in g.get("sub_groupings", []):
                n_items += len(sg.get("line_items", []))
    return n_div, n_grp, n_items


def _extract_boq_from_file(file_path: str) -> dict:
    """
    Parse an Excel file, extract BOQ via LLM, return as dict.
    Parses Excel to sheets and runs extract_boq_parallel.
    """
    sheets_data = _parse_excel_to_sheets(file_path)

    if not sheets_data:
        raise ValueError(f"No sheets with data found in {file_path}")

    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    provider = os.environ.get("LLM_PROVIDER", "vertex").strip().lower()

    boq = extract_boq_parallel(
        sheets_data=sheets_data,
        provider=provider,
        model=model,
        verbose=True,
        chunk_threshold=28000,
        max_concurrent=10,
    )

    return boq.model_dump()


def _extract_boq_from_sheets(
    sheets_data: list[tuple[str, str]],
    reference_items_by_division: dict[str, list[dict]] | None = None,
) -> dict:
    """Extract BOQ from pre-parsed sheets data."""
    if not sheets_data:
        raise ValueError("No sheets data provided")

    import sys

    print(f"\n{'='*60}", flush=True)
    print(f"_extract_boq_from_sheets: {len(sheets_data)} sheets", flush=True)
    for name, text in sheets_data:
        from tools.comm_rfp.comm_rfp_extract import _is_summary_sheet, _is_content_sheet
        print(f"  Sheet '{name}': {len(text):,} chars, summary={_is_summary_sheet(name)}, content={_is_content_sheet(name)}", flush=True)
    if reference_items_by_division:
        ref_items_count = sum(len(v) for v in reference_items_by_division.values())
        print(f"  📎 Reference: {len(reference_items_by_division)} divisions, {ref_items_count} items", flush=True)

    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    provider = os.environ.get("LLM_PROVIDER", "vertex").strip().lower()

    boq = extract_boq_parallel(
        sheets_data=sheets_data,
        provider=provider,
        model=model,
        verbose=True,
        chunk_threshold=28000,
        max_concurrent=10,
        reference_items_by_division=reference_items_by_division,
    )

    result = boq.model_dump()
    n_div = len(result.get("divisions", []))
    print(f"Result: {n_div} divisions", flush=True)
    for d in result.get("divisions", []):
        n_items = sum(
            len(sg.get("line_items", []))
            for g in d.get("grouping", [])
            for sg in g.get("sub_groupings", [])
        )
        print(f"  Division '{d.get('division_name')}': {len(d.get('grouping', []))} groupings, {n_items} items", flush=True)
    print(f"{'='*60}\n", flush=True)

    return result


def _guess_division_names_from_summary(summary_text: str) -> list[str]:
    """
    Quickly extract likely division names from a summary sheet's text content
    without using the LLM.

    Handles formats like:
      - "Section B: Site Work | 300221"
      - "SITE WORK"
      - "Bill No. 2 - CONCRETE WORK"
    """
    import re
    found: list[str] = []
    found_upper: set[str] = set()

    for line in summary_text.split("\n"):
        stripped = line.strip().strip("|").strip()
        if not stripped or len(stripped) < 4:
            continue

        # Pattern 1: "Section X: Division Name" (with optional amount after |)
        m = re.match(
            r"^Section\s+[A-Z]\s*:\s*(.+?)(?:\s*\||\s*$)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            name = m.group(0).split("|")[0].strip()  # Keep "Section X: Name"
            if name.upper() not in found_upper:
                found.append(name)
                found_upper.add(name.upper())
            continue

        # Pattern 2: "Bill No. X - DIVISION NAME"
        m = re.match(
            r"^Bill\s*(?:No\.?)?\s*\d+\s*[-–:]\s*(.+?)(?:\s*\||\s*$)",
            stripped,
            re.IGNORECASE,
        )
        if m:
            name = m.group(1).strip()
            if name.upper() not in found_upper:
                found.append(name)
                found_upper.add(name.upper())
            continue

    # Fallback: look for known all-caps division names
    if not found:
        known = [
            "SITE WORK", "SITE WORKS",
            "CONCRETE WORK", "CONCRETE WORKS",
            "MASONRY", "METALWORK", "WOODWORK",
            "THERMAL AND MOISTURE PROTECTION",
            "DOORS AND WINDOWS", "DOOR AND WINDOWS",
            "FINISHES", "ACCESSORIES", "EQUIPMENT",
            "FURNISHING", "SPECIAL CONSTRUCTION",
            "CONVEYING SYSTEM", "CONVEYING INSTALLATIONS",
            "MECHANICAL ENGINEERING", "MECHANICAL ENGINEERING INSTALLATIONS",
            "ELECTRICAL ENGINEERING", "ELECTRICAL ENGINEERING INSTALLATIONS",
            "EXTERNAL WORKS", "PROVISIONAL SUMS",
        ]
        text_upper = summary_text.upper()
        for name in known:
            if name in text_upper and name not in found_upper:
                found.append(name)
                found_upper.add(name)

    return found


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "commercial-evaluation"}


@app.get("/health")
async def health():
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "not set")
    return {"status": "ok", "credentials": creds}


@app.post("/extract")
async def extract_boq(file: UploadFile = File(...)):
    """Extract a BOQ from a single uploaded Excel file."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix
    if suffix.lower() not in (".xlsx", ".xls"):
        raise HTTPException(400, f"Unsupported file type: {suffix}")

    _check_llm_credentials()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        boq_dict = await asyncio.to_thread(_extract_boq_from_file, tmp_path)
        return boq_dict
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Extraction error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/compare")
async def compare_files(
    vendors: list[UploadFile] = File(...),
    template: UploadFile | None = File(None),
    pte: UploadFile | None = File(None),
):
    """
    Accept direct file uploads: optional template + multiple vendor files.
    If no template is provided, the first vendor file is used as the reference.
    Extracts all, compares, returns ComparisonReport.
    """
    if len(vendors) < 2 and template is None:
        raise HTTPException(
            400,
            "At least 2 vendor files are required, "
            "or 1 template + 1 vendor file.",
        )

    _check_llm_credentials()

    tmp_paths: list[str] = []

    try:
        # Save vendor files to temp
        vendor_paths: dict[str, str] = {}
        for vendor_file in vendors:
            name = Path(vendor_file.filename or "vendor.xlsx").stem
            suffix = Path(vendor_file.filename or "vendor.xlsx").suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await vendor_file.read())
                vendor_paths[name] = tmp.name
                tmp_paths.append(tmp.name)

        # Save template or use first vendor as template
        if template is not None and template.filename:
            suffix = Path(template.filename).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await template.read())
                template_path = tmp.name
                tmp_paths.append(template_path)
            template_filename = template.filename
        else:
            first_vendor_name = next(iter(vendor_paths))
            template_path = vendor_paths[first_vendor_name]
            template_filename = first_vendor_name

        # Save PTE file to temp (optional)
        pte_path = None
        if pte is not None and pte.filename:
            suffix = Path(pte.filename).suffix
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(await pte.read())
                pte_path = tmp.name
                tmp_paths.append(pte_path)

        print(f"\n{'='*60}")
        print(f"COMPARE: template={template_filename}")
        print(f"VENDORS: {list(vendor_paths.keys())}")
        if pte_path:
            print(f"PTE: {pte.filename}")
        print(f"{'='*60}\n")

        # Step 1: Extract TEMPLATE first (defines the structure)
        try:
            print(f"Extracting template: {template_filename}")
            template_boq = await asyncio.to_thread(
                _extract_boq_from_file, template_path
            )
            print(f"Template extracted: {len(template_boq.get('divisions', []))} divisions")
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(500, f"Failed to extract template: {e}")

        # Build reference items from template for guided extraction
        ref_items = build_reference_items_from_boq(template_boq)
        if ref_items:
            ref_total = sum(len(v) for v in ref_items.values())
            print(f"Reference map built: {len(ref_items)} divisions, {ref_total} items")

        # Step 2: Extract PTE + vendors in parallel (guided by template)
        pte_boq_dict = None
        if pte_path:
            try:
                print(f"Extracting PTE (with template reference): {pte.filename}")
                pte_sheets_data = _parse_excel_to_sheets(pte_path)
                pte_boq_dict = await asyncio.to_thread(
                    _extract_boq_from_sheets, pte_sheets_data, ref_items
                )
                print(f"PTE extracted: {len(pte_boq_dict.get('divisions', []))} divisions")
            except Exception as e:
                traceback.print_exc()
                print(f"PTE extraction failed (continuing without PTE): {e}")

        async def extract_vendor(name: str, path: str):
            try:
                print(f"Extracting vendor (with template reference): {name}")
                sheets_data = _parse_excel_to_sheets(path)
                boq = await asyncio.to_thread(
                    _extract_boq_from_sheets, sheets_data, ref_items
                )
                print(f"Vendor '{name}' extracted: {len(boq.get('divisions', []))} divisions")
                return name, boq
            except Exception as e:
                traceback.print_exc()
                raise HTTPException(500, f"Failed to extract vendor '{name}': {e}")

        vendor_tasks = [extract_vendor(n, p) for n, p in vendor_paths.items()]
        vendor_results = await asyncio.gather(*vendor_tasks)
        vendor_boqs = {name: boq for name, boq in vendor_results}

        # Compare vendors against template
        report = _compare_boqs(
            template=template_boq,
            vendor_boqs=vendor_boqs,
            template_file=template_filename,
            pte_boq=pte_boq_dict,
        )

        print(f"\nComparison complete: {report.total_line_items} items, {len(report.vendors)} vendors")
        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Comparison error: {e}")
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Streaming comparison with progress events (SSE)
# ---------------------------------------------------------------------------

@app.post("/compare-stream")
async def compare_files_stream(
    vendors: list[UploadFile] = File(...),
    template: UploadFile | None = File(None),
    pte: UploadFile | None = File(None),
):
    """
    Same as /compare but streams progress events as SSE.
    All vendor files are parsed and extracted IN PARALLEL for maximum speed.
    Events: { type, message, ... }
    Final event: { type: "complete", data: ComparisonReport }
    """
    if len(vendors) < 2 and template is None:
        raise HTTPException(
            400,
            "At least 2 vendor files are required, "
            "or 1 template + 1 vendor file.",
        )

    _check_llm_credentials()

    # Read all files into memory before starting the stream
    vendor_data: list[tuple[str, bytes, str]] = []  # (name, content, suffix)
    for vf in vendors:
        name = Path(vf.filename or "vendor.xlsx").stem
        suffix = Path(vf.filename or "vendor.xlsx").suffix
        content = await vf.read()
        vendor_data.append((name, content, suffix))

    template_data: tuple[str, bytes, str] | None = None
    if template is not None and template.filename:
        suffix = Path(template.filename).suffix
        content = await template.read()
        template_data = (template.filename, content, suffix)

    pte_data: tuple[str, bytes, str] | None = None
    if pte is not None and pte.filename:
        suffix = Path(pte.filename).suffix
        content = await pte.read()
        pte_data = (pte.filename, content, suffix)

    async def generate() -> AsyncGenerator[str, None]:
        tmp_paths: list[str] = []
        total_files = len(vendor_data) + (1 if template_data else 0) + (1 if pte_data else 0)
        start_time = time.time()

        def elapsed() -> str:
            return f"{time.time() - start_time:.1f}s"

        try:
            # --- Summarize input files ---
            file_summary = ", ".join(
                f"{name} ({_format_size(len(content))})"
                for name, content, _ in vendor_data
            )
            if pte_data:
                file_summary += f" + PTE: {pte_data[0]} ({_format_size(len(pte_data[1]))})"
            if template_data:
                file_summary = f"Template: {template_data[0]} ({_format_size(len(template_data[1]))}) + " + file_summary

            yield _sse({
                "type": "start",
                "message": f"Starting extraction for {total_files} files — {file_summary}",
                "total_files": total_files,
            })

            # --- Save files to temp (fast) ---
            vendor_paths: dict[str, str] = {}
            vendor_sizes: dict[str, int] = {}
            for name, content, suffix in vendor_data:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    vendor_paths[name] = tmp.name
                    vendor_sizes[name] = len(content)
                    tmp_paths.append(tmp.name)

            template_path = None
            template_filename = None
            if template_data:
                tpl_name, tpl_content, tpl_suffix = template_data
                with tempfile.NamedTemporaryFile(suffix=tpl_suffix, delete=False) as tmp:
                    tmp.write(tpl_content)
                    template_path = tmp.name
                    tmp_paths.append(template_path)
                template_filename = tpl_name
            else:
                first_vendor_name = next(iter(vendor_paths))
                template_path = vendor_paths[first_vendor_name]
                template_filename = first_vendor_name

            pte_path = None
            pte_filename = None
            if pte_data:
                pte_name, pte_content, pte_suffix = pte_data
                with tempfile.NamedTemporaryFile(suffix=pte_suffix, delete=False) as tmp:
                    tmp.write(pte_content)
                    pte_path = tmp.name
                    tmp_paths.append(pte_path)
                pte_filename = pte_name

            # =================================================================
            # PHASE 1: Parse ALL files in parallel
            # =================================================================
            parse_start = time.time()
            yield _sse({
                "type": "phase",
                "phase": "parsing",
                "message": f"[{elapsed()}] Parsing {total_files} Excel files in parallel...",
            })

            # Build parsing tasks
            all_parse_names: list[str] = []
            all_parse_paths: list[str] = []

            if template_data and template_path:
                all_parse_names.append(f"TEMPLATE ({template_filename})")
                all_parse_paths.append(template_path)

            if pte_data and pte_path:
                all_parse_names.append(f"PTE ({pte_filename})")
                all_parse_paths.append(pte_path)

            for name, path in vendor_paths.items():
                all_parse_names.append(name)
                all_parse_paths.append(path)

            # Run all parsing in parallel
            parse_tasks = [
                asyncio.to_thread(_parse_excel_to_sheets, path)
                for path in all_parse_paths
            ]
            parse_results = await asyncio.gather(*parse_tasks, return_exceptions=True)

            # Collect results
            vendor_sheets: dict[str, list[tuple[str, str]]] = {}
            template_sheets = None
            pte_sheets = None

            for i, (pname, result) in enumerate(zip(all_parse_names, parse_results)):
                if isinstance(result, Exception):
                    yield _sse({
                        "type": "error",
                        "message": f"[{elapsed()}] Failed to parse {pname}: {result}",
                    })
                    return

                sheets = result
                total_chars = sum(len(text) for _, text in sheets)
                sheet_detail = ", ".join(
                    f"{sname} ({len(stext):,} chars)"
                    for sname, stext in sheets
                )

                if pname.startswith("TEMPLATE"):
                    template_sheets = sheets
                    yield _sse({
                        "type": "parsed",
                        "vendor": "TEMPLATE",
                        "sheets": len(sheets),
                        "total_chars": total_chars,
                        "sheet_names": [s[0] for s in sheets],
                        "message": f"[{elapsed()}] Parsed template: {len(sheets)} sheets ({total_chars:,} chars) — {sheet_detail}",
                    })
                elif pname.startswith("PTE"):
                    pte_sheets = sheets
                    yield _sse({
                        "type": "parsed",
                        "vendor": "PTE",
                        "sheets": len(sheets),
                        "total_chars": total_chars,
                        "sheet_names": [s[0] for s in sheets],
                        "message": f"[{elapsed()}] Parsed PTE: {len(sheets)} sheets ({total_chars:,} chars) — {sheet_detail}",
                    })
                else:
                    vendor_sheets[pname] = sheets
                    yield _sse({
                        "type": "parsed",
                        "vendor": pname,
                        "sheets": len(sheets),
                        "total_chars": total_chars,
                        "sheet_names": [s[0] for s in sheets],
                        "message": f"[{elapsed()}] Parsed {pname}: {len(sheets)} sheets ({total_chars:,} chars) — {sheet_detail}",
                    })

            parse_elapsed = time.time() - parse_start
            yield _sse({
                "type": "status",
                "message": f"[{elapsed()}] All files parsed in {parse_elapsed:.1f}s",
            })

            # =================================================================
            # PHASE 1b: FAST TOTALS — regex summary extraction for ALL files
            # =================================================================
            yield _sse({
                "type": "phase",
                "phase": "analysis",
                "message": f"[{elapsed()}] Extracting totals from all files (no LLM — instant)...",
            })

            # Collect summary totals for every file (template + PTE + vendors)
            all_file_sheets: list[tuple[str, list[tuple[str, str]] | None]] = [
                ("TEMPLATE", template_sheets),
            ]
            if pte_sheets:
                all_file_sheets.append(("PTE", pte_sheets))
            for vname, vsheets in vendor_sheets.items():
                all_file_sheets.append((vname, vsheets))

            file_project_totals: dict[str, float] = {}
            file_division_totals: dict[str, dict[str, float]] = {}
            all_division_names: list[str] = []

            for label, sheets in all_file_sheets:
                if sheets is None:
                    continue
                summary_found = [s for s in sheets if _is_summary_sheet(s[0])]
                content_found = [s[0] for s in sheets
                                 if _is_content_sheet(s[0])]

                if summary_found:
                    yield _sse({
                        "type": "status",
                        "message": (
                            f"[{elapsed()}] {label}: summary sheet(s) → "
                            f"{', '.join(s[0] for s in summary_found)} | "
                            f"content → {', '.join(content_found)}"
                        ),
                    })

                    for sname, stext in summary_found:
                        p_name, p_total, div_totals = _extract_summary_totals_fast(
                            sname, stext, verbose=True,
                        )
                        if p_total:
                            try:
                                file_project_totals[label] = float(
                                    p_total.replace(",", "")
                                )
                            except (ValueError, TypeError):
                                pass
                        if div_totals:
                            ft = file_division_totals.setdefault(label, {})
                            for dn, dt in div_totals.items():
                                try:
                                    ft[dn] = float(dt.replace(",", ""))
                                except (ValueError, TypeError):
                                    ft[dn] = 0.0
                                if dn not in all_division_names:
                                    all_division_names.append(dn)

                    # Segmentation preview
                    for sname, stext in summary_found:
                        for cname, ctext in sheets:
                            if _is_summary_sheet(cname) or not _is_content_sheet(cname):
                                continue
                            segments = split_content_by_divisions(
                                ctext,
                                _guess_division_names_from_summary(stext),
                                verbose=False,
                            )
                            if len(segments) >= 2:
                                yield _sse({
                                    "type": "status",
                                    "message": (
                                        f"[{elapsed()}] {label}: sheet '{cname}' "
                                        f"contains {len(segments)} divisions"
                                    ),
                                })
                        break
                else:
                    yield _sse({
                        "type": "status",
                        "message": (
                            f"[{elapsed()}] {label}: no summary sheet — "
                            f"{len(content_found)} content sheets"
                        ),
                    })

            # ── Emit summary_ready with totals comparison ────────────
            if file_project_totals or file_division_totals:
                # Build per-division vendor comparison
                div_comparison: dict[str, dict[str, float]] = {}
                for dn in all_division_names:
                    row: dict[str, float] = {}
                    for label, dtotals in file_division_totals.items():
                        if label in ("TEMPLATE", "PTE"):
                            continue
                        val = dtotals.get(dn, 0.0)
                        if val:
                            row[label] = val
                    if row:
                        div_comparison[dn] = row

                # Determine lowest bidder from project totals (vendors only)
                vendor_proj_totals = {
                    k: v for k, v in file_project_totals.items()
                    if k not in ("TEMPLATE", "PTE")
                }
                lowest = min(vendor_proj_totals, key=vendor_proj_totals.get) if vendor_proj_totals else None
                highest = max(vendor_proj_totals, key=vendor_proj_totals.get) if vendor_proj_totals else None

                yield _sse({
                    "type": "summary_ready",
                    "message": (
                        f"[{elapsed()}] Totals extracted for {len(file_project_totals)} files, "
                        f"{len(all_division_names)} divisions — "
                        f"lowest: {lowest}, highest: {highest}"
                    ),
                    "project_totals": {
                        k: v for k, v in file_project_totals.items()
                    },
                    "division_totals": div_comparison,
                    "division_names": all_division_names,
                    "lowest_bidder": lowest,
                    "highest_bidder": highest,
                    "pte_total": file_project_totals.get("PTE"),
                    "template_total": file_project_totals.get("TEMPLATE"),
                })
            else:
                yield _sse({
                    "type": "status",
                    "message": f"[{elapsed()}] No summary sheets found — totals will come from line-item extraction",
                })

            # =================================================================
            # PHASE 2a: Extract TEMPLATE with per-division streaming
            # =================================================================
            extract_start = time.time()
            has_template = template_sheets is not None
            has_pte = pte_sheets is not None
            total_files = len(vendor_sheets) + (1 if has_template else 0) + (1 if has_pte else 0)

            extracted: dict[str, dict] = {}
            tpl_boq: dict | None = None
            pte_boq: dict | None = None
            reference_items_by_division: dict[str, list[dict]] | None = None
            completed_files = 0

            # Get the pre-extracted summary for the template to skip LLM summary
            tpl_pre_summary: tuple | None = None
            if has_template and file_division_totals.get("TEMPLATE"):
                tpl_totals_str = {
                    k: str(v) for k, v in file_division_totals["TEMPLATE"].items()
                }
                tpl_pre_summary = (
                    None,
                    str(file_project_totals.get("TEMPLATE", "")) or None,
                    tpl_totals_str,
                )

            if has_template:
                yield _sse({
                    "type": "phase",
                    "phase": "extraction",
                    "message": f"[{elapsed()}] Extracting TEMPLATE — per-division streaming...",
                })

                model_name = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
                provider_name = os.environ.get("LLM_PROVIDER", "vertex").strip().lower()

                tpl_divisions_list = []
                tpl_div_count = 0

                # Use asyncio.Queue to collect results + heartbeat
                import asyncio as _aio

                result_q: _aio.Queue = _aio.Queue()
                heartbeat_active = True

                async def _run_template_extraction():
                    try:
                        async for dr in extract_divisions_streaming(
                            template_sheets,
                            provider=provider_name,
                            model=model_name,
                            verbose=True,
                            chunk_threshold=28000,
                            max_concurrent=5,
                            pre_extracted_summary=tpl_pre_summary,
                        ):
                            await result_q.put(("division", dr))
                    except Exception as exc:
                        await result_q.put(("error", exc))
                    finally:
                        await result_q.put(("done", None))

                async def _heartbeat():
                    tick = 0
                    while heartbeat_active:
                        await _aio.sleep(8)
                        if heartbeat_active:
                            tick += 1
                            await result_q.put(("heartbeat", tick))

                extraction_task = _aio.ensure_future(_run_template_extraction())
                heartbeat_task = _aio.ensure_future(_heartbeat())

                try:
                    while True:
                        msg_type, payload = await result_q.get()
                        if msg_type == "division":
                            dr = payload
                            tpl_div_count += 1
                            tpl_divisions_list.append(dr.division)
                            yield _sse({
                                "type": "division_extracted",
                                "vendor": "TEMPLATE",
                                "division_name": dr.division_name,
                                "items": dr.item_count,
                                "duration_seconds": dr.duration_seconds,
                                "divisions_done": tpl_div_count,
                                "message": (
                                    f"[{elapsed()}] TEMPLATE: {dr.division_name} — "
                                    f"{dr.item_count} items in {dr.duration_seconds}s "
                                    f"({tpl_div_count} divisions done)"
                                ),
                            })
                        elif msg_type == "heartbeat":
                            wait_s = round(time.time() - extract_start)
                            yield _sse({
                                "type": "status",
                                "message": (
                                    f"[{elapsed()}] TEMPLATE: LLM extraction in progress — "
                                    f"{tpl_div_count} divisions complete so far, "
                                    f"waiting on remaining ({wait_s}s elapsed)..."
                                ),
                            })
                        elif msg_type == "error":
                            traceback.print_exc()
                            yield _sse({
                                "type": "error",
                                "message": f"[{elapsed()}] Template extraction failed: {payload}",
                            })
                            return
                        elif msg_type == "done":
                            break
                finally:
                    heartbeat_active = False
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except _aio.CancelledError:
                        pass

                if tpl_divisions_list:
                    tpl_boq = {
                        "project_name": None,
                        "contractor": None,
                        "project_total_amount": str(file_project_totals.get("TEMPLATE", "")) or None,
                        "divisions": [d.model_dump() for d in tpl_divisions_list],
                    }
                    tpl_total_items = sum(
                        sum(len(sg.line_items) for sg in g.sub_groupings)
                        for d in tpl_divisions_list for g in d.grouping
                    )
                    completed_files += 1

                    # Build reference map from template
                    reference_items_by_division = build_reference_items_from_boq(tpl_boq)
                    ref_total = sum(len(v) for v in reference_items_by_division.values())
                    yield _sse({
                        "type": "extracted",
                        "vendor": "TEMPLATE",
                        "divisions": len(tpl_divisions_list),
                        "items": tpl_total_items,
                        "message": (
                            f"[{elapsed()}] TEMPLATE complete — "
                            f"{len(tpl_divisions_list)} divisions, {tpl_total_items} items. "
                            f"Reference map: {ref_total} items across {len(reference_items_by_division)} divisions."
                        ),
                    })
                else:
                    yield _sse({
                        "type": "status",
                        "message": f"[{elapsed()}] Template extraction returned no divisions",
                    })

            # =================================================================
            # PHASE 2b: Extract PTE + ALL VENDORS in parallel with streaming
            # =================================================================
            remaining = len(vendor_sheets) + (1 if has_pte else 0)
            ref_status = "with template-guided mapping" if reference_items_by_division else "without reference"
            yield _sse({
                "type": "phase",
                "phase": "extraction",
                "message": f"[{elapsed()}] Extracting {remaining} file(s) in parallel ({ref_status})...",
            })

            model_name = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
            provider_name = os.environ.get("LLM_PROVIDER", "vertex").strip().lower()

            # Shared queue for per-division events from all concurrent extractions
            phase2_q: asyncio.Queue = asyncio.Queue()

            async def _stream_file(
                label: str,
                sheets: list[tuple[str, str]],
                ref: dict[str, list[dict]] | None,
            ) -> tuple[str, list]:
                """Extract one file using streaming, pushing events to the shared queue."""
                t0 = time.time()
                divisions = []
                total_chars = sum(len(t) for _, t in sheets)
                await phase2_q.put(_sse({
                    "type": "extracting",
                    "vendor": label,
                    "message": f"[{elapsed()}] Extracting {label} — {len(sheets)} sheets, {total_chars:,} chars...",
                }))

                # Build pre-extracted summary for this file
                pre_summary = None
                ft = file_division_totals.get(label)
                if ft:
                    pre_summary = (
                        None,
                        str(file_project_totals.get(label, "")) or None,
                        {k: str(v) for k, v in ft.items()},
                    )

                # Limit concurrency per vendor to avoid API rate limits
                max_conc = max(3, 10 // max(len(vendor_sheets), 1))

                try:
                    async for dr in extract_divisions_streaming(
                        sheets,
                        provider=provider_name,
                        model=model_name,
                        verbose=True,
                        chunk_threshold=28000,
                        max_concurrent=max_conc,
                        reference_items_by_division=ref,
                        pre_extracted_summary=pre_summary,
                    ):
                        divisions.append(dr.division)
                        await phase2_q.put(_sse({
                            "type": "division_extracted",
                            "vendor": label,
                            "division_name": dr.division_name,
                            "items": dr.item_count,
                            "duration_seconds": dr.duration_seconds,
                            "message": (
                                f"[{elapsed()}] {label}: {dr.division_name} — "
                                f"{dr.item_count} items in {dr.duration_seconds}s"
                            ),
                        }))
                except Exception as e:
                    traceback.print_exc()
                    await phase2_q.put(_sse({
                        "type": "vendor_error",
                        "vendor": label,
                        "message": f"[{elapsed()}] FAILED {label}: {e}",
                    }))
                    return label, []

                dt = time.time() - t0
                total_items = sum(
                    sum(len(sg.line_items) for sg in g.sub_groupings)
                    for d in divisions for g in d.grouping
                )
                await phase2_q.put(_sse({
                    "type": "extracted",
                    "vendor": label,
                    "divisions": len(divisions),
                    "items": total_items,
                    "duration_seconds": round(dt, 1),
                    "message": (
                        f"[{elapsed()}] {label} complete in {dt:.1f}s — "
                        f"{len(divisions)} divisions, {total_items} items"
                    ),
                }))
                return label, divisions

            # Launch all vendor + PTE extractions concurrently
            phase2_coros = []
            if has_pte:
                phase2_coros.append(
                    _stream_file("PTE", pte_sheets, reference_items_by_division)
                )
            for vname, vsheets in vendor_sheets.items():
                phase2_coros.append(
                    _stream_file(vname, vsheets, reference_items_by_division)
                )

            async def _run_phase2():
                results = await asyncio.gather(*phase2_coros)
                await phase2_q.put(None)
                return results

            phase2_future = asyncio.create_task(_run_phase2())

            # Stream events as they arrive
            while True:
                try:
                    event = await asyncio.wait_for(phase2_q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    yield _sse({
                        "type": "status",
                        "message": f"[{elapsed()}] Extraction in progress...",
                    })
                    continue
                if event is None:
                    break
                yield event
                if '"type": "extracted"' in event:
                    completed_files += 1
                    yield _sse({
                        "type": "progress",
                        "completed": completed_files,
                        "total": total_files,
                        "message": f"[{elapsed()}] File progress: {completed_files}/{total_files}",
                    })

            phase2_results = await phase2_future

            for label, div_list in phase2_results:
                if not div_list:
                    continue
                boq_dict = {
                    "project_name": None,
                    "contractor": None,
                    "project_total_amount": str(file_project_totals.get(label, "")) or None,
                    "divisions": [d.model_dump() for d in div_list],
                }
                if label == "PTE":
                    pte_boq = boq_dict
                else:
                    extracted[label] = boq_dict

            extract_elapsed = time.time() - extract_start
            yield _sse({
                "type": "status",
                "message": f"[{elapsed()}] All extractions completed in {extract_elapsed:.1f}s ({len(extracted)} vendors)",
            })

            if not extracted:
                yield _sse({"type": "error", "message": "No vendor files could be extracted"})
                return

            # =================================================================
            # PHASE 2c: Reconciliation — verify line-item sums vs totals
            # =================================================================
            yield _sse({
                "type": "phase",
                "phase": "reconciliation",
                "message": f"[{elapsed()}] Verifying extraction accuracy (2% variance threshold)...",
            })

            all_labels = (
                [("TEMPLATE", tpl_boq)] if tpl_boq else []
            ) + ([("PTE", pte_boq)] if pte_boq else []) + [(n, b) for n, b in extracted.items()]

            total_pass = 0
            total_fail = 0
            for label, boq_dict in all_labels:
                if boq_dict is None:
                    continue
                try:
                    boq_obj = BOQ(**boq_dict)
                    recon_results = reconcile_boq(boq_obj, threshold_pct=2.0, verbose=False)
                    passed = [r for r in recon_results if r.passed]
                    failed = [r for r in recon_results if not r.passed]
                    total_pass += len(passed)
                    total_fail += len(failed)

                    if failed:
                        fail_details = "; ".join(
                            f"{r.division_name} ({r.variance_pct:.1f}%)"
                            for r in failed
                        )
                        yield _sse({
                            "type": "reconciliation",
                            "vendor": label,
                            "passed": len(passed),
                            "failed": len(failed),
                            "total": len(recon_results),
                            "message": (
                                f"[{elapsed()}] {label}: {len(passed)}/{len(recon_results)} divisions within 2% — "
                                f"⚠ {len(failed)} exceed: {fail_details}"
                            ),
                        })
                    else:
                        yield _sse({
                            "type": "reconciliation",
                            "vendor": label,
                            "passed": len(passed),
                            "failed": 0,
                            "total": len(recon_results),
                            "message": f"[{elapsed()}] {label}: ✓ all {len(passed)} divisions within 2%",
                        })
                except Exception as e:
                    yield _sse({
                        "type": "status",
                        "message": f"[{elapsed()}] {label}: reconciliation skipped — {e}",
                    })

            yield _sse({
                "type": "status",
                "message": f"[{elapsed()}] Reconciliation: {total_pass} passed, {total_fail} flagged",
            })

            # =================================================================
            # PHASE 3: Comparison
            # =================================================================
            compare_start = time.time()
            yield _sse({
                "type": "phase",
                "phase": "comparison",
                "message": f"[{elapsed()}] Comparing {len(extracted)} vendor BOQs...",
            })

            if tpl_boq is not None:
                final_template = tpl_boq
            else:
                first_name = next(iter(extracted))
                final_template = extracted[first_name]
                template_filename = first_name

            report = _compare_boqs(
                template=final_template,
                vendor_boqs=extracted,
                template_file=template_filename,
                pte_boq=pte_boq,
            )

            compare_elapsed = time.time() - compare_start
            total_elapsed = time.time() - start_time

            yield _sse({
                "type": "complete",
                "message": (
                    f"Done in {total_elapsed:.0f}s — "
                    f"{report.total_line_items} items across {len(report.vendors)} vendors "
                    f"(parse: {parse_elapsed:.1f}s, extract: {extract_elapsed:.1f}s, compare: {compare_elapsed:.1f}s)"
                ),
                "elapsed_seconds": round(total_elapsed, 1),
                "timing": {
                    "parse": round(parse_elapsed, 1),
                    "extract": round(extract_elapsed, 1),
                    "compare": round(compare_elapsed, 1),
                    "total": round(total_elapsed, 1),
                },
                "data": report.to_dict(),
            })

        except Exception as e:
            traceback.print_exc()
            yield _sse({"type": "error", "message": f"[{elapsed()}] Error: {e}"})
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"
