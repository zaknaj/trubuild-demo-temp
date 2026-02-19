import re, os
import csv
from io import StringIO
from rapidfuzz import fuzz
from utils.core.log import get_logger
from typing import Optional, List, Dict, Any

"""
pip install rapidfuzz
"""


def extract_exact_text(llm_text: str, page_text: str):
    """
    Finds the exact text in the extracted PDF page text that best matches the LLM response.
    Returns the exact match along with its start and end positions. (highlight functionality)
    """
    logger = get_logger()
    normalized_llm_text = normalize_text(llm_text)
    best_match_start = None
    best_match_end = None
    best_exact_match = None
    best_ratio = 0.0

    words = page_text.split()
    llm_words = llm_text.split()

    for i in range(len(words) - len(llm_words) + 1):
        window_text = " ".join(words[i : i + len(llm_words)])

        ratio = fuzz.ratio(normalize_text(window_text), normalized_llm_text)

        if ratio > best_ratio:
            best_ratio = ratio
            best_match_start = page_text.find(window_text)
            best_match_end = best_match_start + len(window_text)
            best_exact_match = window_text

    return best_match_start, best_match_end, best_exact_match


def normalize_text(txt: str) -> str:
    txt = txt.replace("‘", "'").replace("’", "'").replace("'", '"').replace("'", '"')
    txt = re.sub(r"[^\w\s]", "", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip().lower()


def find_exact_text_match(llm_text: str, tokenized_data: dict, threshold: float = 70):
    "Used in contract reviewer to find exact text match (highlight functionality)"
    normalized_llm_text = normalize_text(llm_text)

    best_ratio = 0.0
    best_match = {
        "page_number": float("inf"),
        "document_name": None,
        "exact_text": None,
        "start_index": None,
        "end_index": None,
    }

    for doc_name, pages in tokenized_data.items():
        if not isinstance(pages, list):
            continue

        for page_data in pages:
            if page_data["type"] != "text":
                continue

            page_number = page_data.get("page")
            original_page_text = page_data["content"]
            normalized_page_text = normalize_text(original_page_text)

            ratio = fuzz.partial_ratio(normalized_llm_text, normalized_page_text)

            if ratio > best_ratio or (
                ratio == best_ratio and page_number < best_match["page_number"]
            ):
                best_ratio = ratio

                # Extract the exact text match
                match_start, match_end, exact_match = extract_exact_text(
                    llm_text, original_page_text
                )

                best_match = {
                    "page_number": page_number,
                    "document_name": doc_name,
                    "exact_text": exact_match,
                    "start_index": match_start,
                    "end_index": match_end,
                }

    if best_ratio >= threshold:
        return best_match
    else:
        return None


def extract_tables_as_csv(response_text):
    """
    chat
    Detects Markdown tables in the response text from chat and converts them to CSV format.
    Returns a list of CSV contents as strings if tables are found, else returns an empty list.
    """
    table_pattern = r"(?:\n|^)((?:\|[^\n]+\|\n)+\|[-:]+[-|\s:]*\|\n(?:\|[^\n]+\|\n?)+)"
    matches = re.findall(table_pattern, response_text)

    csv_tables = []
    for match in matches:
        lines = match.strip().split("\n")
        header_line = lines[0]
        data_lines = lines[2:]

        header_columns = [
            col.strip() for col in header_line.strip().split("|") if col.strip()
        ]
        rows = []

        for row_line in data_lines:
            if row_line.strip() == "":
                continue
            columns = [
                col.strip() for col in row_line.strip().split("|") if col.strip()
            ]
            if columns:
                rows.append(columns)

        output = StringIO()
        csv_writer = csv.writer(output)
        csv_writer.writerow(header_columns)
        csv_writer.writerows(rows)
        csv_tables.append(output.getvalue())

    return csv_tables


def fuzzy_search_tech_rfp(
    llm_text: str, file_names: List[str], threshold: float = 70
) -> Optional[str]:
    """
    tech_rfp
    Matches LLM-generated file names to actual contractor files using fuzzy matching.
    Handles spacing, casing, punctuation inconsistencies.
    Returns best match above threshold or logs low-confidence if between 50-70.
    """
    logger = get_logger()
    if not llm_text or not file_names:
        logger.debug(
            "No LLM text or file names provided for fuzzy_search. Returning None."
        )
        return None

    def normalize(text):
        # Normalize for fair comparison
        text = (
            text.replace("’", "'").replace("‘", "'").replace("'", '"').replace("'", '"')
        )
        text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
        return text.strip().lower()

    normalized_llm_text = normalize(llm_text)
    normalized_map = {normalize(name): name for name in file_names}

    if normalized_llm_text in normalized_map:
        return normalized_map[normalized_llm_text]

    best_match = None
    best_score = 0

    for original_name in file_names:
        normalized_name = normalize(original_name)
        score = max(
            fuzz.ratio(normalized_llm_text, normalized_name),
            fuzz.partial_ratio(normalized_llm_text, normalized_name),
            fuzz.token_sort_ratio(normalized_llm_text, normalized_name),
        )

        if score > best_score:
            best_score = score
            best_match = original_name

    if best_score >= threshold:
        return best_match

    if best_score >= 50:
        logger.debug(
            f"Low-confidence fuzzy match ({best_score}%) for '{llm_text}' -> '{best_match}'"
        )
        return best_match

    logger.debug(
        f"No match found above threshold for '{llm_text}'. Best guess: '{best_match}' ({best_score}%)"
    )
    return None


def promote_fuzzy_key(
    payload: Dict[str, Any],
    canonical: str,
    *,
    score_threshold: int = 70,
) -> None:
    """
    tech rfp
    Mutates payload in‑place:

    - If canonical already exists -> nothing happens.
    - Otherwise finds the key with the highest fuzzy‑ratio to canonical.
      When the ratio >= score_threshold, that key is renamed to canonical.

    Raises
    KeyError
        If no suitable match is found.
    """
    if canonical in payload:
        return

    best_key, best_ratio = None, 0
    for k in payload:
        ratio = fuzz.ratio(k.lower(), canonical.lower())
        if ratio > best_ratio:
            best_key, best_ratio = k, ratio

    if best_key and best_ratio >= score_threshold:
        payload[canonical] = payload.pop(best_key)
    else:
        raise KeyError(f"No fuzzy match for '{canonical}' (best_ratio={best_ratio})")


def normalize_evidence_sources(
    evidence: List[Dict[str, Any]],
    file_names_or_doc_index: List[Any],
    contractor_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    tech_rfp
    - file_names_or_doc_index is List[dict] of doc metadata:
        { "display", "location", "url", "context", "contractor" }
    - Fuzzy matches LLM 'source' against 'display' and attaches:
        source, sourceUrl, sourceContext, sourceLocation, sourceContractor.
    """
    logger = get_logger()
    if not evidence or not file_names_or_doc_index:
        return evidence

    first = file_names_or_doc_index[0]

    doc_index: List[Dict[str, Any]] = file_names_or_doc_index

    def norm_name(name: str) -> str:
        # Lowercase, strip, and remove extension for more robust matching
        base = os.path.splitext(name.strip().lower())[0]
        return base

    # Build a map from normalized display name -> list of doc entries
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for doc in doc_index:
        display = str(doc.get("display") or "").strip()
        if not display:
            continue
        key = norm_name(display)
        by_name.setdefault(key, []).append(doc)

    def pick_best_doc(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prefer tender docs for this contractor
        if contractor_name:
            tender_for_contractor = [
                d
                for d in candidates
                if d.get("context") == "tender"
                and d.get("contractor") == contractor_name
            ]
            if tender_for_contractor:
                return tender_for_contractor[0]

        # Otherwise any tender doc
        tender_any = [d for d in candidates if d.get("context") == "tender"]
        if tender_any:
            return tender_any[0]

        # Otherwise first RFP or just first candidate
        return candidates[0]

    all_display_names = [d["display"] for d in doc_index if d.get("display")]

    for item in evidence:
        original_source = str(item.get("source") or "").strip()
        if not original_source:
            continue

        # Try simple normalized exact match first
        key = norm_name(original_source)
        candidates = by_name.get(key)

        # If no direct hit, fuzzy match against all 'display' names
        if not candidates and all_display_names:
            best_match = fuzzy_search_tech_rfp(original_source, all_display_names)
            if best_match:
                key = norm_name(best_match)
                candidates = by_name.get(key)

        if not candidates:
            logger.debug(
                "No doc match found for evidence source '%s'. Leaving as is.",
                original_source,
            )
            continue

        best = pick_best_doc(candidates)
        display = best.get("display")
        url = best.get("url")
        context = best.get("context")
        location = best.get("location")
        contractor = best.get("contractor")

        logger.debug(
            "Evidence source normalized: original='%s' -> display='%s', url='%s', "
            "context='%s', contractor='%s', location='%s'",
            original_source,
            display,
            url,
            context,
            contractor,
            location,
        )

        if url:
            item["source"] = url
        else:
            # fallback: keep original if somehow url missing
            item["source"] = original_source

    return evidence


def main():
    from utils.core.log import pid_tool_logger, set_logger

    package_id = "system_check"
    logger = pid_tool_logger(package_id, "fuzzy_search")
    set_logger(logger)

    print("FUZZY_SEARCH TEST START")

    try:
        # 1. Test: fuzzy_search_tech_rfp
        llm_input = "tech specs final"
        file_list = ["Tech_Specs_Final.pdf", "budget.pdf"]
        match = fuzzy_search_tech_rfp(llm_input, file_list)

        # 2. Test: extract_tables_as_csv
        markdown = """
                | Name | Age | Role |
                |------|-----|------|
                | John | 30  | Dev  |
                | Jane | 25  | PM   |
                """
        tables = extract_tables_as_csv(markdown)

        # 3. Test: promote_fuzzy_key
        payload = {"tehcnical": "data", "budget": "1000"}
        try:
            promote_fuzzy_key(payload, "technical")
        except KeyError as e:
            print("promote_fuzzy_key KeyError:", e)

        # 4. Test: normalize_evidence_sources
        evidence = [{"source": "tech specs final"}, {"source": "budget"}]
        normalized = normalize_evidence_sources(evidence, file_list)

        # 5. Test: find_exact_text_match
        tokenized_data = {
            "doc1.pdf": [
                {
                    "page": 1,
                    "type": "text",
                    "content": "This is a technical specification document.",
                }
            ]
        }
        match_info = find_exact_text_match("technical specification", tokenized_data)

        # 6. Test: extract_exact_text
        page_text = "This is a technical specification document for project Alpha."
        llm_text = "technical specification document"
        start, end, match = extract_exact_text(llm_text, page_text)

        print("FUZZY SEARCH OK")
        return True
    except Exception as e:
        print("FUZZY SEARCH ERROR")
        raise


if __name__ == "__main__":
    main()
