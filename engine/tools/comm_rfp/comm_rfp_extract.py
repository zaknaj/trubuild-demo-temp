"""
LLM-based BOQ field extraction and mapping module.

This module uses Large Language Models to intelligently extract
and map BOQ data from unstructured text to the schema.json format.

Supported LLM providers:
- OpenAI (GPT-4, GPT-4-turbo, GPT-3.5-turbo)
- Anthropic (Claude 3)
- Google (Gemini Pro)

The LLM approach is superior to regex for BOQ extraction because:
- Handles varied document formats
- Understands context and semantics
- Can process tables and multi-line descriptions
- Adapts to different naming conventions

Usage:
    from llm_extractor import LLMExtractor
    
    extractor = LLMExtractor(provider="openai", model="gpt-4-turbo")
    boq = extractor.extract(text)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from tools.comm_rfp.comm_rfp_models import BOQ, Division, Grouping, SubGrouping, LineItem


# ---------------------------------------------------------------------------
# Sheet classification helpers
# ---------------------------------------------------------------------------

# Keywords that identify a summary sheet (substring match, case-insensitive)
_SUMMARY_KEYWORDS = ["SUMMARY", "BILL 1-SUM", "TOTALS", "BILL SUMMARY"]

# Keywords that identify sheets to skip entirely
_SKIP_KEYWORDS = ["FLY", "COVER", "INDEX", "PREAMBLE"]


def _is_summary_sheet(sheet_name: str) -> bool:
    """Check if a sheet name looks like a summary sheet (substring match)."""
    name_upper = sheet_name.upper().strip()
    return any(kw in name_upper for kw in _SUMMARY_KEYWORDS)


def _is_skip_sheet(sheet_name: str) -> bool:
    """Check if a sheet should be skipped entirely (cover pages, etc.)."""
    name_upper = sheet_name.upper().strip()
    return any(kw in name_upper for kw in _SKIP_KEYWORDS)


def _is_content_sheet(sheet_name: str) -> bool:
    """Check if a sheet contains extractable BOQ content."""
    return not _is_summary_sheet(sheet_name) and not _is_skip_sheet(sheet_name)


def _extract_carried_total(segment_text: str) -> str | None:
    """
    Extract the "Total - Carried To Bill Summary" amount from a division segment.
    This is the authoritative total for the division — no calculation needed.

    Returns the amount as a string, or None if not found.
    """
    # Search from the END of the text (the total is at the bottom)
    for line in reversed(segment_text.split("\n")):
        line_lower = line.lower().strip()
        if "carried to bill summary" in line_lower or "carried to general summary" in line_lower:
            # Extract the number after the last | or :
            parts = line.split("|")
            for part in reversed(parts):
                cleaned = part.strip().replace(",", "").replace(" ", "")
                if cleaned and re.match(r"^-?\d+\.?\d*$", cleaned):
                    return cleaned
            # Try after ":"
            parts = line.split(":")
            for part in reversed(parts):
                cleaned = part.strip().replace(",", "").replace(" ", "")
                if cleaned and re.match(r"^-?\d+\.?\d*$", cleaned):
                    return cleaned
    return None


def _count_items_in_grouping(grouping: Grouping) -> int:
    """Count total line items in a grouping (across all sub-groupings)."""
    return sum(len(sg.line_items) for sg in grouping.sub_groupings)


def _iter_items_in_grouping(grouping: Grouping):
    """Iterate over all line items in a grouping."""
    for sg in grouping.sub_groupings:
        for item in sg.line_items:
            yield item


def _fix_truncated_json(response: str) -> str | None:
    """
    Attempt to fix truncated JSON by closing open brackets/braces.
    Used when the LLM response is cut off (e.g. token limit).
    """
    open_braces = response.count("{") - response.count("}")
    open_brackets = response.count("[") - response.count("]")
    if open_braces < 0 or open_brackets < 0:
        return None
    last_complete = max(
        response.rfind("},"),
        response.rfind("}]"),
        response.rfind("}"),
        response.rfind("]"),
    )
    if last_complete == -1:
        return None
    truncated = response[: last_complete + 1].rstrip().rstrip(",")
    open_braces = truncated.count("{") - truncated.count("}")
    open_brackets = truncated.count("[") - truncated.count("]")
    truncated += "]" * open_brackets + "}" * open_braces
    try:
        json.loads(truncated)
        return truncated
    except json.JSONDecodeError:
        return None


def _parse_llm_json_response(response: str) -> tuple[dict | None, str | None]:
    """
    Robustly parse LLM JSON response. Tries: direct load, regex extract, fix truncated.
    Returns (data, None) on success or (None, error_message) on failure.
    """
    if not response or not response.strip():
        return None, "Empty response"
    text = response.strip()
    # 1) Direct parse
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    # 2) Extract first {...} (handle markdown or extra text)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group()), None
        except json.JSONDecodeError:
            pass
    # 3) Fix truncated JSON
    fixed = _fix_truncated_json(text)
    if fixed:
        try:
            return json.loads(fixed), None
        except json.JSONDecodeError:
            pass
    return None, "Invalid or truncated JSON"


def _sanitize_sheet_text(sheet_text: str, max_chars: int = 500_000) -> str:
    """Normalize sheet text for extraction: BOM, line endings, length cap."""
    if not sheet_text:
        return ""
    s = sheet_text.replace("\r\n", "\n").replace("\r", "\n")
    if s.startswith("\ufeff"):
        s = s[1:]
    s = s.strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars] + "\n\n[... truncated for extraction ...]"
    return s


# Load schema for prompt
SCHEMA_PATH = Path(__file__).parent.parent / "schema.json"


def load_schema() -> dict:
    """Load the BOQ schema from schema.json."""
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def get_extraction_prompt(text: str, schema: dict) -> str:
    """
    Generate the extraction prompt for the LLM.

    Args:
        text: Raw text extracted from the BOQ document
        schema: JSON schema for the output format

    Returns:
        Formatted prompt string
    """
    return f'''You are a construction document expert specializing in Bill of Quantities (BOQ) extraction.

Your task is to extract structured data from the following BOQ document text and return valid JSON that conforms EXACTLY to the provided schema.

## SCHEMA (you must follow this exactly):
```json
{json.dumps(schema, indent=2)}
```

## FIELD DEFINITIONS:

**Project Level:**
- `project_name`: The official project name/title as stated in the document
- `contractor`: The contractor, bidder, or company name
- `project_total_amount`: The total contract value (as a STRING, e.g., "1250000.00")

**Division Level:**
- `division_name`: Major trade/scope name (e.g., "SITE WORK", "CONCRETE WORK", "MASONRY")
- `division_total_amount`: Sum of all items in this division (as a STRING)
- `grouping`: Array of sub-categories within this division

**Grouping Level:**
- `grouping_name`: Sub-category name (e.g., "Site Preparation", "Foundations")
- `line_items`: Array of individual BOQ items

**Line Item Level:**
- `item_id`: Item reference number/code from the document (generate if not present, e.g., "1.1", "A.1")
- `item_description`: Full description of work, material, or service
- `item_quantity`: Numeric quantity (as NUMBER, e.g., 100.5)
- `item_unit`: Unit of measurement (e.g., "m", "m²", "m³", "pcs", "lot", "kg")
- `item_rate`: Unit rate/price (as NUMBER, e.g., 25.50)

## EXTRACTION RULES:

1. **Structure Detection:**
   - Look for section headers, division numbers, or trade names for divisions
   - Look for sub-headers or numbered sections for groupings
   - Look for line items with quantities, units, and rates

2. **Number Formatting:**
   - `item_quantity` and `item_rate` must be NUMBERS (not strings)
   - `project_total_amount` and `division_total_amount` must be STRINGS
   - Remove thousand separators when parsing numbers

3. **Missing Values:**
   - Use `null` for any field that cannot be determined
   - Generate sequential item_ids if not present (e.g., "ITM-001", "ITM-002")
   - Create "UNSPECIFIED" division/grouping if structure is unclear

4. **Accuracy:**
   - Extract exact values from the document, do not invent data
   - Preserve original descriptions and item codes
   - Include all line items found in the document

## DOCUMENT TEXT:
```
{text}
```

## OUTPUT:
Return ONLY valid JSON conforming to the schema. No explanations, no markdown, just the JSON object.
'''


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Send prompt to LLM and get response."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, model: str = "gpt-4-turbo", api_key: str | None = None):
        """
        Initialize OpenAI provider.

        Args:
            model: Model name (gpt-4-turbo, gpt-4, gpt-3.5-turbo)
            api_key: API key (defaults to OPENAI_API_KEY env var)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: pip install openai"
            )

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable not set"
            )

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Send prompt to OpenAI and get response."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise JSON extraction assistant. Return only valid JSON."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    def __init__(self, model: str = "claude-3-sonnet-20240229", api_key: str | None = None):
        """
        Initialize Anthropic provider.

        Args:
            model: Model name (claude-3-opus, claude-3-sonnet, claude-3-haiku)
            api_key: API key (defaults to ANTHROPIC_API_KEY env var)
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set"
            )

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model

    def complete(self, prompt: str) -> str:
        """Send prompt to Anthropic and get response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider."""

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        """
        Initialize Google Gemini provider.

        Args:
            model: Model name (gemini-2.5-flash, gemini-2.5-pro)
            api_key: API key (defaults to GOOGLE_API_KEY env var)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package required. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable not set"
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def complete(self, prompt: str) -> str:
        """Send prompt to Gemini and get response."""
        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )
        return response.text


class ServiceAccountProvider(BaseLLMProvider):
    """Google Gemini provider using service account authentication."""

    def __init__(self, model: str = "gemini-3-flash-preview", api_key: str | None = None):
        """
        Initialize Gemini provider with service account.

        Args:
            model: Model name (gemini-2.0-flash, gemini-1.5-pro, etc.)
            api_key: Not used - uses GOOGLE_APPLICATION_CREDENTIALS env var

        Note:
            Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file path.
        """
        try:
            import google.generativeai as genai
            from google.oauth2 import service_account
        except ImportError:
            raise ImportError(
                "google-generativeai and google-auth packages required. "
                "Install with: pip install google-generativeai google-auth"
            )

        credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
                "Set it to the path of your service account JSON file."
            )

        # Load credentials from service account
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/generative-language']
        )

        # Configure the client with credentials
        genai.configure(credentials=credentials)
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    _TRANSIENT_KEYWORDS = (
        "504", "503", "429", "deadline", "timed out", "timeout",
        "unavailable", "resource exhausted", "rate limit",
        "stream cancelled", "connection reset", "internal error",
    )

    @staticmethod
    def _is_transient(err: Exception) -> bool:
        """Return True if *err* looks like a transient / retryable API error."""
        msg = str(err).lower()
        return any(kw in msg for kw in ServiceAccountProvider._TRANSIENT_KEYWORDS)

    def complete(self, prompt: str, max_retries: int = 3) -> str:
        """Send prompt to Gemini with exponential backoff on transient errors.

        Retries up to *max_retries* times for 504 / 429 / connection errors,
        using exponential backoff with jitter to avoid thundering-herd.
        """
        import random
        import time as _time

        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 65536,
                        "response_mime_type": "application/json",
                    },
                    request_options={"timeout": 180},
                )
                return response.text
            except Exception as e:
                last_err = e
                if self._is_transient(e) and attempt < max_retries - 1:
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    print(
                        f"  ⟳ Transient error (attempt {attempt + 1}/{max_retries}): "
                        f"{str(e)[:120]}… retrying in {delay:.1f}s"
                    )
                    _time.sleep(delay)
                    continue
                raise
        raise last_err  # type: ignore[misc]


class LLMExtractor:
    """
    LLM-based BOQ extractor.

    Uses a Large Language Model to extract structured BOQ data
    from unstructured text and map it to the schema.json format.
    """

    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "vertex": ServiceAccountProvider,
    }

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None
    ):
        """
        Initialize the LLM extractor.

        Args:
            provider: LLM provider ("openai", "anthropic", "google")
            model: Model name (provider-specific, uses default if None)
            api_key: API key (uses env var if None)

        Raises:
            ValueError: If provider is not supported
        """
        if provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: {list(self.PROVIDERS.keys())}"
            )

        # Default models per provider
        default_models = {
            "openai": "gpt-4-turbo",
            "anthropic": "claude-3-sonnet-20240229",
            "google": "gemini-2.5-flash",
            "vertex": "gemini-2.5-flash",
        }

        self.provider_name = provider
        self.model_name = model or default_models[provider]

        # Initialize provider
        provider_class = self.PROVIDERS[provider]
        self.provider = provider_class(model=self.model_name, api_key=api_key)

        # Load schema
        self.schema = load_schema()

    def extract(self, text: str, verbose: bool = False) -> BOQ:
        """
        Extract BOQ data from text using the LLM.

        Args:
            text: Raw text from the BOQ document
            verbose: Print debug information

        Returns:
            BOQ object conforming to schema.json

        Raises:
            ValueError: If LLM response is not valid JSON
            ValidationError: If response doesn't conform to schema
        """
        # Generate prompt
        prompt = get_extraction_prompt(text, self.schema)
        
        if verbose:
            print(f"Prompt length: {len(prompt)} characters")

        # Get LLM response
        response = self.provider.complete(prompt)
        
        if verbose:
            print(f"Response length: {len(response)} characters")
            print(f"Response preview: {response[:500]}...")

        # Parse JSON response
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            # Try to extract JSON from response (sometimes wrapped in markdown)
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try to fix common JSON issues - truncated arrays/objects
                    fixed_response = self._fix_truncated_json(response)
                    if fixed_response:
                        data = json.loads(fixed_response)
                    else:
                        raise ValueError(f"LLM returned invalid JSON: {e}")
            else:
                raise ValueError(f"LLM returned invalid JSON: {e}")
        
        if verbose:
            print(f"Parsed data keys: {data.keys() if isinstance(data, dict) else 'NOT A DICT'}")
            if isinstance(data, dict) and 'divisions' in data:
                print(f"Number of divisions: {len(data.get('divisions', []))}")

        # Convert to Pydantic models (validates against schema)
        return self._dict_to_boq(data)

    def extract_with_retry(self, text: str, max_retries: int = 3, verbose: bool = False) -> BOQ:
        """
        Extract BOQ data with retry on failure.

        Args:
            text: Raw text from the BOQ document
            max_retries: Maximum number of retry attempts
            verbose: Print debug information

        Returns:
            BOQ object conforming to schema.json
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                if verbose:
                    print(f"Extraction attempt {attempt + 1}/{max_retries}")
                return self.extract(text, verbose=verbose)
            except (ValueError, json.JSONDecodeError) as e:
                last_error = e
                if verbose:
                    print(f"Attempt {attempt + 1} failed: {e}")
                continue

        raise ValueError(f"Failed after {max_retries} attempts: {last_error}")

    def _fix_truncated_json(self, response: str) -> str | None:
        """
        Attempt to fix truncated JSON by closing open brackets/braces.
        
        Args:
            response: Potentially truncated JSON string
            
        Returns:
            Fixed JSON string or None if unfixable
        """
        # Count open brackets and braces
        open_braces = response.count('{') - response.count('}')
        open_brackets = response.count('[') - response.count(']')
        
        if open_braces < 0 or open_brackets < 0:
            return None  # More closing than opening - can't fix
        
        # Try to close at a reasonable point
        # Find the last complete item (ends with } or ])
        # Remove trailing incomplete content after last complete structure
        # Look for patterns like: }, or ] followed by incomplete data
        last_complete = max(
            response.rfind('},'),
            response.rfind('}]'),
            response.rfind('}'),
            response.rfind(']'),
        )
        
        if last_complete == -1:
            return None
            
        # Truncate at the last complete point
        truncated = response[:last_complete + 1]
        
        # Remove trailing comma if present
        truncated = truncated.rstrip().rstrip(',')
        
        # Close remaining open brackets/braces
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        
        # Add closing brackets in correct order (] before } typically)
        truncated += ']' * open_brackets
        truncated += '}' * open_braces
        
        try:
            json.loads(truncated)
            return truncated
        except json.JSONDecodeError:
            return None

    def _dict_to_boq(self, data: dict) -> BOQ:
        """
        Convert dictionary to BOQ Pydantic model.

        Args:
            data: Dictionary from LLM response

        Returns:
            Validated BOQ object
        """
        # Handle case where LLM returns a list (divisions array) instead of object
        if isinstance(data, list):
            data = {"divisions": data}
        
        # Build line items
        divisions = []
        for div_data in data.get("divisions", []):
            groupings = []
            for grp_data in div_data.get("grouping", []):
                sub_groupings = []
                
                # Check for new format with sub_groupings
                sub_groupings_data = grp_data.get("sub_groupings", [])
                
                if sub_groupings_data:
                    # New format with sub_groupings
                    for sub_grp_data in sub_groupings_data:
                        line_items = []
                        for item_data in sub_grp_data.get("line_items", []):
                            line_items.append(LineItem(
                                item_id=item_data.get("item_id"),
                                item_description=item_data.get("item_description"),
                                item_quantity=item_data.get("item_quantity"),
                                item_unit=item_data.get("item_unit"),
                                item_rate=item_data.get("item_rate"),
                                item_rate_raw=item_data.get("item_rate_raw"),
                                item_amount=item_data.get("item_amount"),
                                item_amount_raw=item_data.get("item_amount_raw"),
                            ))
                        sub_groupings.append(SubGrouping(
                            sub_grouping_name=sub_grp_data.get("sub_grouping_name") or "GENERAL",
                            line_items=line_items,
                        ))
                else:
                    # Old format - wrap line_items in a default sub_grouping
                    line_items = []
                    for item_data in grp_data.get("line_items", []):
                        line_items.append(LineItem(
                            item_id=item_data.get("item_id"),
                            item_description=item_data.get("item_description"),
                            item_quantity=item_data.get("item_quantity"),
                            item_unit=item_data.get("item_unit"),
                            item_rate=item_data.get("item_rate"),
                            item_rate_raw=item_data.get("item_rate_raw"),
                            item_amount=item_data.get("item_amount"),
                            item_amount_raw=item_data.get("item_amount_raw"),
                        ))
                    if line_items:
                        sub_groupings.append(SubGrouping(
                            sub_grouping_name="GENERAL",
                            line_items=line_items,
                        ))

                groupings.append(Grouping(
                    grouping_name=grp_data.get("grouping_name"),
                    grouping_total_amount=grp_data.get("grouping_total_amount"),
                    sub_groupings=sub_groupings,
                ))

            divisions.append(Division(
                division_name=div_data.get("division_name"),
                division_total_amount=div_data.get("division_total_amount"),
                grouping=groupings,
            ))

        return BOQ(
            project_name=data.get("project_name"),
            contractor=data.get("contractor"),
            project_total_amount=data.get("project_total_amount"),
            divisions=divisions,
        )


def extract_boq_with_llm(
    text: str,
    provider: str = "openai",
    model: str | None = None
) -> BOQ:
    """
    Convenience function to extract BOQ using LLM.

    Args:
        text: Raw text from BOQ document
        provider: LLM provider name
        model: Model name (optional)

    Returns:
        Extracted BOQ object
    """
    extractor = LLMExtractor(provider=provider, model=model)
    return extractor.extract(text)


def _build_reference_block(reference_items: list[dict] | None) -> str:
    """Build a lightweight reference summary from template items.

    Instead of listing every item (which bloats prompts and causes API timeouts),
    this provides a structural summary: grouping names and item counts so the LLM
    knows the expected hierarchy without massive token overhead.
    """
    if not reference_items:
        return ""

    # Build a grouping → sub_grouping → count structure
    from collections import OrderedDict
    structure: OrderedDict[str, OrderedDict[str, int]] = OrderedDict()
    for it in reference_items:
        grp = it.get("grouping_name") or "GENERAL"
        sub = it.get("sub_grouping_name") or "GENERAL"
        if grp not in structure:
            structure[grp] = OrderedDict()
        structure[grp][sub] = structure[grp].get(sub, 0) + 1

    total_items = len(reference_items)
    lines = [
        f"TEMPLATE STRUCTURE REFERENCE ({total_items} line items expected):",
        "",
    ]
    for grp_name, subs in structure.items():
        grp_total = sum(subs.values())
        lines.append(f"  Grouping: {grp_name} ({grp_total} items)")
        for sub_name, count in subs.items():
            if sub_name != "GENERAL" or len(subs) > 1:
                lines.append(f"    Sub-grouping: {sub_name} ({count} items)")
    lines.append("")
    lines.append(f"IMPORTANT: Extract ALL {total_items} line items to match the template structure.")
    lines.append("Use the SAME grouping names and sub-grouping names as listed above.")
    lines.append("If an item has no price in this BOQ, include it with null rate/amount.")
    lines.append("")
    return "\n".join(lines)


def get_sheet_extraction_prompt(
    sheet_text: str,
    sheet_name: str,
    schema: dict = None,
    reference_items: list[dict] | None = None,
) -> str:
    """
    Generate a prompt for extracting a single division/sheet.
    Uses a focused prompt format for sheet extraction.
    
    Args:
        sheet_text: Text content from a single sheet
        sheet_name: Name of the sheet (usually the division name)
        schema: JSON schema (unused, kept for compatibility)
        reference_items: Optional list of template items for this division
            to guide vendor extraction mapping
        
    Returns:
        Formatted prompt string
    """
    # Electrical: fixed groupings so structure is consistent (was giving good results)
    electrical_guidance = ""
    if "ELECTRICAL" in sheet_name.upper():
        electrical_guidance = """
IMPORTANT FOR ELECTRICAL ENGINEERING:
Use ONLY these high-level groupings (in ALL CAPS):
- SUB-MAIN INSTALLATION
- SMALL POWER INSTALLATION
- LIGHTING INSTALLATION
- EARTHING AND BONDING SYSTEM
- FIRE ALARM AND VOICE EVACUATION INSTALLATION
- ICT INSTALLATION
- INTERCOM INSTALLATION
- SUNDRIES TO ELECTRICAL ENGINEERING INSTALLATIONS
- WORK INCIDENTAL TO ELECTRICAL ENGINEERING INSTALLATIONS
- GENERALLY

Do NOT create sub-groupings for cable types, conduit sizes, or individual equipment. Keep items under their main section header.
"""

    ref_block = _build_reference_block(reference_items)
    
    return f'''You are a construction BOQ expert. Extract ALL line items from this sheet with proper hierarchy.

SHEET: {sheet_name}

Return JSON with this EXACT structure (note the 4-level hierarchy: Division → Grouping → Sub-Grouping → Line Items):
{{
  "division_name": "ACTUAL DIVISION NAME FROM CONTENT (e.g., SITE WORK, CONCRETE WORK, MASONRY - NOT the sheet code like 3A-B)",
  "division_total_amount": "123456.00",
  "grouping": [
    {{
      "grouping_name": "SECTION NAME IN CAPS",
      "grouping_total_amount": "12345.00",
      "sub_groupings": [
        {{
          "sub_grouping_name": "Work category description",
          "line_items": [
            {{"item_id": "A", "item_description": "desc", "item_quantity": 10.0, "item_unit": "m²", "item_rate": 25.5, "item_rate_raw": null, "item_amount": 255.0, "item_amount_raw": null}},
            {{"item_id": "B", "item_description": "Provisional sum", "item_quantity": null, "item_unit": "Item", "item_rate": null, "item_rate_raw": "Included", "item_amount": null, "item_amount_raw": "Included"}}
          ]
        }}
      ]
    }}
  ]
}}

{ref_block}RULES:
- CRITICAL: division_name must be the ACTUAL TRADE/SCOPE NAME from the content header (e.g., "SITE WORK", "CONCRETE WORK", "MASONRY", "WOODWORK", "ELECTRICAL ENGINEERING"), NOT the sheet code/number (like "3A-B", "#A-B", "Bill 1"). Look for the division title near the top of the content.
- Sub-groupings are descriptive headers that appear BEFORE a set of related items (use "GENERAL" if no clear sub-grouping exists)
- Extract EVERY row with item codes (A, B, C, D, E, F, G, H, J, K, L, M, N, etc.) - even if the description is just dimensions/sizes
- ALSO extract rows WITHOUT item codes if they have amounts - especially in GENERALLY sections (e.g., "Additional items", "Provisional sum", "Contingency")
- IMPORTANT: Items with descriptions like "Overall size; 1285mm long x 550mm wide x 550mm high" ARE valid items - extract them!
- CRITICAL: When there are MANY similar items (like multiple "Overall size XXXxYYYmm" or "100mm thick; X.XXm high"), extract EACH ONE separately - do NOT summarize or skip any!
- WOODWORK/JOINERY: Extract ALL sub-items (A, B, C, D, E, F, G, H...) - e.g., "G | Overall size; 1285mm long x 550mm wide | 1 | Nr. | 4821 | 4821" is item G with amount 4821!
- ACCESSORIES: Extract ALL items with product type codes like "Type SA2-SW20; flush plate" or "Type SA2-SW50; toilet roll holder" - these have codes A, B, C with amounts!
- item_quantity, item_rate, and item_amount must be numbers (not strings) when they are numeric values
- item_amount is the total amount for the line item (usually in the last column, labeled "Amount" or "Total")
- CRITICAL: ALWAYS extract item_amount DIRECTLY from the Amount/Total column as-is. Do NOT recalculate it from qty × rate.
  In construction BOQs, item_amount often does NOT equal qty × rate (lump sums, provisional sums, rates that include multiple components).
  Trust the Amount column value even when it differs from qty × rate.
- ACCURACY CHECK: The SUM of all item_amount values in a grouping MUST closely match grouping_total_amount. The SUM across all groupings MUST closely match division_total_amount. Double-check arithmetic — a 1-2% variance is acceptable but aim for < 0.5%.
- For provisional sums or allowances, extract the amount even if qty/rate are null
- Use null for missing values (use null for item_id if no code exists)
- SPECIAL VALUES: If rate or amount contains text like "Included", "Excluded", "N/A", "-", or any non-numeric value:
  * Set item_rate/item_amount to null
  * Set item_rate_raw/item_amount_raw to the exact text value (e.g., "Included", "Excluded", "N/A")
- If rate/amount is a normal number, set item_rate_raw/item_amount_raw to null
- Groupings are the MAIN SECTION HEADERS (usually in ALL CAPS like "SITE PREPARATION", "EARTHWORKS", "POURED CONCRETE", "REINFORCEMENT", "SUB-MAIN INSTALLATION")
- IMPORTANT: The FIRST grouping often appears at the very beginning of the content (e.g., "SITE PREPARATION", "GENERALLY") - DO NOT MISS IT
- Do NOT use detailed descriptions as grouping names - only use the high-level section headers
- Extract grouping_total_amount from the "To Collection" or "To Collection AED" row at the end of each section
- IMPORTANT: Extract division_total_amount from the "To Summary" or "TOTAL FOR [DIVISION NAME]" row at the end of the sheet
- IMPORTANT: "GENERALLY" sections often have items like "Additional items (Refer attached breakdown)" or "Allow for contingency" with amounts - CAPTURE THESE even if they don't have item codes
- Even if an item has null quantity/rate, still include it if it has an amount
- DO NOT extract sub-total rows as line items (e.g., "To Collection", "Sub-total", "Total for", "Carried Forward", "C/F", "B/F")
- For CONVEYING SYSTEM/ELEVATORS: Extract EVERY elevator, lift, escalator item. Items have codes like "A", "B" with descriptions like "PL-1; 3 stops (G, L01, L02)" - these are HIGH VALUE items (often 100,000+ AED each)! DO NOT SKIP ANY!
{electrical_guidance}
CONTENT:
{sheet_text}

Return ONLY valid JSON, no explanations.'''


def get_chunk_extraction_prompt(
    chunk_text: str,
    sheet_name: str,
    chunk_num: int,
    total_chunks: int,
    reference_items: list[dict] | None = None,
) -> str:
    """
    Generate a prompt for extracting items from a chunk of a large sheet.
    Uses the simpler, more effective prompt format.
    
    Args:
        chunk_text: Text content of this chunk
        sheet_name: Name of the sheet (division name)
        chunk_num: Current chunk number (1-based)
        total_chunks: Total number of chunks
        reference_items: Optional list of template items for context
        
    Returns:
        Formatted prompt string
    """
    # Electrical: fixed groupings (same as sheet prompt — was giving good results)
    electrical_guidance = ""
    if "ELECTRICAL" in sheet_name.upper():
        electrical_guidance = """
IMPORTANT FOR ELECTRICAL ENGINEERING:
Use ONLY these high-level groupings (in ALL CAPS):
- SUB-MAIN INSTALLATION
- SMALL POWER INSTALLATION
- LIGHTING INSTALLATION
- EARTHING AND BONDING SYSTEM
- FIRE ALARM AND VOICE EVACUATION INSTALLATION
- ICT INSTALLATION
- INTERCOM INSTALLATION
- SUNDRIES TO ELECTRICAL ENGINEERING INSTALLATIONS
- WORK INCIDENTAL TO ELECTRICAL ENGINEERING INSTALLATIONS
- GENERALLY

Do NOT create sub-groupings for cable types, conduit sizes, or individual equipment. Keep items under their main section header.
"""

    # WOODWORK/JOINERY: many lines are lump sums — Amount column = total, do NOT use qty×rate
    woodwork_guidance = ""
    if "WOODWORK" in sheet_name.upper() or "JOINERY" in sheet_name.upper():
        woodwork_guidance = """
CRITICAL FOR WOODWORK/JOINERY:
- The "Amount" (or "Total") column is the TOTAL for the line. Copy it exactly into item_amount.
- Do NOT set item_amount = item_quantity × item_rate. Many lines are lump sums where Amount is the total even when Qty > 1.
- Example: if the row shows Qty 2, Rate 4821, Amount 4821 → set item_quantity=2, item_rate=4821, item_amount=4821 (not 9642).
"""

    ref_block = _build_reference_block(reference_items)
    
    return f'''You are a construction BOQ expert. Extract ALL line items from this sheet with proper hierarchy.

SHEET: {sheet_name} (Part {chunk_num})

Return JSON with this EXACT structure (note the 4-level hierarchy: Division → Grouping → Sub-Grouping → Line Items):
{{
  "division_name": "ACTUAL DIVISION NAME FROM CONTENT (e.g., SITE WORK, CONCRETE WORK - NOT the sheet code)",
  "division_total_amount": "123456.00",
  "grouping": [
    {{
      "grouping_name": "SECTION NAME IN CAPS",
      "grouping_total_amount": "12345.00",
      "sub_groupings": [
        {{
          "sub_grouping_name": "Work category description",
          "line_items": [
            {{"item_id": "A", "item_description": "desc", "item_quantity": 10.0, "item_unit": "m²", "item_rate": 25.5, "item_rate_raw": null, "item_amount": 255.0, "item_amount_raw": null}},
            {{"item_id": "B", "item_description": "Provisional sum", "item_quantity": null, "item_unit": "Item", "item_rate": null, "item_rate_raw": "Included", "item_amount": null, "item_amount_raw": "Included"}}
          ]
        }}
      ]
    }}
  ]
}}

HIERARCHY EXPLANATION:
- **Grouping**: Main section header (e.g., "SITE PREPARATION", "EARTHWORKS", "POURED CONCRETE")
- **Sub-Grouping**: Work category within a grouping that groups related items. Examples:
  * "Site clearance; including all ancillary works as necessary"
  * "Anti-Termite Treatment"
  * "Concrete Grade 40 N/mm2; in:"
  * "Fabric reinforcement to BS 4483 reference A142"
- **Line Item**: Individual priced items (A, B, C...) with quantity, rate, and amount

SUB-GROUPING RULES:
- Sub-groupings are descriptive headers that appear BEFORE a set of related items
- They usually end with ":" or describe a category of work
- If no clear sub-grouping exists, use "GENERAL" as the sub_grouping_name
- Sub-groupings do NOT have totals - only groupings have totals

{ref_block}EXTRACTION RULES:
- CRITICAL: division_name must be the ACTUAL TRADE/SCOPE NAME from the content (e.g., "SITE WORK", "CONCRETE WORK", "MASONRY"), NOT the sheet code (like "3A-B", "#A-B")
- Extract EVERY row with item codes (A, B, C, D, E, F, G, H, J, K, L, M, N, etc.)
- ALSO extract rows WITHOUT item codes if they have amounts (e.g., in GENERALLY sections)
- item_quantity, item_rate, and item_amount must be numbers when numeric
- item_amount is the total amount (usually last column) — extract it DIRECTLY from the column, do NOT recalculate from qty × rate.
  Trust the Amount column even when it differs from qty × rate (common for lump sums and provisional items).
- ACCURACY CHECK: The SUM of all item_amount values MUST closely match grouping_total_amount and division_total_amount. Aim for < 0.5% variance.
- Use null for missing values
- SPECIAL VALUES: If rate/amount is text like "Included", "Excluded", "N/A":
  * Set item_rate/item_amount to null
  * Set item_rate_raw/item_amount_raw to the text value
- Groupings are MAIN SECTION HEADERS (ALL CAPS)
- Extract grouping_total_amount from "To Collection" rows
- DO NOT extract sub-total rows as line items
- For CONVEYING SYSTEM/ELEVATORS: Extract ALL items - these are high value!
{electrical_guidance}
{woodwork_guidance}
CONTENT:
{chunk_text}

Return ONLY valid JSON, no explanations.'''


def split_content_by_divisions(
    sheet_text: str,
    division_names: list[str],
    verbose: bool = False,
) -> list[tuple[str, str]]:
    """
    Split a single sheet's text into per-division segments using known division names.

    BOQ files often have one summary sheet listing all division names + totals,
    and one main content sheet containing ALL divisions.  This function uses the
    known division names to find boundaries and slice the text.

    Handles formats like:
      - "Section B: Site Work"
      - "Section B: Site Work (CONT'D)"
      - "SITE WORK"

    The FIRST occurrence of each division name marks its start boundary.
    Subsequent "(CONT'D)" occurrences are NOT treated as new boundaries.

    Args:
        sheet_text: Full text content of the content sheet
        division_names: Division names extracted from the summary sheet
        verbose: Print debug info

    Returns:
        List of (division_name, segment_text) tuples.
        If fewer than 2 divisions can be located the function returns an empty
        list, signalling the caller to fall back to whole-sheet extraction.
    """
    lines = sheet_text.split("\n")
    lines_upper = [l.upper() for l in lines]

    positions: list[tuple[int, str]] = []
    found_names: set[str] = set()

    def _is_contd(line: str) -> bool:
        return "CONT'D" in line or "CONT)" in line or "(CONTINUED)" in line

    # Normalize division name for matching: "Section B: Site Work" -> "SITE WORK"
    # so we can match content that has either "Section B: Site Work" or "SITE WORK"
    def _norm_div_key(n: str) -> str:
        u = n.upper().strip()
        if ":" in u:
            return u.split(":")[-1].strip()
        return u

    for name in division_names:
        name_upper = name.upper().strip()
        if not name_upper:
            continue
        norm_key = _norm_div_key(name)
        if not norm_key:
            continue

        matched = False

        # PASS 1: Look for "Section X: <name>" pattern — most reliable
        for line_idx, line_u in enumerate(lines_upper):
            stripped = line_u.strip().strip("|").strip()
            if _is_contd(stripped):
                continue
            if re.match(r"^SECTION\s+[A-Z]\s*:", stripped):
                after_colon = re.sub(r"^SECTION\s+[A-Z]\s*:\s*", "", stripped).strip()
                # Match full name or normalized (e.g. "SITE WORK" vs "Section B: Site Work")
                if (
                    name_upper == after_colon
                    or norm_key == after_colon
                    or name_upper in after_colon
                    or norm_key in after_colon
                ):
                    if norm_key not in found_names:
                        positions.append((line_idx, name))
                        found_names.add(norm_key)
                        matched = True
                    break

        if matched:
            continue

        # PASS 2: Look for "Bill No. N - Division Name" pattern
        if not matched:
            for line_idx, line_u in enumerate(lines_upper):
                stripped = line_u.strip().strip("|").strip()
                if _is_contd(stripped):
                    continue
                m = re.match(r"^BILL\s*(?:NO\.?)?\s*\d+\s*[-–:]\s*(.+)$", stripped)
                if m:
                    after_dash = m.group(1).strip()
                    if (
                        norm_key == after_dash
                        or name_upper == after_dash
                        or norm_key in after_dash
                        or name_upper in after_dash
                    ):
                        if norm_key not in found_names:
                            positions.append((line_idx, name))
                            found_names.add(norm_key)
                            matched = True
                        break

        if matched:
            continue

        # PASS 3: Look for the name as a standalone header line (short line)
        for line_idx, line_u in enumerate(lines_upper):
            stripped = line_u.strip().strip("|").strip()
            if _is_contd(stripped):
                continue
            if stripped == name_upper or stripped == norm_key or (
                (name_upper in stripped or norm_key in stripped)
                and len(stripped) < max(len(name_upper), len(norm_key)) + 25
            ):
                if norm_key not in found_names:
                    positions.append((line_idx, name))
                    found_names.add(norm_key)
                break

    if verbose:
        print(f"  Division segmentation: found {len(positions)}/{len(division_names)} division headers")
        for line_idx, name in positions:
            print(f"    → line {line_idx}: {name}")

    # Need at least 2 divisions to segment meaningfully
    if len(positions) < 2:
        return []

    # Sort by line position
    positions.sort(key=lambda x: x[0])

    segments: list[tuple[str, str]] = []
    for i, (start_line, name) in enumerate(positions):
        if i + 1 < len(positions):
            end_line = positions[i + 1][0]
        else:
            end_line = len(lines)

        segment_text = "\n".join(lines[start_line:end_line]).strip()
        if segment_text:
            segments.append((name, segment_text))

    if verbose:
        for name, seg in segments:
            seg_lines = seg.count("\n") + 1
            print(f"    Segment '{name}': {seg_lines} lines, {len(seg):,} chars")

    return segments


def split_sheet_into_chunks(
    sheet_text: str,
    max_chars: int = 8000,
    max_lines: int = 150,
    overlap: int = 20,
) -> list[str]:
    """
    Split a large sheet text into smaller chunks for processing.
    
    Uses line-based splitting (150 lines per chunk) with a 20-line overlap
    so items at chunk boundaries appear in both adjacent chunks.  The
    downstream ``merge_chunk_groupings`` deduplicates by item key, so the
    overlap only adds safety — it never causes double-counting.
    
    Args:
        sheet_text: Full text of the sheet
        max_chars: Maximum characters per chunk (fallback)
        max_lines: Maximum lines per chunk (default 150)
        overlap: Number of overlapping lines between consecutive chunks
        
    Returns:
        List of text chunks
    """
    lines = sheet_text.split('\n')
    
    # If small enough, return as single chunk
    if len(lines) <= max_lines and len(sheet_text) <= max_chars:
        return [sheet_text]
    
    step = max(max_lines - overlap, 1)
    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + max_lines, len(lines))
        chunks.append('\n'.join(lines[start:end]))
        start += step
        if end == len(lines):
            break
    
    return chunks


def merge_chunk_groupings(all_groupings: list[Grouping]) -> list[Grouping]:
    """
    Merge groupings from multiple chunks, combining sub-groupings and items with the same grouping name.
    Removes "(CONT'D)" suffix for better matching.
    SUMS extracted grouping_total_amount values (from "To Collection" rows across chunks).
    
    Args:
        all_groupings: List of Grouping objects from all chunks
        
    Returns:
        Merged list of Grouping objects with summed extracted totals
    """
    merged = OrderedDict()
    extracted_totals = {}  # Track SUMMED extracted totals by base name
    
    for grp in all_groupings:
        name = grp.grouping_name or "UNSPECIFIED"
        # Remove "(CONT'D)" or "(CONT)" suffix for matching
        base_name = name.replace("(CONT'D)", "").replace("(CONT)", "").strip()
        # Also handle regex patterns
        base_name = re.sub(r"\s*\(CONT'?D?\).*$", "", base_name, flags=re.IGNORECASE).strip()
        
        # SUM extracted totals (don't just take first one!)
        if grp.grouping_total_amount:
            try:
                amount = float(str(grp.grouping_total_amount).replace(",", ""))
                if base_name in extracted_totals:
                    extracted_totals[base_name] += amount
                else:
                    extracted_totals[base_name] = amount
            except ValueError:
                pass  # Skip invalid amounts
        
        if base_name in merged:
            # Add sub-groupings to existing grouping
            merged[base_name].sub_groupings.extend(grp.sub_groupings)
        else:
            # Create new grouping with clean base name
            merged[base_name] = Grouping(
                grouping_name=base_name,
                grouping_total_amount=None,
                sub_groupings=list(grp.sub_groupings)
            )
    
    # Build result with SUMMED extracted totals and merged sub-groupings
    result = []
    for base_name, grp in merged.items():
        # Merge sub-groupings by name
        merged_sub_groupings = OrderedDict()
        for sub_grp in grp.sub_groupings:
            sub_name = sub_grp.sub_grouping_name or "GENERAL"
            # Clean sub-grouping name
            sub_name = re.sub(r"\s*\(CONT'?D?\).*$", "", sub_name, flags=re.IGNORECASE).strip()
            
            if sub_name in merged_sub_groupings:
                merged_sub_groupings[sub_name].line_items.extend(sub_grp.line_items)
            else:
                merged_sub_groupings[sub_name] = SubGrouping(
                    sub_grouping_name=sub_name,
                    line_items=list(sub_grp.line_items)
                )
        
        # De-duplicate only EXACT duplicates (same row in two chunks). Do NOT dedup
        # by (item_id, amount) or (item_id, desc_snippet, amount) — BOQs reuse item
        # codes (A, B, C) across sections, so that collapses 600+ valid rows.
        final_sub_groupings = []
        for sub_name, sub_grp in merged_sub_groupings.items():
            seen_exact = set()
            unique_items = []
            for item in sub_grp.line_items:
                item_id = (item.item_id or "").strip()
                desc = (item.item_description or "").strip()
                qty = item.item_quantity
                rate = item.item_rate
                amount = item.item_amount
                exact_key = (item_id, desc, qty, rate, amount)
                if exact_key not in seen_exact:
                    seen_exact.add(exact_key)
                    unique_items.append(item)
            
            final_sub_groupings.append(SubGrouping(
                sub_grouping_name=sub_grp.sub_grouping_name,
                line_items=unique_items
            ))
        
        # Use summed extracted total if available
        if base_name in extracted_totals:
            final_total = f"{extracted_totals[base_name]:.2f}"
        else:
            # Calculate as fallback from all sub-groupings.
            # Prefer item_amount (direct LLM extraction from Amount column)
            # over qty*rate (which can be wrong for lump sums).
            total = 0.0
            has_amounts = False
            for sub_grp in final_sub_groupings:
                for item in sub_grp.line_items:
                    if item.item_amount is not None:
                        total += item.item_amount
                        has_amounts = True
                    elif item.item_quantity is not None and item.item_rate is not None:
                        total += item.item_quantity * item.item_rate
                        has_amounts = True
            final_total = f"{total:.2f}" if has_amounts else None
        
        result.append(Grouping(
            grouping_name=grp.grouping_name,
            grouping_total_amount=final_total,
            sub_groupings=final_sub_groupings
        ))
    
    return result


def _normalize_division_name_to_summary(
    candidate: str, division_totals_map: dict[str, str]
) -> str:
    """If candidate matches a summary key (case-insensitive or contains), return that key for consistency."""
    if not candidate or not division_totals_map:
        return candidate
    cand_upper = candidate.upper().strip()
    for key in division_totals_map:
        if not key:
            continue
        key_upper = key.upper()
        if key_upper == cand_upper:
            return key
        if cand_upper in key_upper or key_upper in cand_upper:
            return key
    return candidate


def _resolve_division_name(
    sheet_name: str,
    sheet_text: str,
    division_totals_map: dict[str, str],
    data_division_name: str | None = None
) -> str:
    """Resolve the best division name for a sheet; prefer summary/template key for 1:1 match."""
    import re
    
    # If LLM provided a division name and it's not just the sheet code, use it (then normalize to summary key)
    if data_division_name and data_division_name.strip():
        name = data_division_name.strip()
        if not re.match(r'^[\d#].*[-]|^Bill\s*\d+$', name, re.IGNORECASE):
            return _normalize_division_name_to_summary(name, division_totals_map)

    # Try to match a summary division name that appears in the sheet text
    text_upper = sheet_text.upper()
    matches = [
        name for name in division_totals_map.keys()
        if name and name.upper() in text_upper
    ]
    if len(matches) == 1:
        return matches[0]
    
    # Try to extract division name from common patterns in sheet text
    header_text = sheet_text[:2000]
    
    # Common division names to look for
    known_divisions = [
        "SITE WORK", "SITEWORK", "SITE WORKS",
        "CONCRETE WORK", "CONCRETE WORKS", "CONCRETEWORK",
        "MASONRY", "MASONRY WORK",
        "METALWORK", "METAL WORK", "STRUCTURAL STEEL",
        "WOODWORK", "WOOD WORK", "CARPENTRY", "JOINERY",
        "THERMAL & MOISTURE PROTECTION", "THERMAL AND MOISTURE PROTECTION",
        "WATERPROOFING", "INSULATION",
        "DOORS AND WINDOWS", "DOORS & WINDOWS", "WINDOWS AND DOORS",
        "FINISHES", "FINISHING", "FINISH WORK",
        "ACCESSORIES", "SPECIALTIES",
        "EQUIPMENT", "SPECIAL EQUIPMENT",
        "CONVEYING SYSTEM", "CONVEYING SYSTEMS", "ELEVATORS", "LIFTS",
        "MECHANICAL ENGINEERING", "MECHANICAL WORKS", "HVAC", "PLUMBING",
        "ELECTRICAL ENGINEERING", "ELECTRICAL WORKS", "ELECTRICAL",
        "ALTERNATIVE", "ALTERNATIVES", "PROVISIONAL SUMS",
        "EXTERNAL WORKS", "LANDSCAPING",
    ]
    
    header_upper = header_text.upper()
    for div_name in known_divisions:
        if div_name in header_upper:
            return _normalize_division_name_to_summary(div_name, division_totals_map)
    
    # Pattern: "BILL NO. X - DIVISION NAME" or "BILL X - NAME"
    bill_pattern = re.search(r'BILL\s*(?:NO\.?)?\s*\d+\s*[-–:]\s*([A-Z][A-Z\s&]+)', header_upper)
    if bill_pattern:
        return _normalize_division_name_to_summary(
            bill_pattern.group(1).strip(), division_totals_map
        )
    
    # Pattern: Division header in all caps at start of content
    div_header = re.search(r'^([A-Z]{4,}(?:\s+[A-Z&]{2,})*)\s*$', header_upper, re.MULTILINE)
    if div_header:
        candidate = div_header.group(1).strip()
        if not re.match(r'^[\d#]+', candidate):
            return _normalize_division_name_to_summary(candidate, division_totals_map)

    return _normalize_division_name_to_summary(sheet_name, division_totals_map)


def _extract_summary_totals(
    llm_provider: BaseLLMProvider,
    sheet_name: str,
    sheet_text: str,
    verbose: bool = False
) -> tuple[str | None, dict[str, str]]:
    """
    Extract project total and division totals from a summary sheet.
    
    Args:
        llm_provider: Initialized LLM provider
        sheet_name: Name of the summary sheet
        sheet_text: Text content of the summary sheet
        verbose: Print debug info
        
    Returns:
        Tuple of (project_total, division_totals_dict)
    """
    prompt = f'''You are a construction document expert. Extract the PROJECT TOTAL and DIVISION TOTALS from this BOQ summary sheet.

## SHEET NAME: {sheet_name}

## OUTPUT FORMAT:
Return a JSON object with this structure:
{{
  "project_name": "project name if found",
  "project_total": "total amount as string (e.g., '7019123.82')",
  "division_totals": {{
    "SITE WORK": "123456.78",
    "CONCRETE WORK": "234567.89"
  }}
}}

## CRITICAL RULES:
1. Extract the EXACT total amounts as they appear in the document
2. DO NOT calculate - just read the values directly
3. IMPORTANT: Look for "SUB-TOTAL FOR BILL NO. 1" or "SUB-TOTAL" as the project total
4. IGNORE any "MULTIPLICATION" lines that multiply the sub-total for multiple units/villas
5. The project total should be the cost for ONE unit/villa, NOT the multiplied total for multiple units
6. Division totals: include EVERY division row from the table — do not omit or merge any. Use the exact division name as shown (e.g. "Section B: Site Work" or "SITE WORK"). Typically 10–20 divisions.
7. Return amounts as strings with decimals (e.g., "7019123.82")

## SHEET CONTENT:
```
{sheet_text}
```

## OUTPUT:
Return ONLY the JSON object, no explanations.
'''
    
    try:
        response = llm_provider.complete(prompt)
        data, parse_err = _parse_llm_json_response(response)
        if data is None:
            if verbose:
                print(f"  Summary JSON parse failed: {parse_err}")
            return None, None, {}

        project_total = data.get("project_total")
        project_name = data.get("project_name")
        division_totals = data.get("division_totals", {})
        
        if verbose:
            print(f"  Summary extraction: project_total={project_total}")
            print(f"  Found {len(division_totals)} division totals")
        
        return project_name, project_total, division_totals
        
    except Exception as e:
        if verbose:
            print(f"  Error extracting summary: {e}")
        return None, None, {}


def _extract_summary_totals_fast(
    sheet_name: str,
    sheet_text: str,
    verbose: bool = False,
) -> tuple[str | None, str | None, dict[str, str]]:
    """Extract project name, project total, and division totals using regex only.

    This is ~1000x faster than the LLM-based ``_extract_summary_totals`` because
    the summary sheet text is already pipe-delimited structured data.  It falls
    back gracefully (returns empty results) so the caller can retry with the LLM.

    Returns:
        (project_name, project_total, division_totals)
    """

    project_name: str | None = None
    project_total: str | None = None
    division_totals: dict[str, str] = {}

    lines = sheet_text.split("\n")

    # ── Helpers ──────────────────────────────────────────────────────
    _NUM_RE = re.compile(r"[\d,]+\.?\d*")

    def _last_number(line: str) -> str | None:
        """Return the last numeric token in a line (comma-separated ok)."""
        nums = _NUM_RE.findall(line)
        for candidate in reversed(nums):
            cleaned = candidate.replace(",", "")
            if cleaned and cleaned not in ("0", "0.0", "0.00"):
                try:
                    float(cleaned)
                    return cleaned
                except ValueError:
                    continue
        return None

    # ── Pass 1: Extract division name → total rows ───────────────────
    # Patterns we handle:
    #   "Section B: Site Work | | | | | 300221.00"
    #   "Section B : Site Work | 300221"
    #   "Bill No. 2 - CONCRETE WORK | | | | 500123"
    #   "SITE WORK | | | | | 300221"
    #   "B | Site Work | 300221"

    _SECTION_RE = re.compile(
        r"^Section\s+[A-Z]\s*:\s*(.+)",
        re.IGNORECASE,
    )
    _BILL_RE = re.compile(
        r"^Bill\s*(?:No\.?)?\s*\d+\s*[-–:]\s*(.+)",
        re.IGNORECASE,
    )
    _LETTER_PREFIX_RE = re.compile(
        r"^[A-Z]\s*\|\s*(.+)",
        re.IGNORECASE,
    )

    # Keywords that mark summary / total rows — NOT divisions
    _SKIP_LINE_KW = {
        "SUB-TOTAL", "SUB TOTAL", "SUBTOTAL", "TOTAL", "GRAND TOTAL",
        "MULTIPLICATION", "MULTIPLY", "NUMBER OF", "NO. OF", "UNITS",
        "AMOUNT", "DESCRIPTION", "ITEM", "# COLUMNS", "NOTE:",
        "BILL 1-SUM", "GENERAL SUMMARY",
    }

    for line in lines:
        stripped = line.strip().strip("|").strip()
        if not stripped or len(stripped) < 4:
            continue
        stripped_upper = stripped.upper()

        # Skip header/total rows
        if any(kw in stripped_upper for kw in _SKIP_LINE_KW):
            # But try to extract project total from sub-total lines
            if "SUB-TOTAL" in stripped_upper or "SUB TOTAL" in stripped_upper:
                num = _last_number(stripped)
                if num and not project_total:
                    project_total = num
            elif "GRAND TOTAL" in stripped_upper and not project_total:
                num = _last_number(stripped)
                if num:
                    project_total = num
            continue

        # Skip very short tokens or pure numbers
        text_part = re.sub(r"[\d,.|]+", "", stripped).strip()
        if len(text_part) < 3:
            continue

        # Try pattern matching
        div_name: str | None = None

        m = _SECTION_RE.match(stripped)
        if m:
            # "Section B: Site Work | 300221" → name = "Section B: Site Work"
            div_name = m.group(0).split("|")[0].strip()
        else:
            m = _BILL_RE.match(stripped)
            if m:
                div_name = m.group(1).split("|")[0].strip()
            else:
                m = _LETTER_PREFIX_RE.match(stripped)
                if m:
                    div_name = m.group(1).split("|")[0].strip()

        # Fallback: check for known division names in the line
        if not div_name:
            _KNOWN = [
                "SITE WORK", "SITE WORKS", "CONCRETE WORK", "CONCRETE WORKS",
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
            for known in _KNOWN:
                if known in stripped_upper:
                    div_name = known
                    break

        if div_name:
            num = _last_number(stripped)
            if num:
                division_totals[div_name] = num
            elif div_name.upper() not in {k.upper() for k in division_totals}:
                # Division row with no amount — record with "0" so we know it exists
                division_totals[div_name] = "0"

    # ── Pass 2: Extract project name (first non-empty text line) ─────
    for line in lines:
        stripped = line.strip().strip("|").strip()
        if not stripped or len(stripped) < 4:
            continue
        stripped_upper = stripped.upper()
        # Skip header keywords
        if any(kw in stripped_upper for kw in ("# COLUMNS", "ITEM_ID", "DESCRIPTION")):
            continue
        # Skip if it's a division row we already parsed
        is_div = False
        for dname in division_totals:
            if dname.upper() in stripped_upper:
                is_div = True
                break
        if is_div:
            continue
        # Skip "Section" / "Bill" lines
        if stripped_upper.startswith("SECTION") or stripped_upper.startswith("BILL"):
            # Might be the project name if it looks like "BILL OF QUANTITIES FOR <project>"
            m = re.search(r"(?:FOR|OF)\s+(.{5,})", stripped, re.IGNORECASE)
            if m:
                project_name = m.group(1).strip().strip("|").strip()
                break
            continue
        # Use this line as project name
        text_only = re.sub(r"[\d,.|]+", "", stripped).strip()
        if len(text_only) > 4:
            project_name = text_only
            break

    # ── Pass 3: Project total fallback — sum of division totals ──────
    if not project_total and division_totals:
        total = 0.0
        for v in division_totals.values():
            try:
                total += float(v.replace(",", ""))
            except (ValueError, TypeError):
                pass
        if total > 0:
            project_total = f"{total:.2f}"

    if verbose:
        print(f"  Fast summary ({sheet_name}): project_total={project_total}, "
              f"{len(division_totals)} divisions", flush=True)
        for dn, dt in list(division_totals.items())[:5]:
            print(f"    {dn}: {dt}", flush=True)
        if len(division_totals) > 5:
            print(f"    ... and {len(division_totals) - 5} more", flush=True)

    return project_name, project_total, division_totals


def extract_boq_by_sheets(
    sheets_data: list[tuple[str, str]],
    provider: str = "vertex",
    model: str | None = None,
    verbose: bool = False,
    chunk_threshold: int = 12000
) -> BOQ:
    """
    Extract BOQ by processing each sheet separately.
    
    For large sheets (exceeding chunk_threshold characters), uses chunked
    extraction to ensure all items are captured.
    
    Args:
        sheets_data: List of (sheet_name, sheet_text) tuples
        provider: LLM provider name
        model: Model name
        verbose: Print debug info
        chunk_threshold: Character threshold above which to use chunked extraction
        
    Returns:
        Combined BOQ object
    """
    # Default models
    default_models = {
        "openai": "gpt-4-turbo",
        "anthropic": "claude-3-sonnet-20240229",
        "google": "gemini-2.5-flash",
        "vertex": "gemini-2.5-flash",
    }
    
    model_name = model or default_models.get(provider, "gemini-2.5-flash")
    
    # Initialize provider
    if provider == "vertex":
        llm_provider = ServiceAccountProvider(model=model_name)
    elif provider == "openai":
        llm_provider = OpenAIProvider(model=model_name)
    elif provider == "anthropic":
        llm_provider = AnthropicProvider(model=model_name)
    elif provider == "google":
        llm_provider = GoogleProvider(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    schema = load_schema()
    divisions = []
    project_name = None
    project_total = None
    division_totals_map = {}
    
    # First pass: extract summary totals from ALL summary sheets
    # Some files split project total and division breakdown across sheets
    # (e.g. "General Summary" + "Bill 2 Summary"). Process every summary sheet
    # and merge division_totals so the template has the full set of divisions.
    for sheet_name, sheet_text in sheets_data:
        if not sheet_text.strip():
            continue
        if _is_summary_sheet(sheet_name):
            if verbose:
                print(f"\nExtracting summary from: {sheet_name}")
            extracted_name, extracted_total, div_totals = _extract_summary_totals(
                llm_provider, sheet_name, sheet_text, verbose
            )
            if extracted_name and not project_name:
                project_name = extracted_name
            if extracted_total and not project_total:
                project_total = extracted_total
            if div_totals:
                division_totals_map.update(div_totals)
    
    # Second pass: build list of content sheets and segment multi-division ones
    content_sheets: list[tuple[str, str]] = [
        (name, text)
        for name, text in sheets_data
        if text.strip() and _is_content_sheet(name)
    ]

    # Segment multi-division sheets using summary division names
    if division_totals_map and len(division_totals_map) > len(content_sheets):
        expanded: list[tuple[str, str]] = []
        for sheet_name, sheet_text in content_sheets:
            segments = split_content_by_divisions(
                sheet_text, list(division_totals_map.keys()), verbose=verbose
            )
            if len(segments) >= 2:
                if verbose:
                    print(f"\n✂ Sheet '{sheet_name}' segmented into {len(segments)} divisions")
                expanded.extend(segments)
            else:
                expanded.append((sheet_name, sheet_text))
        content_sheets = expanded

    max_retries = 2
    reconcile_threshold = 3.0

    for sheet_name, sheet_text in content_sheets:
        if verbose:
            print(f"\nProcessing sheet: {sheet_name}")
            print(f"  Text length: {len(sheet_text)} characters")

        best_division: Division | None = None
        best_variance: float | None = None

        for attempt in range(1, max_retries + 1):
            try:
                division_total_from_llm = None
                division_name_from_llm = None

                if len(sheet_text) > chunk_threshold:
                    groupings, division_name_from_llm = _extract_sheet_chunked(
                        llm_provider, sheet_name, sheet_text, verbose
                    )
                else:
                    prompt = get_sheet_extraction_prompt(sheet_text, sheet_name, schema)
                    response = llm_provider.complete(prompt)

                    if verbose:
                        print(f"  Response length: {len(response)} characters")

                    data, parse_err = _parse_llm_json_response(response)
                    if data is None:
                        raise ValueError(parse_err or "Invalid JSON")
                    raw_groupings = _parse_groupings_from_response(data)
                    groupings = merge_chunk_groupings(raw_groupings)

                    division_total_from_llm = data.get("division_total_amount")
                    division_name_from_llm = data.get("division_name")

                if not groupings:
                    if verbose:
                        print(f"  Attempt {attempt}: no groupings extracted")
                    continue

                # Resolve division total (same 3-tier priority)
                div_total = None

                carried_total = _extract_carried_total(sheet_text)
                if carried_total:
                    div_total = carried_total
                    if verbose:
                        print(f"  Total for {sheet_name}: {carried_total} (from 'Carried to Summary')")

                if not div_total:
                    for summary_div_name, summary_total in division_totals_map.items():
                        if summary_div_name.upper() == sheet_name.upper() or \
                           summary_div_name.upper() in sheet_name.upper() or \
                           sheet_name.upper() in summary_div_name.upper():
                            div_total = summary_total
                            if verbose:
                                print(f"  Total for {sheet_name}: {summary_total} (from summary sheet)")
                            break

                if not div_total and division_total_from_llm:
                    div_total = str(division_total_from_llm)
                    if verbose:
                        print(f"  Total for {sheet_name}: {division_total_from_llm} (from LLM)")

                division_name = _resolve_division_name(
                    sheet_name, sheet_text, division_totals_map, division_name_from_llm
                )

                division = Division(
                    division_name=division_name,
                    division_total_amount=div_total,
                    grouping=groupings,
                )

                # Reconciliation check
                recon = reconcile_division(division, threshold_pct=reconcile_threshold, verbose=verbose)

                cur_var = recon.variance_pct if recon.variance_pct is not None else 0.0
                if best_division is None or (best_variance is not None and cur_var < best_variance):
                    best_division = division
                    best_variance = cur_var

                if recon.passed:
                    total_items = sum(_count_items_in_grouping(g) for g in groupings)
                    if verbose:
                        print(f"  ✓ {sheet_name}: {total_items} items, "
                              f"variance={cur_var:.1f}% [PASS]")
                    break  # No need to retry

                if verbose:
                    print(f"  ⟳ {sheet_name} attempt {attempt}/{max_retries}: "
                          f"variance {cur_var:.1f}% > {reconcile_threshold}% — retrying")

            except Exception as e:
                if verbose:
                    print(f"  Error processing {sheet_name} (attempt {attempt}): {e}")
                    import traceback
                    traceback.print_exc()
                continue

        if best_division:
            divisions.append(best_division)
            if verbose and best_variance is not None and best_variance > reconcile_threshold:
                print(f"  ⚠ {sheet_name}: best variance {best_variance:.1f}% after {max_retries} attempts")
    
    # Project total: use extracted value only, never calculate from line items.
    final_project_total = project_total
    if not final_project_total:
        total_amount = 0
        for div in divisions:
            if div.division_total_amount:
                try:
                    total_amount += float(str(div.division_total_amount).replace(",", ""))
                except (ValueError, TypeError):
                    pass
        if total_amount > 0:
            final_project_total = f"{total_amount:.2f}"

    boq = BOQ(
        project_name=project_name,
        contractor=None,
        project_total_amount=final_project_total,
        divisions=divisions,
    )

    if verbose:
        recon_results = reconcile_boq(boq, threshold_pct=2.0, verbose=False)
        passed = sum(1 for r in recon_results if r.passed)
        failed = len(recon_results) - passed
        print(f"\n── Reconciliation Summary ──")
        print(f"  {passed}/{len(recon_results)} divisions within 2% variance")
        if failed > 0:
            print(f"  ⚠ {failed} division(s) exceed 2% variance:")
            for r in recon_results:
                if not r.passed:
                    var_str = f"{r.variance_pct:.1f}%" if r.variance_pct is not None else "N/A"
                    print(f"    • {r.division_name}: extracted={r.extracted_total}, "
                          f"calculated={r.calculated_total:.2f}, variance={var_str}")

    return boq


def _extract_sheet_chunked(
    llm_provider: BaseLLMProvider,
    sheet_name: str,
    sheet_text: str,
    verbose: bool = False
) -> tuple[list[Grouping], str | None]:
    """
    Extract items from a large sheet using chunked processing.
    
    Args:
        llm_provider: Initialized LLM provider
        sheet_name: Name of the sheet
        sheet_text: Full text of the sheet
        verbose: Print debug info
        
    Returns:
        Tuple of (merged Grouping objects, division_name_from_llm)
    """
    chunks = split_sheet_into_chunks(sheet_text, max_chars=8000)
    
    if verbose:
        print(f"  Split into {len(chunks)} chunks for processing")
    
    all_groupings = []
    division_name_from_llm = None
    
    for i, chunk in enumerate(chunks, 1):
        if verbose:
            print(f"  Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
        
        prompt = get_chunk_extraction_prompt(chunk, sheet_name, i, len(chunks))
        
        try:
            response = llm_provider.complete(prompt)
            data, parse_err = _parse_llm_json_response(response)
            if data is None:
                if verbose:
                    print(f"    Error in chunk {i}: {parse_err}")
                continue
            chunk_groupings = _parse_groupings_from_response(data)
            all_groupings.extend(chunk_groupings)
            if not division_name_from_llm:
                division_name_from_llm = data.get("division_name")
            chunk_items = sum(_count_items_in_grouping(g) for g in chunk_groupings)
            if verbose:
                print(f"    Extracted {chunk_items} items from chunk {i}")
        except Exception as e:
            if verbose:
                print(f"    Error in chunk {i}: {e}")
            continue
    
    # Merge groupings with the same name
    merged = merge_chunk_groupings(all_groupings)
    
    if verbose:
        total_items = sum(_count_items_in_grouping(g) for g in merged)
        print(f"  Merged into {len(merged)} groupings with {total_items} total items")
    
    return merged, division_name_from_llm


_SUBTOTAL_EXACT: set[str] = {
    "TO COLLECTION",
    "TO COLLECTION AED",
    "TO SUMMARY",
    "SUB-TOTAL",
    "SUBTOTAL",
    "SUB TOTAL",
    "C/F",
    "B/F",
    "C/FWD",
    "B/FWD",
    "CARRIED FORWARD",
    "BROUGHT FORWARD",
    "PAGE TOTAL",
    "SHEET TOTAL",
    "AMOUNT CARRIED",
    "AMOUNT BROUGHT",
    "TOTAL CARRIED",
    "TOTAL BROUGHT",
    "CARRIED TO SUMMARY",
    "CARRIED TO COLLECTION",
}

# Only match known sub-total row labels. Do NOT match "Total for section 2" or
# "Total for Bill 1" — those can be real section headers the LLM put as description.
_SUBTOTAL_RE = re.compile(
    r'^('
    r'TO COLLECTION|TO SUMMARY|SUB-?TOTAL|SUB\s+TOTAL'
    r'|CARRIED FORWARD|BROUGHT FORWARD'
    r'|C/?F(?:WD)?|B/?F(?:WD)?'
    r'|PAGE TOTAL|SHEET TOTAL'
    r'|TOTAL[\s\-]+(?:CARRIED|BROUGHT)\s'
    r'|AMOUNT[\s\-]*(?:CARRIED|BROUGHT)[\s\-\w]*'
    r'|CARRIED TO (?:SUMMARY|COLLECTION|BILL)'
    r')(\s+AED)?(\s*:?\s*[\d,\.]*)?$',
    re.IGNORECASE,
)


def _is_subtotal_row(description: str) -> bool:
    """Check if a line item is actually a sub-total row that shouldn't be counted.

    Uses both exact-match and regex patterns to catch variants like
    "B/Fwd", "C/F", "Page Total", "Total - Carried to Summary", etc.
    """
    if not description:
        return False
    desc_upper = description.upper().strip()

    if desc_upper in _SUBTOTAL_EXACT:
        return True

    if _SUBTOTAL_RE.match(desc_upper):
        return True

    return False


def _validate_and_correct_item(item: LineItem) -> LineItem:
    """Validate a parsed line item and auto-correct common LLM errors.

    Conservative corrections only:
      1. Negative amounts/rates — flips sign (BOQ amounts are always positive).
      2. Missing/zero amount when qty and rate are both available — fills it in.
      3. Small rounding errors (<2%) — corrects to qty*rate.

    IMPORTANT: Never override the LLM's extracted amount when the discrepancy
    is large (>=2%).  In construction BOQs, amount != qty*rate is common for
    lump sums, provisional sums, and items with non-standard pricing.
    The LLM reads the amount directly from the 'Amount' column and is
    generally more reliable than a naive qty*rate recalculation.
    """
    qty = item.item_quantity
    rate = item.item_rate
    amount = item.item_amount

    # Skip validation for items with raw (non-numeric) values
    if item.item_rate_raw or item.item_amount_raw:
        return item

    # Auto-correct: negative amounts
    if amount is not None and amount < 0:
        item.item_amount = abs(amount)
        amount = item.item_amount

    if rate is not None and rate < 0:
        item.item_rate = abs(rate)
        rate = item.item_rate

    # Only fill in amount when it's missing/zero
    if qty is not None and rate is not None and qty != 0 and rate != 0:
        expected = qty * rate
        if amount is None or amount == 0:
            item.item_amount = round(expected, 2)
        elif abs(expected) > 0:
            # Only correct small rounding errors (<2%).
            # Large discrepancies (e.g. lump sums where amount != qty*rate)
            # should be trusted as-is from the LLM extraction.
            pct_diff = abs(expected - amount) / abs(expected)
            if pct_diff < 0.02:
                item.item_amount = round(expected, 2)

    return item


def _is_junk_line_item(
    item_id: str | None,
    description: str,
    qty: float | None,
    rate: float | None,
    amount: float | None,
    rate_raw: str | None = None,
    amount_raw: str | None = None,
) -> bool:
    """True only when the row has no usable value at all (drop it).

    Keep any row with:
      - numeric amount, OR
      - qty + numeric rate, OR
      - qty + description (item to be priced — valid BOQ row), OR
      - a non-empty raw rate/amount string (e.g. 'Included in MEP',
        'Refer additional item') which is meaningful BOQ data.
    """
    has_numeric_value = amount is not None or (qty is not None and rate is not None)
    has_raw_value = bool(rate_raw) or bool(amount_raw)
    has_description = bool(description and description.strip())
    has_qty_and_desc = qty is not None and has_description
    return not (has_numeric_value or has_qty_and_desc or (has_raw_value and has_description))


def _safe_float(val) -> tuple[float | None, str | None]:
    """Try to parse *val* as float. Returns (number, None) on success,
    or (None, raw_string) when the value is a non-numeric string."""
    if val is None:
        return None, None
    if isinstance(val, (int, float)):
        return float(val), None
    s = str(val).strip().replace(",", "")
    if not s:
        return None, None
    try:
        return float(s), None
    except (ValueError, TypeError):
        return None, str(val)


def _parse_line_items_from_data(items_data: list) -> list[LineItem]:
    """
    Parse line items from response data, filtering out sub-total rows and
    rows with no usable value (no amount and no qty×rate). Keeps any row with
    amount or qty+rate OR a meaningful raw string. Validates arithmetic.
    """
    line_items = []
    for item_data in items_data:
        description = item_data.get("item_description", "") or item_data.get("description", "")
        if _is_subtotal_row(description):
            continue

        item_id = item_data.get("item_id") or item_data.get("item_code") or item_data.get("id")

        qty_raw = item_data.get("item_quantity") or item_data.get("quantity")
        qty, _ = _safe_float(qty_raw)

        rate_raw_field = item_data.get("item_rate_raw")
        rate_val = item_data.get("item_rate") or item_data.get("rate")
        rate_num, rate_fallback_raw = _safe_float(rate_val)
        rate_raw = rate_raw_field or rate_fallback_raw

        amount_raw_field = item_data.get("item_amount_raw")
        amount_val = item_data.get("item_amount") or item_data.get("amount")
        amount_num, amount_fallback_raw = _safe_float(amount_val)
        amount_raw = amount_raw_field or amount_fallback_raw

        if _is_junk_line_item(item_id, description, qty, rate_num, amount_num,
                              rate_raw=rate_raw, amount_raw=amount_raw):
            continue

        item = LineItem(
            item_id=item_id,
            item_description=description,
            item_quantity=qty,
            item_unit=item_data.get("item_unit") or item_data.get("unit"),
            item_rate=rate_num,
            item_rate_raw=rate_raw,
            item_amount=amount_num,
            item_amount_raw=amount_raw,
        )
        item = _validate_and_correct_item(item)
        line_items.append(item)
    return line_items


def _parse_groupings_from_response(data: dict) -> list[Grouping]:
    """
    Parse groupings from LLM response data.
    Handles both old format (line_items directly in grouping) and new format (sub_groupings).
    Filters out sub-total rows that shouldn't be counted as line items.
    
    Args:
        data: Parsed JSON response from LLM
        
    Returns:
        List of Grouping objects
    """
    groupings = []
    
    # Handle case where LLM returns a list directly instead of a dict
    if isinstance(data, list):
        # If the list contains grouping-like objects (with grouping_name), use it directly
        if data and isinstance(data[0], dict) and ("grouping_name" in data[0] or "name" in data[0]):
            grouping_data = data
        else:
            # Assume it's a list of line items - wrap in a default grouping
            grouping_data = [{"grouping_name": "GENERALLY", "line_items": data}]
    else:
        # Handle case where data might have "groupings" instead of "grouping"
        grouping_data = data.get("grouping", []) or data.get("groupings", [])
    
    # If no grouping array but there's a line_items at top level, create default grouping
    if not grouping_data and data.get("line_items"):
        grouping_data = [{"grouping_name": "GENERALLY", "line_items": data.get("line_items")}]
    
    for grp_data in grouping_data:
        sub_groupings = []
        
        # Handle LLM key variations: sub_groupings, sub_grouping (singular), sub_groups
        sub_groupings_data = (
            grp_data.get("sub_groupings")
            or grp_data.get("sub_grouping")
            or grp_data.get("sub_groups")
            or []
        )
        
        if sub_groupings_data:
            for sub_grp_data in sub_groupings_data:
                # Handle LLM key variations for line items
                items_data = (
                    sub_grp_data.get("line_items")
                    or sub_grp_data.get("items")
                    or sub_grp_data.get("item_details")
                    or []
                )
                line_items = _parse_line_items_from_data(items_data)
                
                if line_items:
                    sub_groupings.append(SubGrouping(
                        sub_grouping_name=(
                            sub_grp_data.get("sub_grouping_name")
                            or sub_grp_data.get("name")
                            or sub_grp_data.get("sub_grouping_total")  # sometimes LLM puts name here
                            or "GENERAL"
                        ),
                        line_items=line_items,
                    ))
        else:
            # Flat format: line_items directly in grouping - wrap in a default sub_grouping
            items_data = (
                grp_data.get("line_items")
                or grp_data.get("items")
                or grp_data.get("item_details")
                or []
            )
            line_items = _parse_line_items_from_data(items_data)
            
            if line_items:
                sub_groupings.append(SubGrouping(
                    sub_grouping_name="GENERAL",
                    line_items=line_items,
                ))
        
        if sub_groupings:
            # Extract grouping_total_amount from the response (from "To Collection" row)
            extracted_total = grp_data.get("grouping_total_amount") or grp_data.get("total")
            # Normalize the total - remove commas, ensure it's a string
            if extracted_total is not None:
                if isinstance(extracted_total, (int, float)):
                    extracted_total = f"{extracted_total:.2f}"
                else:
                    # Remove commas and clean up
                    extracted_total = str(extracted_total).replace(",", "").strip()
            
            groupings.append(Grouping(
                grouping_name=grp_data.get("grouping_name") or grp_data.get("name") or "GENERALLY",
                grouping_total_amount=extracted_total,
                sub_groupings=sub_groupings,
            ))
    
    # Debug: Log if no groupings found
    if not groupings and data:
        print(f"Warning: No groupings parsed from response. Keys in data: {list(data.keys())}")
        if "grouping" in data:
            print(f"  Grouping data: {data['grouping'][:200] if isinstance(data['grouping'], str) else data['grouping'][:2] if data['grouping'] else 'empty'}")
    
    return groupings


# =============================================================================
# RECONCILIATION — ensure extracted totals match line-item sums within 3%
# =============================================================================

@dataclass
class ReconciliationResult:
    """Result of reconciling a division's extracted total vs line-item sum."""
    division_name: str
    extracted_total: float | None
    calculated_total: float
    item_count: int
    variance_pct: float | None  # None if no extracted total
    passed: bool  # True if variance <= threshold or no extracted total


def build_reference_items_from_boq(boq_dict: dict) -> dict[str, list[dict]]:
    """Build a reference-items-by-division dict from an extracted BOQ dict.

    This is used to give the LLM context when extracting vendor BOQs so it can
    map items 1-to-1 with the template structure.

    Returns:
        { "DIVISION NAME": [ { item_id, item_description, item_quantity, item_unit, grouping_name, sub_grouping_name }, ... ] }
    """
    ref: dict[str, list[dict]] = {}
    for div in boq_dict.get("divisions", []):
        div_name = div.get("division_name") or "UNKNOWN"
        items: list[dict] = []
        for grp in div.get("grouping", []):
            grp_name = grp.get("grouping_name") or ""
            if grp.get("sub_groupings"):
                for sub in grp.get("sub_groupings", []):
                    sub_name = sub.get("sub_grouping_name") or "GENERAL"
                    for it in sub.get("line_items", []):
                        items.append({
                            "item_id": it.get("item_id"),
                            "item_description": it.get("item_description"),
                            "item_quantity": it.get("item_quantity"),
                            "item_unit": it.get("item_unit"),
                            "grouping_name": grp_name,
                            "sub_grouping_name": sub_name,
                        })
            else:
                for it in grp.get("line_items", []):
                    items.append({
                        "item_id": it.get("item_id"),
                        "item_description": it.get("item_description"),
                        "item_quantity": it.get("item_quantity"),
                        "item_unit": it.get("item_unit"),
                        "grouping_name": grp_name,
                        "sub_grouping_name": "GENERAL",
                    })
        if items:
            ref[div_name] = items
    return ref


def _sum_division_items(division: Division) -> tuple[float, int]:
    """
    Sum all line-item amounts in a division.
    Uses item_amount if available, else falls back to qty * rate.
    Returns (total, item_count).
    """
    total = 0.0
    count = 0
    for grp in division.grouping:
        for item in _iter_items_in_grouping(grp):
            count += 1
            if item.item_amount is not None:
                total += item.item_amount
            elif item.item_quantity is not None and item.item_rate is not None:
                total += item.item_quantity * item.item_rate
    return total, count


def _correct_division_doubled_amounts(division: Division) -> None:
    """
    If the division's calculated total is ~2x the extracted total (common when
    the LLM sets item_amount = qty*rate for lump-sum lines where the sheet
    Amount column is the total), halve item_amount for items where amount ≈ qty*rate.
    Mutates the division in place.
    """
    ext_total: float | None = None
    if division.division_total_amount:
        try:
            ext_total = float(str(division.division_total_amount).replace(",", ""))
        except (ValueError, TypeError):
            pass
    if ext_total is None or ext_total <= 0:
        return
    calc_total, item_count = _sum_division_items(division)
    if calc_total <= 0 or item_count == 0:
        return
    ratio = calc_total / ext_total
    # Only when calculated is roughly 2x extracted (90–110% variance)
    if ratio < 1.85 or ratio > 2.15:
        return
    # Count items where amount is close to qty*rate (LLM likely calculated)
    n_calculated = 0
    n_with_amount = 0
    for grp in division.grouping:
        for item in _iter_items_in_grouping(grp):
            if item.item_amount is None:
                continue
            n_with_amount += 1
            q, r = item.item_quantity, item.item_rate
            if q is not None and r is not None and q != 0 and r != 0:
                expected = q * r
                if abs(expected) > 0 and abs(item.item_amount - expected) / expected < 0.02:
                    n_calculated += 1
    if n_with_amount == 0:
        return
    # Only correct if a majority of items with amount look like qty*rate
    if n_calculated < 0.5 * n_with_amount:
        return
    # Halve amount for items where amount ≈ qty*rate
    for grp in division.grouping:
        for sg in grp.sub_groupings:
            for item in sg.line_items:
                if item.item_amount is None:
                    continue
                q, r = item.item_quantity, item.item_rate
                if q is not None and r is not None and q != 0 and r != 0:
                    expected = q * r
                    if abs(expected) > 0 and abs(item.item_amount - expected) / expected < 0.02:
                        item.item_amount = round(item.item_amount / 2, 2)


def reconcile_division(
    division: Division,
    threshold_pct: float = 3.0,
    verbose: bool = False,
) -> ReconciliationResult:
    """
    Check that the extracted division_total_amount matches the sum of line items
    within a given threshold (default 3%).

    Args:
        division: Extracted Division object
        threshold_pct: Maximum allowed variance in percent
        verbose: Print reconciliation details

    Returns:
        ReconciliationResult with pass/fail and variance info
    """
    calc_total, item_count = _sum_division_items(division)

    # Count items that only have raw (non-numeric) values
    raw_only_count = 0
    for grp in division.grouping:
        for item in _iter_items_in_grouping(grp):
            has_numeric = (item.item_amount is not None) or (
                item.item_quantity is not None and item.item_rate is not None
            )
            has_raw = bool(item.item_rate_raw) or bool(item.item_amount_raw)
            if has_raw and not has_numeric:
                raw_only_count += 1

    ext_total: float | None = None
    if division.division_total_amount:
        try:
            ext_total = float(str(division.division_total_amount).replace(",", ""))
        except (ValueError, TypeError):
            ext_total = None

    variance_pct: float | None = None
    passed = True

    # If most items are raw-value-only (e.g. "Included in MEP"), reconciliation
    # is meaningless — there's nothing numeric to sum. Auto-pass.
    if item_count > 0 and raw_only_count > item_count * 0.5:
        variance_pct = None
        passed = True
        if verbose:
            print(
                f"  Reconcile {division.division_name}: "
                f"items={item_count} ({raw_only_count} raw-value-only), "
                f"skipping numeric reconciliation [✓ PASS — raw values]"
            )
    elif ext_total is not None and ext_total > 0 and calc_total > 0:
        variance_pct = abs(ext_total - calc_total) / ext_total * 100
        passed = variance_pct <= threshold_pct
        if verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            var_str = f"{variance_pct:.1f}%"
            print(
                f"  Reconcile {division.division_name}: "
                f"extracted={ext_total}, calculated={calc_total:.2f}, "
                f"items={item_count}, variance={var_str} [{status}]"
            )
    elif ext_total is not None and ext_total > 0 and calc_total == 0:
        variance_pct = 100.0
        passed = False
        if verbose:
            print(
                f"  Reconcile {division.division_name}: "
                f"extracted={ext_total}, calculated=0.00, "
                f"items={item_count}, variance=100.0% [✗ FAIL]"
            )
    else:
        if verbose:
            print(
                f"  Reconcile {division.division_name}: "
                f"extracted={ext_total}, calculated={calc_total:.2f}, "
                f"items={item_count}, variance=N/A [✓ PASS]"
            )

    return ReconciliationResult(
        division_name=division.division_name or "UNKNOWN",
        extracted_total=ext_total,
        calculated_total=calc_total,
        item_count=item_count,
        variance_pct=variance_pct,
        passed=passed,
    )


def reconcile_boq(
    boq: BOQ,
    threshold_pct: float = 3.0,
    verbose: bool = False,
) -> list[ReconciliationResult]:
    """
    Reconcile all divisions in a BOQ.

    Returns a list of ReconciliationResult, one per division.
    """
    results = []
    for div in boq.divisions:
        results.append(reconcile_division(div, threshold_pct, verbose))
    return results


# =============================================================================
# PARALLEL EXTRACTION - Much faster for multi-sheet BOQs
# =============================================================================

def extract_boq_parallel(
    sheets_data: list[tuple[str, str]],
    provider: str = "vertex",
    model: str | None = None,
    verbose: bool = False,
    chunk_threshold: int = 28000,
    max_concurrent: int = 10,
    reference_items_by_division: dict[str, list[dict]] | None = None,
) -> BOQ:
    """
    Extract BOQ by processing sheets IN PARALLEL for much faster extraction.
    
    This can be 3-5x faster than sequential processing for multi-sheet BOQs.
    
    Args:
        sheets_data: List of (sheet_name, sheet_text) tuples
        provider: LLM provider name
        model: Model name
        verbose: Print debug info
        chunk_threshold: Character threshold for chunked extraction (default 28k so
            typical division segments stay single-call; avoids 2–3x extra LLM calls)
        max_concurrent: Maximum concurrent LLM calls (default 10)
        reference_items_by_division: Optional dict mapping division names
            to lists of template item dicts for guided vendor extraction
        
    Returns:
        Combined BOQ object
    """
    return asyncio.run(_extract_boq_parallel_async(
        sheets_data, provider, model, verbose, chunk_threshold, max_concurrent,
        reference_items_by_division=reference_items_by_division,
    ))


# ---------------------------------------------------------------------------
# Streaming extraction — yields per-division results as they complete
# ---------------------------------------------------------------------------

@dataclass
class DivisionResult:
    """Result of extracting a single division from one file."""
    label: str            # "TEMPLATE", "PTE", or vendor name
    division_name: str
    division: Division
    duration_seconds: float
    item_count: int


async def extract_divisions_streaming(
    sheets_data: list[tuple[str, str]],
    provider: str = "vertex",
    model: str | None = None,
    verbose: bool = False,
    chunk_threshold: int = 28000,
    max_concurrent: int = 10,
    reference_items_by_division: dict[str, list[dict]] | None = None,
    pre_extracted_summary: tuple[str | None, str | None, dict[str, str]] | None = None,
):
    """Async generator that yields DivisionResult as each division completes.

    Unlike ``extract_boq_parallel`` which returns a full BOQ at the end, this
    streams results incrementally so the server can pipeline vendor extraction
    and emit per-division SSE events.

    Args:
        sheets_data: List of (sheet_name, sheet_text) tuples
        provider: LLM provider name
        model: Model name
        verbose: Print debug info
        chunk_threshold: Char threshold for chunked extraction
        max_concurrent: Max concurrent LLM calls
        reference_items_by_division: Template reference items for guided extraction
        pre_extracted_summary: (project_name, project_total, division_totals)
            from fast regex extraction to skip LLM summary calls

    Yields:
        DivisionResult for each successfully extracted division
    """
    import time as _time

    default_models = {
        "openai": "gpt-4-turbo",
        "anthropic": "claude-3-sonnet-20240229",
        "google": "gemini-2.5-flash",
        "vertex": "gemini-2.5-flash",
    }
    model_name = model or default_models.get(provider, "gemini-2.5-flash")

    if provider == "vertex":
        llm_provider = ServiceAccountProvider(model=model_name)
    elif provider == "openai":
        llm_provider = OpenAIProvider(model=model_name)
    elif provider == "anthropic":
        llm_provider = AnthropicProvider(model=model_name)
    elif provider == "google":
        llm_provider = GoogleProvider(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    schema = load_schema()

    # Use pre-extracted summary if available, otherwise do fast regex + LLM fallback
    project_name = None
    project_total = None
    division_totals_map: dict[str, str] = {}

    if pre_extracted_summary:
        project_name, project_total, division_totals_map = pre_extracted_summary
        if verbose:
            print(f"Using pre-extracted summary: {len(division_totals_map)} divisions", flush=True)

    if len(division_totals_map) < 2:
        for sheet_name, sheet_text in sheets_data:
            if not sheet_text.strip() or not _is_summary_sheet(sheet_name):
                continue
            pn, pt, dt = _extract_summary_totals_fast(sheet_name, sheet_text, verbose)
            if pn and not project_name:
                project_name = pn
            if pt and not project_total:
                project_total = pt
            if dt:
                division_totals_map.update(dt)

    if len(division_totals_map) < 2:
        for sheet_name, sheet_text in sheets_data:
            if not sheet_text.strip() or not _is_summary_sheet(sheet_name):
                continue
            pn, pt, dt = _extract_summary_totals(llm_provider, sheet_name, sheet_text, verbose)
            if pn and not project_name:
                project_name = pn
            if pt and not project_total:
                project_total = pt
            if dt:
                division_totals_map.update(dt)

    # Build content sheets and expand multi-division sheets
    _MIN_CONTENT_CHARS = 200
    content_sheets = []
    for name, text in sheets_data:
        if not _is_content_sheet(name):
            continue
        sanitized = _sanitize_sheet_text(text)
        if not sanitized.strip() or len(sanitized.strip()) < _MIN_CONTENT_CHARS:
            continue
        content_sheets.append((name, sanitized))

    if division_totals_map and len(division_totals_map) > len(content_sheets):
        expanded: list[tuple[str, str]] = []
        for sheet_name, sheet_text in content_sheets:
            segments = split_content_by_divisions(
                sheet_text, list(division_totals_map.keys()), verbose=verbose,
            )
            if len(segments) >= 2:
                expanded.extend(segments)
            else:
                expanded.append((sheet_name, sheet_text))
        content_sheets = expanded

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _do_extract(sheet_name: str, sheet_text: str, ref: list[dict] | None) -> Division | None:
        async with semaphore:
            return await _extract_sheet_async(
                llm_provider, sheet_name, sheet_text, schema,
                chunk_threshold, division_totals_map, verbose, max_concurrent,
                reference_items=ref,
            )

    # Launch all divisions concurrently, yield as each completes
    pending: dict[asyncio.Task, tuple[str, str]] = {}
    for name, text in content_sheets:
        ref = None
        if reference_items_by_division:
            sn_upper = name.upper()
            for ref_div_name, ref_items in reference_items_by_division.items():
                rdn_upper = ref_div_name.upper()
                if rdn_upper == sn_upper or rdn_upper in sn_upper or sn_upper in rdn_upper:
                    ref = ref_items
                    break
        t0 = _time.time()
        task = asyncio.create_task(_do_extract(name, text, ref))
        pending[task] = (name, str(t0))

    while pending:
        done, _ = await asyncio.wait(pending.keys(), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            sheet_name, t0_str = pending.pop(task)
            t0 = float(t0_str)
            dt = _time.time() - t0
            try:
                division = task.result()
            except Exception as e:
                if verbose:
                    print(f"  Error extracting {sheet_name}: {e}", flush=True)
                continue
            if division is None:
                continue
            item_count = sum(_count_items_in_grouping(g) for g in division.grouping)
            yield DivisionResult(
                label="",
                division_name=division.division_name or sheet_name,
                division=division,
                duration_seconds=round(dt, 1),
                item_count=item_count,
            )


async def _extract_boq_parallel_async(
    sheets_data: list[tuple[str, str]],
    provider: str,
    model: str | None,
    verbose: bool,
    chunk_threshold: int,
    max_concurrent: int,
    reference_items_by_division: dict[str, list[dict]] | None = None,
) -> BOQ:
    """Async implementation of parallel BOQ extraction."""
    if verbose:
        print(f"\nStarting extraction: {len(sheets_data)} sheets (summary first, then content in parallel)", flush=True)

    default_models = {
        "openai": "gpt-4-turbo",
        "anthropic": "claude-3-sonnet-20240229",
        "google": "gemini-2.5-flash",
        "vertex": "gemini-2.5-flash",
    }
    
    model_name = model or default_models.get(provider, "gemini-2.5-flash")
    
    # Initialize provider
    if provider == "vertex":
        llm_provider = ServiceAccountProvider(model=model_name)
    elif provider == "openai":
        llm_provider = OpenAIProvider(model=model_name)
    elif provider == "anthropic":
        llm_provider = AnthropicProvider(model=model_name)
    elif provider == "google":
        llm_provider = GoogleProvider(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    schema = load_schema()
    
    project_name = None
    project_total = None
    division_totals_map = {}
    
    # First: Extract summary totals from ALL summary sheets.
    # Use fast regex extraction first; fall back to LLM only if regex yields < 2 divisions.
    for sheet_name, sheet_text in sheets_data:
        if not sheet_text.strip():
            continue
        if _is_summary_sheet(sheet_name):
            if verbose:
                print(f"Fast-extracting summary from: {sheet_name}", flush=True)
            extracted_name, extracted_total, div_totals = _extract_summary_totals_fast(
                sheet_name, sheet_text, verbose
            )
            if extracted_name and not project_name:
                project_name = extracted_name
            if extracted_total and not project_total:
                project_total = extracted_total
            if div_totals:
                division_totals_map.update(div_totals)

    # Fall back to LLM summary if regex found < 2 divisions
    if len(division_totals_map) < 2:
        if verbose:
            print(f"  Regex found {len(division_totals_map)} divisions — falling back to LLM summary extraction", flush=True)
        for sheet_name, sheet_text in sheets_data:
            if not sheet_text.strip():
                continue
            if _is_summary_sheet(sheet_name):
                extracted_name, extracted_total, div_totals = _extract_summary_totals(
                    llm_provider, sheet_name, sheet_text, verbose
                )
                if extracted_name and not project_name:
                    project_name = extracted_name
                if extracted_total and not project_total:
                    project_total = extracted_total
                if div_totals:
                    division_totals_map.update(div_totals)
    
    # Filter and sanitize content sheets. Use a low min length (50) so we only drop
    # near-empty sheets; a 300-char minimum was dropping real content and cutting
    # template line items from ~1100 to ~478.
    _MIN_CONTENT_CHARS = 200
    content_sheets = []
    for name, text in sheets_data:
        if not _is_content_sheet(name):
            continue
        sanitized = _sanitize_sheet_text(text)
        if not sanitized.strip() or len(sanitized.strip()) < _MIN_CONTENT_CHARS:
            continue
        content_sheets.append((name, sanitized))

    # ── Multi-division sheet detection & segmentation ──────────────────
    # If the summary lists more divisions than we have content sheets,
    # the main content is likely packed into one (or few) large sheets.
    # Split those sheets by the known division names so each division
    # gets its own, smaller LLM call.
    if division_totals_map and len(division_totals_map) > len(content_sheets):
        expanded: list[tuple[str, str]] = []
        for sheet_name, sheet_text in content_sheets:
            segments = split_content_by_divisions(
                sheet_text, list(division_totals_map.keys()), verbose=verbose
            )
            if len(segments) >= 2:
                if verbose:
                    print(f"\n✂ Sheet '{sheet_name}' segmented into {len(segments)} divisions:")
                    for seg_name, seg_text in segments:
                        print(f"    → {seg_name} ({len(seg_text):,} chars)")
                expanded.extend(segments)
            else:
                expanded.append((sheet_name, sheet_text))
        content_sheets = expanded
    
    if verbose:
        print(f"\nExtracting {len(content_sheets)} sheets/segments in PARALLEL (max {max_concurrent} concurrent). This may take 2–8 min depending on size...", flush=True)
    
    # Create semaphore to limit concurrent calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Process all sheets in parallel
    async def process_sheet(sheet_name: str, sheet_text: str) -> Optional[Division]:
        # Look up reference items for this division name
        ref = None
        if reference_items_by_division:
            # Try exact match first, then substring match
            sn_upper = sheet_name.upper()
            for ref_div_name, ref_items in reference_items_by_division.items():
                rdn_upper = ref_div_name.upper()
                if rdn_upper == sn_upper or rdn_upper in sn_upper or sn_upper in rdn_upper:
                    ref = ref_items
                    if verbose:
                        print(f"  📎 {sheet_name}: matched {len(ref)} reference items from '{ref_div_name}'")
                    break

        async with semaphore:
            return await _extract_sheet_async(
                llm_provider, sheet_name, sheet_text, schema, 
                chunk_threshold, division_totals_map, verbose, max_concurrent,
                reference_items=ref,
            )
    
    # Run all extractions concurrently
    tasks = [
        process_sheet(name, text) 
        for name, text in content_sheets
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect successful divisions
    divisions = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            if verbose:
                print(f"  Error in {content_sheets[i][0]}: {result}")
        elif result is not None:
            divisions.append(result)
    
    # Project total: use extracted value from summary sheet only.
    # If not available, SUM extracted division totals (never recalculate from line items).
    final_project_total = project_total
    if not final_project_total:
        total_amount = 0
        for div in divisions:
            if div.division_total_amount:
                try:
                    total_amount += float(str(div.division_total_amount).replace(",", ""))
                except (ValueError, TypeError):
                    pass
        if total_amount > 0:
            final_project_total = f"{total_amount:.2f}"
    
    boq = BOQ(
        project_name=project_name,
        contractor=None,
        project_total_amount=final_project_total,
        divisions=divisions,
    )

    if verbose:
        total_items = sum(
            sum(_count_items_in_grouping(g) for g in d.grouping)
            for d in divisions
        )
        print(f"\n✓ Extracted {len(divisions)} divisions, {total_items} total items")

        # Final reconciliation summary
        recon_results = reconcile_boq(boq, threshold_pct=2.0, verbose=False)
        passed = sum(1 for r in recon_results if r.passed)
        failed = len(recon_results) - passed
        print(f"\n── Reconciliation Summary ──")
        print(f"  {passed}/{len(recon_results)} divisions within 2% variance")
        if failed > 0:
            print(f"  ⚠ {failed} division(s) exceed 2% variance:")
            for r in recon_results:
                if not r.passed:
                    var_str = f"{r.variance_pct:.1f}%" if r.variance_pct is not None else "N/A"
                    print(f"    • {r.division_name}: extracted={r.extracted_total}, "
                          f"calculated={r.calculated_total:.2f}, variance={var_str}")

    # ── Structural validation against template reference ─────────────
    if reference_items_by_division and verbose:
        _log_structural_validation(boq, reference_items_by_division)

    return boq


def _log_structural_validation(
    boq: BOQ,
    reference: dict[str, list[dict]],
) -> None:
    """Compare an extracted BOQ against the template reference structure.

    Prints a diagnostic summary showing how well the extraction matches the
    template in terms of divisions, groupings, and item counts.  This is
    purely informational — it does not modify the BOQ.
    """
    ref_div_names = set(k.upper() for k in reference)
    ext_div_names = set(
        (d.division_name or "").upper() for d in boq.divisions
    )

    matched_divs = ref_div_names & ext_div_names
    missing_divs = ref_div_names - ext_div_names
    extra_divs = ext_div_names - ref_div_names

    ref_total_items = sum(len(v) for v in reference.values())
    ext_total_items = sum(
        sum(_count_items_in_grouping(g) for g in d.grouping)
        for d in boq.divisions
    )

    # Per-division item comparison
    per_div: list[str] = []
    for ref_name, ref_items in reference.items():
        ref_upper = ref_name.upper()
        # Find matching extracted division
        ext_div = next(
            (d for d in boq.divisions if (d.division_name or "").upper() == ref_upper),
            None,
        )
        if ext_div is None:
            per_div.append(f"    ✗ {ref_name}: MISSING (expected {len(ref_items)} items)")
            continue

        ext_count = sum(_count_items_in_grouping(g) for g in ext_div.grouping)
        ref_count = len(ref_items)
        pct = (ext_count / ref_count * 100) if ref_count > 0 else 0
        symbol = "✓" if abs(pct - 100) <= 30 else "⚠"
        per_div.append(
            f"    {symbol} {ref_name}: {ext_count}/{ref_count} items ({pct:.0f}%)"
        )

    item_pct = (ext_total_items / ref_total_items * 100) if ref_total_items > 0 else 0

    print(f"\n── Structural Validation vs Template ──")
    print(
        f"  Divisions: {len(matched_divs)}/{len(ref_div_names)} matched"
        + (f", {len(missing_divs)} missing" if missing_divs else "")
        + (f", {len(extra_divs)} extra" if extra_divs else "")
    )
    print(f"  Items: {ext_total_items}/{ref_total_items} ({item_pct:.1f}%)")
    for line in per_div:
        print(line)


def _build_targeted_fix_prompt(
    sheet_name: str,
    sheet_text: str,
    prev_groupings: list[Grouping],
    expected_total: float,
    calculated_total: float,
    reference_items: list[dict] | None = None,
) -> str:
    """Build a targeted fix prompt for reconciliation failures.

    Instead of re-extracting the entire sheet, this tells the LLM:
    - What we already extracted (grouping summary + totals)
    - The expected total and the gap
    - Asks it to find missing items or correct wrong amounts
    """
    gap = expected_total - calculated_total
    gap_pct = (abs(gap) / expected_total * 100) if expected_total > 0 else 0

    # Summarize what we already have
    existing_summary_lines = []
    for grp in prev_groupings:
        grp_items = sum(len(sg.line_items) for sg in grp.sub_groupings)
        grp_total = 0.0
        for sg in grp.sub_groupings:
            for it in sg.line_items:
                if it.item_amount is not None:
                    grp_total += it.item_amount
                elif it.item_quantity is not None and it.item_rate is not None:
                    grp_total += it.item_quantity * it.item_rate
        existing_summary_lines.append(
            f"  - {grp.grouping_name}: {grp_items} items, subtotal={grp_total:,.2f}"
            + (f" (grouping_total={grp.grouping_total_amount})" if grp.grouping_total_amount else "")
        )

    ref_block = _build_reference_block(reference_items)

    return f'''You are a construction BOQ expert. A previous extraction of this sheet had a TOTAL VARIANCE issue.

SHEET: {sheet_name}

EXPECTED DIVISION TOTAL: {expected_total:,.2f}
CALCULATED FROM ITEMS:   {calculated_total:,.2f}
GAP:                     {gap:,.2f} ({gap_pct:.1f}%)

PREVIOUS EXTRACTION SUMMARY:
{chr(10).join(existing_summary_lines)}

The gap of {gap:,.2f} means {"items are MISSING or amounts are TOO LOW" if gap > 0 else "some amounts are DOUBLED or TOO HIGH"}.

YOUR TASK: Re-extract ALL line items from this sheet. Pay special attention to:
{"- Items near the end of the sheet that may have been missed" if gap > 0 else "- Items where amount may have been doubled (set to qty*rate instead of the actual Amount column value)"}
- "GENERALLY" sections with provisional sums, allowances, and contingencies
- Items WITHOUT item codes but WITH amounts
- The "To Collection" subtotals must be verified against item sums

{ref_block}Return the COMPLETE division JSON (same format as before):
{{
  "division_name": "ACTUAL NAME",
  "division_total_amount": "{expected_total:.2f}",
  "grouping": [...]
}}

ACCURACY: The SUM of all item_amount values MUST equal approximately {expected_total:,.2f}. Check your arithmetic.

CONTENT:
{sheet_text}

Return ONLY valid JSON.'''


async def _extract_sheet_async(
    llm_provider: BaseLLMProvider,
    sheet_name: str,
    sheet_text: str,
    schema: dict,
    chunk_threshold: int,
    division_totals_map: dict,
    verbose: bool,
    max_concurrent: int,
    reconcile_threshold: float = 2.0,
    max_retries: int = 2,
    reference_items: list[dict] | None = None,
) -> Optional[Division]:
    """Extract a single sheet asynchronously with smart reconciliation.

    Strategy:
      Attempt 1 — full extraction (standard prompt)
      Attempt 2 — targeted fix prompt if variance > threshold, giving the LLM
                  the previous result + the gap to fill
    The attempt with the lowest variance is kept.
    """

    sheet_text = _sanitize_sheet_text(sheet_text)
    if verbose:
        print(f"  → Starting: {sheet_name} ({len(sheet_text)} chars)")

    # ── Pre-compute anchored total from raw text (regex, no LLM) ──
    anchored_total = _extract_carried_total(sheet_text)

    best_division: Division | None = None
    best_variance: float | None = None
    prev_groupings: list[Grouping] | None = None
    prev_calc_total: float = 0.0

    for attempt in range(1, max_retries + 1):
        try:
            division_total_from_llm = None
            division_name_from_llm = None

            loop = asyncio.get_event_loop()

            # ── Attempt 2+: Use targeted fix prompt if we have a previous result ──
            use_targeted_fix = (
                attempt > 1
                and prev_groupings is not None
                and anchored_total is not None
                and best_variance is not None
                and best_variance > reconcile_threshold
            )

            if use_targeted_fix:
                if verbose:
                    print(f"  ⟳ {sheet_name} attempt {attempt}: targeted fix "
                          f"(gap: expected={anchored_total}, got={prev_calc_total:.2f})")
                prompt = _build_targeted_fix_prompt(
                    sheet_name, sheet_text, prev_groupings,
                    float(anchored_total.replace(",", "")), prev_calc_total,
                    reference_items=reference_items,
                )
                with ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(
                        executor, llm_provider.complete, prompt
                    )
                data, parse_err = _parse_llm_json_response(response)
                if data is None:
                    raise ValueError(parse_err or "Invalid JSON from targeted fix")
                raw_groupings = _parse_groupings_from_response(data)
                groupings = merge_chunk_groupings(raw_groupings)
                division_total_from_llm = data.get("division_total_amount")
                division_name_from_llm = data.get("division_name")

            elif len(sheet_text) > chunk_threshold:
                groupings, division_name_from_llm = await _extract_sheet_chunked_async(
                    llm_provider, sheet_name, sheet_text, verbose, max_concurrent,
                    chunk_threshold=chunk_threshold,
                    reference_items=reference_items,
                )
            else:
                prompt = get_sheet_extraction_prompt(
                    sheet_text, sheet_name, schema,
                    reference_items=reference_items,
                )

                with ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(
                        executor, llm_provider.complete, prompt
                    )

                data, parse_err = _parse_llm_json_response(response)
                if data is None:
                    raise ValueError(parse_err or "Invalid JSON")
                raw_groupings = _parse_groupings_from_response(data)
                groupings = merge_chunk_groupings(raw_groupings)
                division_total_from_llm = data.get("division_total_amount")
                division_name_from_llm = data.get("division_name")

            if not groupings:
                if verbose:
                    print(f"  ✗ {sheet_name} attempt {attempt}: no groupings extracted")
                continue

            # ── Resolve division total (prefer anchored regex total) ──
            div_total = None

            if anchored_total:
                div_total = anchored_total
                if verbose and attempt == 1:
                    print(f"  Total for {sheet_name}: {anchored_total} (anchored from text)")

            if not div_total:
                for summary_div_name, summary_total in division_totals_map.items():
                    if (summary_div_name.upper() == sheet_name.upper()
                            or summary_div_name.upper() in sheet_name.upper()
                            or sheet_name.upper() in summary_div_name.upper()):
                        div_total = summary_total
                        if verbose and attempt == 1:
                            print(f"  Total for {sheet_name}: {summary_total} (from summary)")
                        break

            if not div_total and division_total_from_llm:
                div_total = str(division_total_from_llm)
                if verbose and attempt == 1:
                    print(f"  Total for {sheet_name}: {division_total_from_llm} (from LLM)")

            division_name = _resolve_division_name(
                sheet_name, sheet_text, division_totals_map, division_name_from_llm
            )

            division = Division(
                division_name=division_name,
                division_total_amount=div_total,
                grouping=groupings,
            )

            _correct_division_doubled_amounts(division)

            # ── Reconciliation check ──────────────────────────────────
            recon = reconcile_division(division, threshold_pct=reconcile_threshold, verbose=verbose)

            cur_var = recon.variance_pct if recon.variance_pct is not None else 0.0
            if best_division is None or (best_variance is not None and cur_var < best_variance):
                best_division = division
                best_variance = cur_var
                prev_groupings = groupings
                prev_calc_total = recon.calculated_total

            if recon.passed:
                total_items = sum(_count_items_in_grouping(g) for g in groupings)
                if verbose:
                    print(f"  ✓ Done: {sheet_name} - {total_items} items, "
                          f"total={div_total}, variance={cur_var:.1f}% [PASS]")
                return division

            if verbose:
                print(f"  ⟳ {sheet_name} attempt {attempt}/{max_retries}: "
                      f"variance {cur_var:.1f}% > {reconcile_threshold}%")

        except Exception as e:
            if verbose:
                print(f"  ✗ {sheet_name} attempt {attempt} error: {e}")
            continue

    if best_division and verbose:
        total_items = sum(_count_items_in_grouping(g) for g in best_division.grouping)
        var_str = f"{best_variance:.1f}%" if best_variance is not None else "N/A"
        print(f"  ⚠ {sheet_name}: best variance {var_str} after {max_retries} attempts — "
              f"keeping best result ({total_items} items)")

    return best_division


async def _extract_sheet_chunked_async(
    llm_provider: BaseLLMProvider,
    sheet_name: str,
    sheet_text: str,
    verbose: bool,
    max_concurrent: int,
    chunk_threshold: int = 28000,
    reference_items: list[dict] | None = None,
) -> tuple[list[Grouping], str | None]:
    """Extract items from a large sheet using parallel chunk processing.
    
    Returns:
        Tuple of (groupings, division_name_from_llm)
    """
    chunks = split_sheet_into_chunks(sheet_text, max_chars=chunk_threshold)
    
    if verbose:
        print(f"    Chunked: {sheet_name} → {len(chunks)} chunks (parallel)")
    
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(max_concurrent)
    division_name_from_llm = None
    
    async def process_chunk(chunk: str, chunk_idx: int) -> tuple[list[Grouping], str | None]:
        async with semaphore:
            prompt = get_chunk_extraction_prompt(
                chunk, sheet_name, chunk_idx, len(chunks),
                reference_items=reference_items,
            )
            for attempt in range(2):  # initial + 1 retry on parse failure
                with ThreadPoolExecutor() as executor:
                    response = await loop.run_in_executor(
                        executor, llm_provider.complete, prompt
                    )
                data, parse_err = _parse_llm_json_response(response)
                if data is not None:
                    groupings = _parse_groupings_from_response(data)
                    div_name = data.get("division_name")
                    return groupings, div_name
                if attempt == 0 and verbose:
                    print(f"    Chunk {chunk_idx} JSON parse failed ({parse_err}), retrying once...")
            if verbose:
                print(f"    Chunk {chunk_idx} failed after retry: {parse_err}")
            raise ValueError(parse_err or "Invalid JSON")
    
    # Process all chunks in parallel
    tasks = [process_chunk(chunk, i+1) for i, chunk in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect successful results
    all_groupings = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            if verbose:
                print(f"    Chunk {i+1} error: {result}")
        else:
            groupings, div_name = result
            all_groupings.extend(groupings)
            # Use first valid division name we find
            if div_name and not division_name_from_llm:
                division_name_from_llm = div_name
    
    return merge_chunk_groupings(all_groupings), division_name_from_llm
