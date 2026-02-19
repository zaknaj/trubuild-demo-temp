import json
import re
from typing import Any, Dict, Optional
from utils.core.log import get_logger

"""
Script intended for validating and correcting JSON from LLM output.
"""


def is_valid_json(input):
    try:
        json.loads(input)
        return True
    except json.JSONDecodeError:
        return False


def validate_and_correct_response(response):
    """Validate and correct the response format to ensure proper JSON structure and syntax."""
    logger = get_logger()
    try:
        response_json = json.loads(response)

        # If it's a list containing one object, extract it
        if isinstance(response_json, list):
            if len(response_json) == 1 and isinstance(response_json[0], dict):
                response_json = response_json[0]
            else:
                logger.debug("List must contain exactly one object.")
                return False, response

    except json.JSONDecodeError:
        logger.debug("Failed to parse JSON due to syntax error.")
        return False, response

    required_keys = ["clauses", "impact", "impactLevel", "riskType", "risk"]
    if not all(key in response_json for key in required_keys):
        logger.debug("Missing one or more required keys.")
        return False, response

    # normalize clauses
    if isinstance(response_json["clauses"], dict):
        response_json["clauses"] = [response_json["clauses"]]

    # valiadte clauses list
    if not isinstance(response_json["clauses"], list) or not all(
        "id" in clause and "summary" in clause for clause in response_json["clauses"]
    ):
        logger.debug("Invalid clause format.")
        return False, response

    # normalize riskType
    if not isinstance(response_json["riskType"], list):
        response_json["riskType"] = [response_json["riskType"]]

    return True, response_json


def parse_json_strings(json_strings):
    "Used in contract analyzer to parse JSON strings"
    logger = get_logger()

    json_objects = []
    for json_string in json_strings:
        try:
            json_object = json.loads(json_string)
            json_objects.append(json_object)
        except json.JSONDecodeError as e:
            logger.debug(f"Error parsing JSON string: {e}")
            logger.debug(f"Problematic JSON string: {json_string}")
            json_objects.append(None)

    return [obj for obj in json_objects if obj is not None]


def validate_source_response(response):
    "Used in contract review to validate response"
    logger = get_logger()

    required_keys = ["clauseNumber", "clauseText"]
    try:
        response_data = json.loads(response)

        if (
            isinstance(response_data, list)
            and len(response_data) == 1
            and isinstance(response_data[0], dict)
        ):
            response_data = response_data[0]

        if not isinstance(response_data, dict):
            logger.debug(f"Unexpected response format (not a dict): {response_data}")
            return False, "Invalid format: expected a JSON object."

        for key in required_keys:
            if key not in response_data:
                logger.debug(
                    f"Missing key: {key} in response data:\n{json.dumps(response_data, indent=2)}"
                )
                return False, f"Missing key: {key}"

        response_data["page"] = "Not Found"
        response_data["sourceDocumentName"] = "Not Found"

        if not isinstance(response_data["clauseNumber"], str):
            return False, "clauseNumber must be a string."
        if not isinstance(response_data["clauseText"], str):
            return False, "clauseText must be a string."

        return True, response_data

    except json.JSONDecodeError:
        logger.debug(f"Failed to parse response as JSON:\n{response}")
        return False, "Response is not valid JSON."


def clean_malformed_json(raw: str, *, label: Optional[str] = None) -> str:
    """
    Best‑effort scrub for common Gemini JSON glitches used by **all**
    TruBuild tools (PTC, tech‑RFP, commercial‑RFP …).

    The heuristics are idempotent – running twice is safe.
    """
    logger = get_logger()

    try:
        # fix '}, ], {' breaks in arrays
        raw = re.sub(r"\},\s*\],\s*\{", r"}, {", raw)

        # drop trailing commas before ] or }
        raw = re.sub(r",\s*([\]}])", r"\1", raw)

        # add missing brace before list‑terminator  … } ]
        raw = re.sub(r'("pageNumber"\s*:\s*\d+)\s*\]', r"\1}\n]", raw)

        # collapse three or more trailing braces
        raw = re.sub(r"\}\s*\}\s*\}\s*$", "}}", raw)

        # replace raw control characters (0x00‑0x1F) with space
        raw = re.sub(r"(?<!\\)[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", raw)

        return raw
    except Exception as e:
        logger.debug(f"[clean_malformed_json] ({label or 'json'}) failed: {e}")
        return raw


ALLOWED_GRADES = {
    "Non-Compliant",
    "Compliant",
    "Major Concerns",
    "Serious Concerns",
    "High Concerns",
    "Minor-Moderate Concerns",
    "Moderate Confidence",
    "Sound Confidence",
    "Good Confidence",
    "Very Good Confidence",
    "Excellent Confidence",
    "Outstanding",
}


def _valid_page_number(pn: Any) -> bool:
    """Allow None, numbers, or strings containing at least one digit."""
    if pn is None:
        return True
    if isinstance(pn, bool):
        return False
    if isinstance(pn, (int, float)):
        return True
    if isinstance(pn, str) and re.search(r"\d+", pn):
        return True
    return False


def validate_llm_response_tech_rfp(data: Dict[str, Any]) -> bool:
    """Validate Tech-RFP evaluator output (grade-based, optional numeric score)."""
    if not isinstance(data, dict):
        return False

    grade = data.get("grade")
    if not isinstance(grade, str) or grade.strip() not in ALLOWED_GRADES:
        return False

    if "score" in data:
        try:
            score = float(data["score"])
            if not (0 <= score <= 100):
                return False
        except (ValueError, TypeError):
            return False

    reasoning = data.get("reasoning")
    if not isinstance(reasoning, list):
        return False

    for item in reasoning:
        if not isinstance(item, dict):
            return False
        text = item.get("text")
        if not isinstance(text, str):
            return False

        source = item.get("source", None)
        if not (source is None or isinstance(source, str)):
            return False

        pn = item.get("pageNumber", None)
        if not _valid_page_number(pn):
            return False

    return True


def _coerce_json(value):
    """Return a Python object from storage: handles dict/list/str/bytes/None."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return json.loads(value.decode("utf-8"))
    if isinstance(value, str):
        return json.loads(value)
    return value


def main():
    from utils.core.log import pid_tool_logger, set_logger

    package_id = "system_check"
    logger = pid_tool_logger(package_id, "json_validation")
    set_logger(logger)

    print("JSONVAL TEST START")

    try:
        # 1. validate_and_correct_response
        raw_response = '{"clauses":[{"id":"1","summary":"Sample"}],"impact":"High","impactLevel":"3","riskType":"Legal","risk":"Breach"}'
        valid, corrected = validate_and_correct_response(raw_response)

        # 2. validate_source_response
        source_response = '{"clauseNumber":"5.2","clauseText":"Sample clause text."}'
        valid, parsed = validate_source_response(source_response)

        # 3. clean_malformed_json
        malformed = '{"score": 85, "reasoning": [{"text": "text", "source": "doc", "pageNumber": 1}, ],}'
        cleaned = clean_malformed_json(malformed)

        # 4. validate_llm_response_tech_rfp
        tech_data = {
            "score": 92,
            "reasoning": [{"text": "ok", "source": "doc.pdf", "pageNumber": 1}],
        }
        result = validate_llm_response_tech_rfp(tech_data)

        # 5. parse_json_strings
        json_inputs = ['{"a": 1, "b": 2}', '{"bad": true,,}', '{"valid": "yes"}']
        parsed_objs = parse_json_strings(json_inputs)

        print("JSONVAL OK")
        return True
    except Exception as e:
        print("JSONVAL ERROR")
        raise


if __name__ == "__main__":
    main()
