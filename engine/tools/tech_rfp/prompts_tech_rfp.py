import json
from utils.core.log import get_logger
from typing import Dict, Any, List, Optional, Tuple
from google.genai.types import Part, Content


def summarize_rfp_prompt_library() -> tuple[Dict[str, str], Dict[str, dict]]:
    """Return dictionaries of prompt-text and JSON-schemas -- for rfp_summarizer tool."""
    prompts = {
        "rfpDates": """
                Extract and return the following key dates from the RFP document. If a date is not explicitly mentioned, do NOT include the key in the response. Use yyyy-mm-dd format for the dates. Use the following JSON structure for your response:
                {
                    "rfpDates": {
                        "rfpRelease": "The release date of the RFP",
                        "vendorQuestionsDeadline": "The deadline for vendors to submit their questions",
                        "midTenderInterviews": "The date(s) of mid-tender interviews",
                        "proposalSubmissionDeadline": "The final deadline for proposal submissions"
                    }
                }
                Ensure your output strictly follows the JSON format above and does not include keys for missing dates.
            """,
        "scopeOfWork": """
                Extract and summarize the scope of work (otherwise known as works or works information) of the RFP. The first sentence should be an overall summary of the scope of work. Then, the next sentence in a new line should read, "The scope of work includes but is not limited to:" and list the scopes in bullet point format. **Bold** important words/phrases that need to be highlighted.
                {
                    "scopeOfWork": "a reasonably detailed scope of work here"
                }
                Ensure your output strictly follows the JSON format above.
            """,
        "rfp_with_details": """
            "rfpDetails" : "
            Extract all critical information from the RFP that will be used to evaluate tender proposals. Your output should include, but not be limited to:
            - **Key Dates:** Identify all relevant dates such as the RFP release, deadline for vendor questions, mid-tender interview dates, and proposal submission deadlines.
            - **Scope of Work:** Provide a comprehensive summary of the scope of work, outlining the main objectives and deliverables.
            - **Evaluation Criteria:** Detail the criteria that will be used to assess the proposals, including any specific requirements, performance indicators, or scoring metrics.
            - **Submission Guidelines:** Summarize all instructions related to how proposals should be submitted, including format, required documentation, and any special submission protocols.
            - **Additional Details:** Capture any extra information that might be important for the evaluation process, such as contractual terms, mandatory qualifications, or specific evaluation notes.
            Ensure that your output is clear and comprehensive, written in a narrative format that highlights and organizes the most important information for a thorough tender evaluation.
                "
            """,
    }
    schemas = {
        "rfpDates": {
            "type": "object",
            "properties": {
                "rfpDates": {
                    "type": "object",
                    "properties": {
                        "rfpRelease": {"type": "string", "format": "date"},
                        "vendorQuestionsDeadline": {"type": "string", "format": "date"},
                        "midTenderInterviews": {"type": "string", "format": "date"},
                        "proposalSubmissionDeadline": {
                            "type": "string",
                            "format": "date",
                        },
                    },
                }
            },
            "required": ["rfpDates"],
        },
        "scopeOfWork": {
            "type": "object",
            "properties": {"scopeOfWork": {"type": "string"}},
            "required": ["scopeOfWork"],
        },
    }
    return prompts, schemas


def build_eval_extractor_prompt() -> str:

    prompt = """
You are an expert at parsing and restructuring data from Excel files.
The table generally contains these columns:

- 'Evaluation Scope' (or 'evaluation_scope' / 'scope' / 'section' / 'theme')
- 'Evaluation Breakdown' (sometimes labeled 'Criterion' or similar)
- 'Weight' (a numeric value)
- 'Description' (optional)

The table given to you might contain one more column: "Description".
Your task is to extract **all valid rows** and convert them into a structured JSON object with the following rules:

Output Format:
OUTPUT FORMAT (include description if it exists for that row):
{
  "<Evaluation Scope 1>": {
    "<Evaluation Breakdown 1>": {
      "weight": "<Weight-as-string>",
      "description": "<Description-string>"
    },
    "<Evaluation Breakdown 2>": {
      "weight": "<Weight-as-string>"
      // (omit "description" if not present)
    }
  },
  "<Evaluation Scope 2>": {
    ...
  }
}

VERY IMPORTANT:
- Do **not** add extra keys, lists, arrays, or metadata.
- If multiple rows share the same Evaluation Scope, group them together under that scope.
- If 'Description' is missing/empty for a row, simply omit the "description" field for that item.
- Use the **exact values** as found in the table. Do **not** reword, summarize, or clean them.
- Keep all weights as **strings** (e.g., "0.05", not 0.05 or 5%).

DO NOT:
- Add any explanations, comments, headings, or markdown.
- Alter the format or add nested objects not described above.
- Repeat any "Evaluation Scope" keys in your JSON. If multiple rows belong to the same scope, combine all criteria under a single key. Duplicate keys will be lost when parsed.

Output should be:
- Valid JSON
- Only the content described above
- Consistent with the format and raw values in the source table
        """
    return prompt


def build_ptc_prompt(evaluation_block: str) -> Tuple[str, dict[str, Any]]:
    """
    Return the full PTC prompt for a single evaluator block. For tech_rfp_ptc tool
    """
    ptc_prompt = f"""
You are a senior tender evaluator preparing **Post-Tender Clarification (PTC)** comments in the style commonly used by master developers.

You are reviewing a previously completed scoring evaluation of a contractor's submission. The evaluation includes:
- Evaluator reasoning / notes
- Score / grade
- Evidence list citing documents and page numbers

Your task: decide whether a PTC is required and, if yes, write the PTC comment in the **same style and tone** as real tender PTC sheets.

---

## When should you generate a PTC?
Generate a PTC only if the evidence indicates one of the following:
- **MISSING_INFO**: a required document/appendix/detail is referenced or expected but not provided.
- **INCOMPLETE**: information is provided but is vague, partial, unrealistic, not project-specific, or lacks required detail.
If the submission is poor and clarification would not reasonably change the evaluation, set **refType = "N/A"** and **queryDescription = "N/A"**.

---

## Required writing style for queryDescription
If refType is not "N/A", your `queryDescription` MUST:
1) Be written as a **multi-line numbered list** using this format exactly:
   - Line 1 starts with `1. `
   - Line 2 starts with `2. `, etc.
2) Use directive contractual phrasing such as:
   - `Bidder to ...`
   - `The Contractor is required to ...`
   - `The Contractor shall ...`
   - `Please submit ...`
   - `Revise and resubmit ...`
3) Be specific and actionable (request exactly what is missing / what must be revised).
4) Where relevant, include compliance language used in PTC sheets, e.g.:
   - `Provide compliant bid.`
   - `To be signed and stamped.`
   - `No further action required.` (ONLY if fully compliant; but if fully compliant you should return N/A instead.)
5) If the evidence clearly indicates missing signature/stamp on a submitted letter/appendix, include as the FIRST point:
   `Appendix letter: Signed but not stamped – please submit signed and stamped copy.`
   (Only include this if supported by the evidence.)

### Do NOT do the following:
- Do not mention the score, grade, or evaluator thought process.
- Do not invent facts.
- Do not reference documents/pages that are not present in the provided evidence.
- Do not output anything except the JSON object.

---

## Output format
Return exactly one JSON object with ONLY these fields:

{{
  "ptc": {{
    "queryDescription": "<Either 'N/A' OR a numbered multi-line PTC comment>",
    "refType": "<One of: 'N/A', 'MISSING_INFO', 'INCOMPLETE'>"
  }}
}}

Rules:
- If `refType` is `"N/A"`, `queryDescription` must be exactly `"N/A"`.
- If `refType` is `"MISSING_INFO"` or `"INCOMPLETE"`, `queryDescription` must be a numbered multi-line list starting at `1.`

---

## Evaluation to review:
{evaluation_block}
""".strip()

    ptc_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "ptc": {
                "type": "object",
                "properties": {
                    "queryDescription": {"type": "string"},
                    "refType": {
                        "type": "string",
                        "enum": ["N/A", "MISSING_INFO", "INCOMPLETE"],
                    },
                },
                "required": ["queryDescription", "refType"],
            }
        },
        "required": ["ptc"],
    }

    return ptc_prompt, ptc_schema


def generate_evaluation_criteria_prompt(rfp_text: str) -> str:
    """
    Contract-aware, model-reasoned technical rubric prompt.
    - The model FIRST reasons (silently) about contract type from the RFP text:
      {Construction Contractor (prime) | Subcontractor (package) | Consultant/Service Provider}.
    - It then generates a purely technical evaluation rubric tailored to that type.
    - Output is ONE JSON object only, matching your existing schema.
    """
    example = """{
  "Technical Solution": {
    "System architecture and scalability": { "weight": 20.0, "description": "Quality of design, modularity, performance headroom, and future extensibility." },
    "Safety and regulatory compliance":   { "weight": 10.0, "description": "Alignment with relevant codes/standards and completeness of safety analysis." }
  },
  "Team & Experience": {
    "Relevant team credentials":          { "weight": 12.0, "description": "Experience and certifications of key personnel aligned to scope." },
    "Comparable past projects":           { "weight": 10.0, "description": "Evidence of successful delivery on similar size/complexity and outcomes achieved." }
  },
  "Schedule & Delivery": {
    "Feasible work plan and resourcing":  { "weight": 16.0, "description": "Realism of milestones, critical path awareness, and adequacy of resources." }
  },
  "Local Context": {
    "Site/location constraints":          { "weight": 8.0,  "description": "Awareness of location-specific constraints, authorities approvals, and logistics." }
  }
}""".strip()

    # CONTRACT-TYPE GUIDES (used by the model for silent reasoning; do NOT change output schema)
    guides = """
CONTRACT-TYPE ADAPTATION (Reason silently; do NOT include this section in output):
- If the RFP is for a Construction Contractor (prime):
  Emphasize: construction methodology & sequencing; HSE controls; schedule integration with other trades; QA/QC systems; team capability & similar projects; risk & temporary works; commissioning readiness.
- If the RFP is for a Subcontractor (work package):
  Emphasize: technical compliance to the package specification; interfaces with adjacent systems/trades; materials/standards/shop drawings; ITPs & QA hold points; resource plan & fabrication/installation schedule; site logistics; HSE.
- If the RFP is for a Consultant / Service Provider:
  Emphasize: technical approach & methodology; key personnel qualifications & continuity; relevant experience & references; work plan & deliverables; tools/software & QA for deliverables; stakeholder & risk management.
Keyword cues (non-exhaustive, for internal reasoning only):
  - Consultant/Services: "Terms of Reference", ToR, consultancy, advisory, design services, supervision, feasibility, study.
  - Subcontractor: subcontract, work package, discipline scope (e.g., HVAC, façade, electrical), nominated subcontractor, shop drawings.
  - Contractor: main contractor, EPC/DB/DBB construction, prime contract, overall delivery.
"""

    prompt = (
        "You are an expert evaluator for construction procurement.\n"
        "First, reason silently to determine the contract type from the RFP text "
        "(Construction Contractor | Subcontractor | Consultant/Service Provider). "
        "Use the adaptation guides below to shape the technical focus. Do not output your reasoning.\n"
        "\n---\n"
        f"{guides}\n"
        "---\n"
        "### OUTPUT RULES (strict):\n"
        "- Return **one valid JSON object** only. No commentary before or after.\n"
        '- Schema: { <Scope>: { <Criterion>: { "weight": <number>, "description": "<2–4 sentence factual description aligned to the RFP>" } } }\n'
        "- Provide max 4 scopes total. All weights are numeric and sum to 100.\n"
        "- The rubric must be purely technical:\n"
        "  Exclude price/cost/commercial topics (fees, payment terms, insurances, warranties, contract conditions).\n"
        "  Keep schedule, QA/QC, HSE, risk, interfaces, methodology, deliverables, and team capability as technical aspects.\n"
        "- Include exactly one criterion that explicitly references local context / site constraints (authorities, logistics, environment).\n"
        "- Use concise, neutral, non-marketing language.\n"
        "\n---\n"
        "### Example JSON Shape (for structure only — adapt content/weights to the RFP and contract type):\n"
        f"{example}\n"
        "\n---\n"
        "### RFP TEXT:\n"
        f"{rfp_text}\n"
    )
    return prompt



GRADE_RUBRIC_TEXT = r"""
Non-Compliant  (Binary - use ONLY for criteria that are a simple yes/no or pass/fail checklist item)
- The submission does not meet the requirement or has not submitted a document for a checklist.

Compliant  (Binary — use ONLY for criteria that are a simple yes/no or pass/fail checklist item)
- The submission meets the requirement if they have submitted a document for a checklist.

Major Concerns
- Wholly unsatisfactory response and/or very poor approach
- Evidence/documentation not provided
- Major concerns about technical abilities
- Very low confidence in a successful outcome

Serious Concerns
- Predominantly unsatisfactory or incomplete response
- Evidence/documentation substantially incomplete
- Significant concerns about technical capability
- Very low confidence in a successful outcome

High Concerns
- Unsatisfactory in many respects
- Missing key evidence/documentation elements
- Clear concerns regarding technical ability
- Low confidence in a successful outcome

Minor-Moderate Concerns
- Satisfactory in some respects but negatively affected by gaps
- Evidence/documentation provided but significant elements missing
- Minor concerns about technical capability
- Low to moderate confidence in a successful outcome

Moderate Confidence
- Satisfactory in the majority of respects
- Evidence/documentation substantially complete and relevant
- Technical capability acceptable
- Moderate confidence in a successful outcome

Sound Confidence
- Clearly satisfactory response
- Evidence/documentation complete and relevant
- Sound technical capability demonstrated
- Moderate to good confidence in a successful outcome

Good Confidence
- Satisfactory and well-aligned with requirements
- Evidence/documentation complete and relevant
- Good technical capability
- Good confidence in a successful outcome

Very Good Confidence
- Full and robust response
- Evidence/documentation complete, relevant, and well-presented
- Strong technical capability
- Very good confidence in a successful outcome

Excellent Confidence
- Fully meets requirements and strongly supports the assessment
- Evidence/documentation comprehensive and clearly relevant
- Very strong technical capability
- High confidence in a successful outcome

Outstanding
- Exceeds the project requirements
- Evidence/documentation exemplary and adds clear value
- Outstanding technical capability
- Exceptional confidence in a successful outcome
""".strip()


def build_prompt_for_criterion_grade(
    criterion_name: str,
    weight: float,
    contractor_parts: List[Part],
    rfp_parts: List[Part],
    description: Optional[str] = None,
) -> Content:
    logger = get_logger()
    desc_blk = (
        f"\nEvaluation criterion description:\n{description}\n"
        if description
        else ""
    )

    instruction = (
        "You are an expert construction proposal evaluator reviewing a contractor submission against an RFP.\n"
        "You will receive two buckets of documents:\n"
        "1) RFP-REFERENCE (authoritative requirements)\n"
        "2) CONTRACTOR-SUBMISSION (bidder response)\n\n"
        "Task: For the criterion below, choose exactly ONE discrete Grade label from the rubric.\n"
        "Do NOT output any numeric percentages. Do NOT output a numeric score.\n\n"
        f"Criterion: {criterion_name}\n"
        f"{desc_blk}"
        "Use the grading rubric exactly as provided:\n"
        f"{GRADE_RUBRIC_TEXT}\n\n"
        "OUTPUT FORMAT (JSON only):\n"
        "{\n"
        '  "grade": "<One of: Non-Compliant | Compliant | Major Concerns | Serious Concerns | High Concerns | Minor-Moderate Concerns | '
        'Moderate Confidence | Sound Confidence | Good Confidence | Very Good Confidence | Excellent Confidence | Outstanding>",\n'
        '  "reasoning": [\n'
        '    {"text": "Why this grade fits (cite specific section/clause/paragraph numbers where possible).", "source": "<Exact filename from Start of File: header>", "pageNumber": <int or null>}\n'
        "  ]\n"
        "}\n\n"
        "BINARY CRITERIA RULE:\n"
        "- If the criterion is purely a yes/no compliance checklist item (e.g., 'Did the contractor submit document X?', "
        "'Is the contractor ISO-certified?'), or if in the evaluation criterion description, it is stated that the criteria is binary, use ONLY 'Compliant' or 'Non-Compliant'. "
        "You are not looking to assess quality, but rather to determine the presence or absence of a document. "
        "Do NOT use the 10-level grading scale for binary criteria.\n"
        "- For all other criteria that require qualitative judgment, use the 10-level grading scale "
        "(Major Concerns through Outstanding). Do NOT use Compliant/Non-Compliant for these.\n\n"
        "CONSTRAINTS:\n"
        "- Provide exactly one JSON object.\n"
        "- `reasoning` must be a list with 2 to 7 items total.\n"
        "- The first reasoning item must start exactly with: `Final Grade: <GRADE>.` and then justify why that grade fits.\n"
        "- If the chosen grade is NOT the highest possible (`Outstanding` or `Compliant`), the last reasoning item must be a cohesive paragraph explaining why the next higher grade was not awarded.\n"
        "- In `reasoning.text`, explicitly reference the specific section/clause/paragraph number where possible.\n"
        "- For the `source` key, use the EXACT filename as found in the document header (the text immediately following the 'Start of File:' tag). Do not invent or approximate filenames.\n"
        "- If the criterion is binary, and no relevant evidence is provided, choose 'Non-Compliant'. Otherwise (non-binary), choose 'Major Concerns'.\n"
        "- Do not invent evidence or filenames.\n"
    )

    message_parts = (
        [Part.from_text(text=instruction)]
        + [Part.from_text(text="### BEGIN RFP-REFERENCE\n")]
        + rfp_parts
        + [Part.from_text(text="### END RFP-REFERENCE\n")]
        + [Part.from_text(text="\n### BEGIN CONTRACTOR-SUBMISSION\n")]
        + contractor_parts
        + [Part.from_text(text="\n### END CONTRACTOR-SUBMISSION\n")]
    )

    logger.debug(f"Processing Criterion: {criterion_name} - Grade")
    return Content(role="user", parts=message_parts)


EVAL_SYS_PROMPT = (
    "You are a professional who reviews and evaluates tenders against request for proposals. "
    "Many of the clients/contractors are from Saudi Arabia (otherwise known as KSA). When reviewing, interpret acronyms and other terms accordingly. "
    "Be holistic, fair, and thoughtful in your evaluations - assessing the contractor's ability based solely on the documents provided. "
    "The tender files could include text and images. You are to review all documents. "
    "Use judgment as a human expert would."
)

SENIOR_SYS_PROMPT = (
    "You are the senior (final) evaluator. Read the full RFP, the "
    "contractor submission, and the two junior reviews, then issue a "
    "fresh, independent score and reasoning."
)
