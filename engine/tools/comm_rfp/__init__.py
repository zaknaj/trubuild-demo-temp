"""
Commercial RFP module - BOQ extraction, vendor comparison, and evaluation.

Submodules:
- comm_rfp_server: FastAPI server for streaming extraction and comparison
- comm_rfp_extract: LLM-based BOQ extraction (divisions, groupings, line items)
- comm_rfp_compare: Multi-vendor BOQ comparison and cross-vendor validation
- comm_rfp_models: Pydantic data models (BOQ, Division, Grouping, LineItem)
- comm_rfp_parse_excel: Excel file parsing via openpyxl
- comm_rfp_parse_pdf: PDF text extraction with OCR fallback
- comm_rfp_ocr: Google Cloud Vision OCR wrapper
- comm_rfp_build_json: State-based BOQ assembly from PDF text
"""
