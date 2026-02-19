"""
PDF text extraction module with OCR fallback support.

This module handles the extraction of text content from BOQ PDF documents.
Supports both text-based PDFs (via pdfplumber) and scanned PDFs (via Google OCR).

The extraction process:
1. Opens the PDF file
2. Attempts text extraction with pdfplumber
3. Falls back to OCR if text extraction yields insufficient content
4. Returns extracted text or lines

Usage:
    from parse_pdf import extract_text, extract_lines_from_pdf
    
    # Simple text extraction (auto-detects best method)
    text = extract_text("path/to/boq.pdf", use_ocr=True)
    
    # Line-by-line extraction
    lines = extract_lines_from_pdf("path/to/boq.pdf")
"""

from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto

import pdfplumber
from pdfplumber.page import Page


class ExtractionMethod(Enum):
    """PDF text extraction method."""
    PDFPLUMBER = auto()
    GOOGLE_OCR = auto()


@dataclass
class ExtractedLine:
    """
    Represents a single line of text extracted from a PDF.

    Attributes:
        text: The raw text content of the line
        page_number: The 1-indexed page number where the line was found
        line_number: The 0-indexed line number within the page
        is_empty: Whether the line contains only whitespace
    """
    text: str
    page_number: int
    line_number: int
    is_empty: bool = field(init=False)

    def __post_init__(self):
        self.is_empty = not self.text.strip()

    @property
    def stripped(self) -> str:
        """Return the line with leading/trailing whitespace removed."""
        return self.text.strip()


@dataclass
class ExtractedTable:
    """
    Represents a table extracted from a PDF page.

    Attributes:
        rows: List of rows, each row is a list of cell values
        page_number: The page number where the table was found
        table_index: The index of the table on the page (0-indexed)
    """
    rows: list[list[str | None]]
    page_number: int
    table_index: int


@dataclass
class ExtractionResult:
    """
    Result of PDF text extraction.

    Attributes:
        text: Full extracted text
        lines: List of extracted lines
        method: Extraction method used
        page_count: Number of pages in the PDF
    """
    text: str
    lines: list[ExtractedLine]
    method: ExtractionMethod
    page_count: int


class PDFParser:
    """
    Parser for extracting text content from BOQ PDF documents.

    This parser uses pdfplumber for text extraction with optional
    Google OCR fallback for scanned documents.

    Attributes:
        pdf_path: Path to the PDF file
        _pdf: The pdfplumber PDF object (lazily loaded)
    """

    # Minimum text length to consider extraction successful
    MIN_TEXT_THRESHOLD = 100

    def __init__(self, pdf_path: str | Path):
        """
        Initialize the PDF parser.

        Args:
            pdf_path: Path to the PDF file to parse

        Raises:
            FileNotFoundError: If the PDF file does not exist
        """
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        self._pdf: pdfplumber.PDF | None = None

    def __enter__(self) -> "PDFParser":
        """Context manager entry."""
        self._pdf = pdfplumber.open(self.pdf_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        if self._pdf:
            self._pdf.close()
            self._pdf = None

    @property
    def pdf(self) -> pdfplumber.PDF:
        """
        Get the pdfplumber PDF object.

        Returns:
            The opened PDF object

        Raises:
            RuntimeError: If the PDF hasn't been opened via context manager
        """
        if self._pdf is None:
            raise RuntimeError(
                "PDF not opened. Use PDFParser as a context manager: "
                "with PDFParser(path) as parser: ..."
            )
        return self._pdf

    @property
    def page_count(self) -> int:
        """Return the number of pages in the PDF."""
        return len(self.pdf.pages)

    def extract_lines(self) -> list[ExtractedLine]:
        """
        Extract all text lines from the PDF using pdfplumber.

        Iterates through all pages and extracts text line-by-line.
        Empty lines are preserved to maintain document structure.

        Returns:
            List of ExtractedLine objects containing the text and metadata

        Note:
            Lines are split using newline characters. Some PDFs may have
            inconsistent line breaks, which may require post-processing.
        """
        all_lines: list[ExtractedLine] = []

        for page_num, page in enumerate(self.pdf.pages, start=1):
            page_text = page.extract_text() or ""
            lines = page_text.split("\n")

            for line_num, line_text in enumerate(lines):
                extracted_line = ExtractedLine(
                    text=line_text,
                    page_number=page_num,
                    line_number=line_num
                )
                all_lines.append(extracted_line)

        return all_lines

    def extract_full_text(self) -> str:
        """
        Extract full text from all pages.

        Returns:
            Combined text from all pages
        """
        texts = []
        for page in self.pdf.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n\n".join(texts)

    def extract_text_by_page(self) -> dict[int, str]:
        """
        Extract text organized by page number.

        Returns:
            Dictionary mapping page numbers (1-indexed) to full page text
        """
        return {
            page_num: page.extract_text() or ""
            for page_num, page in enumerate(self.pdf.pages, start=1)
        }

    def extract_tables(self) -> list[ExtractedTable]:
        """
        Extract tables from all pages of the PDF.

        Uses pdfplumber's table detection capabilities to find and
        extract tabular data.

        Returns:
            List of ExtractedTable objects

        Note:
            Table detection accuracy varies significantly based on PDF
            structure. Manual verification is recommended.
        """
        all_tables: list[ExtractedTable] = []

        for page_num, page in enumerate(self.pdf.pages, start=1):
            tables = page.extract_tables() or []

            for table_idx, table_data in enumerate(tables):
                if table_data:
                    extracted_table = ExtractedTable(
                        rows=table_data,
                        page_number=page_num,
                        table_index=table_idx
                    )
                    all_tables.append(extracted_table)

        return all_tables

    def get_page(self, page_number: int) -> Page:
        """
        Get a specific page from the PDF.

        Args:
            page_number: 1-indexed page number

        Returns:
            The pdfplumber Page object

        Raises:
            IndexError: If page number is out of range
        """
        if page_number < 1 or page_number > self.page_count:
            raise IndexError(
                f"Page {page_number} out of range. "
                f"PDF has {self.page_count} pages."
            )
        return self.pdf.pages[page_number - 1]

    def is_scanned(self) -> bool:
        """
        Check if the PDF appears to be scanned (image-based).

        Returns:
            True if the PDF appears to be scanned
        """
        text = self.extract_full_text()
        return len(text.strip()) < self.MIN_TEXT_THRESHOLD


def extract_lines_from_pdf(pdf_path: str | Path) -> list[ExtractedLine]:
    """
    Convenience function to extract lines from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of extracted lines
    """
    with PDFParser(pdf_path) as parser:
        return parser.extract_lines()


def extract_text(
    pdf_path: str | Path,
    use_ocr: bool = False,
    ocr_dpi: int = 300,
    force_ocr: bool = False
) -> ExtractionResult:
    """
    Extract text from a PDF file with optional OCR fallback.

    This is the recommended method for BOQ extraction as it handles
    both text-based and scanned PDFs.

    Args:
        pdf_path: Path to the PDF file
        use_ocr: Enable OCR fallback for scanned PDFs
        ocr_dpi: DPI for OCR image conversion
        force_ocr: Force OCR even if text extraction works

    Returns:
        ExtractionResult with text, lines, and metadata

    Raises:
        ImportError: If OCR is requested but dependencies not installed
    """
    pdf_path = Path(pdf_path)

    # First, try pdfplumber extraction
    with PDFParser(pdf_path) as parser:
        page_count = parser.page_count

        if not force_ocr:
            text = parser.extract_full_text()
            lines = parser.extract_lines()

            # Check if we got sufficient text
            if len(text.strip()) >= PDFParser.MIN_TEXT_THRESHOLD:
                return ExtractionResult(
                    text=text,
                    lines=lines,
                    method=ExtractionMethod.PDFPLUMBER,
                    page_count=page_count
                )

    # Fall back to OCR if enabled
    if use_ocr or force_ocr:
        try:
            from tools.comm_rfp.comm_rfp_ocr import GoogleOCR
        except ImportError as e:
            raise ImportError(
                "OCR dependencies not installed. Install with:\n"
                "  pip install google-cloud-vision pdf2image\n"
                "And set GOOGLE_APPLICATION_CREDENTIALS environment variable."
            ) from e

        ocr = GoogleOCR(dpi=ocr_dpi)
        result = ocr.extract_from_pdf(pdf_path)

        # Convert OCR result to lines
        lines = []
        for page in result.pages:
            page_lines = page.text.split("\n")
            for line_num, line_text in enumerate(page_lines):
                lines.append(ExtractedLine(
                    text=line_text,
                    page_number=page.page_number,
                    line_number=line_num
                ))

        return ExtractionResult(
            text=result.full_text,
            lines=lines,
            method=ExtractionMethod.GOOGLE_OCR,
            page_count=result.page_count
        )

    # No OCR and insufficient text
    with PDFParser(pdf_path) as parser:
        return ExtractionResult(
            text=parser.extract_full_text(),
            lines=parser.extract_lines(),
            method=ExtractionMethod.PDFPLUMBER,
            page_count=parser.page_count
        )


def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing whitespace and removing artifacts.

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    # Normalize multiple spaces to single space
    text = re.sub(r" +", " ", text)
    # Remove common PDF artifacts
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


# TODO: Add support for encrypted/password-protected PDFs
# TODO: Add page rotation detection and handling
# TODO: Add support for extracting embedded images