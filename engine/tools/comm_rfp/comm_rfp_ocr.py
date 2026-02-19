"""
Google Cloud Vision OCR module for scanned/image-based PDFs.

This module provides OCR capabilities using Google Cloud Vision API
to extract text from scanned PDFs or image files.

Setup:
    1. Create a Google Cloud project
    2. Enable the Cloud Vision API
    3. Create a service account and download the JSON key
    4. Set GOOGLE_APPLICATION_CREDENTIALS environment variable:
       export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

Usage:
    from ocr import GoogleOCR
    
    ocr = GoogleOCR()
    text = ocr.extract_from_pdf("scanned_boq.pdf")
    # or
    text = ocr.extract_from_image("page.png")
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator

# PDF to image conversion
try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

# Google Cloud Vision
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False


@dataclass
class OCRPage:
    """
    Represents OCR results for a single page.

    Attributes:
        page_number: 1-indexed page number
        text: Extracted text content
        confidence: Overall confidence score (0-1)
        blocks: List of text blocks with positions (for future table detection)
    """
    page_number: int
    text: str
    confidence: float = 0.0
    blocks: list[dict] = None

    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []


@dataclass  
class OCRResult:
    """
    Complete OCR result for a document.

    Attributes:
        pages: List of OCR results per page
        full_text: Combined text from all pages
        source_file: Path to the source file
    """
    pages: list[OCRPage]
    source_file: str
    
    @property
    def full_text(self) -> str:
        """Get combined text from all pages."""
        return "\n\n".join(page.text for page in self.pages)
    
    @property
    def page_count(self) -> int:
        """Get total number of pages."""
        return len(self.pages)


class GoogleOCR:
    """
    Google Cloud Vision OCR client for text extraction.

    This client handles both image files and PDF documents,
    converting PDFs to images when necessary.

    Attributes:
        client: Google Cloud Vision client
        dpi: DPI for PDF to image conversion (higher = better quality but slower)
    """

    def __init__(self, dpi: int = 300):
        """
        Initialize the Google OCR client.

        Args:
            dpi: Resolution for PDF to image conversion (default: 300)

        Raises:
            ImportError: If google-cloud-vision is not installed
            EnvironmentError: If credentials are not configured
        """
        if not VISION_AVAILABLE:
            raise ImportError(
                "google-cloud-vision is required for OCR. "
                "Install with: pip install google-cloud-vision"
            )

        # Check for credentials
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            raise EnvironmentError(
                "GOOGLE_APPLICATION_CREDENTIALS environment variable not set. "
                "Set it to the path of your service account JSON key file."
            )

        self.client = vision.ImageAnnotatorClient()
        self.dpi = dpi

    def extract_from_image(self, image_path: str | Path) -> OCRPage:
        """
        Extract text from a single image file.

        Args:
            image_path: Path to the image file (PNG, JPEG, etc.)

        Returns:
            OCRPage with extracted text

        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image content
        with open(image_path, "rb") as f:
            content = f.read()

        return self._process_image_content(content, page_number=1)

    def extract_from_image_bytes(self, image_bytes: bytes, page_number: int = 1) -> OCRPage:
        """
        Extract text from image bytes.

        Args:
            image_bytes: Raw image bytes
            page_number: Page number for the result

        Returns:
            OCRPage with extracted text
        """
        return self._process_image_content(image_bytes, page_number)

    def _process_image_content(self, content: bytes, page_number: int) -> OCRPage:
        """
        Process image content through Google Vision API.

        Args:
            content: Image bytes
            page_number: Page number for the result

        Returns:
            OCRPage with extracted text and metadata
        """
        image = vision.Image(content=content)

        # Use document_text_detection for better results on documents
        response = self.client.document_text_detection(image=image)

        if response.error.message:
            raise RuntimeError(f"Vision API error: {response.error.message}")

        # Extract full text
        full_text = ""
        if response.full_text_annotation:
            full_text = response.full_text_annotation.text

        # Extract blocks for potential table detection
        blocks = []
        if response.full_text_annotation:
            for page in response.full_text_annotation.pages:
                for block in page.blocks:
                    block_text = ""
                    for paragraph in block.paragraphs:
                        for word in paragraph.words:
                            word_text = "".join(
                                symbol.text for symbol in word.symbols
                            )
                            block_text += word_text + " "
                    
                    # Get bounding box
                    vertices = block.bounding_box.vertices
                    blocks.append({
                        "text": block_text.strip(),
                        "bounds": {
                            "x1": vertices[0].x,
                            "y1": vertices[0].y,
                            "x2": vertices[2].x,
                            "y2": vertices[2].y,
                        },
                        "confidence": block.confidence if hasattr(block, 'confidence') else 0.0
                    })

        # Calculate average confidence
        confidence = 0.0
        if blocks:
            confidence = sum(b.get("confidence", 0) for b in blocks) / len(blocks)

        return OCRPage(
            page_number=page_number,
            text=full_text,
            confidence=confidence,
            blocks=blocks
        )

    def extract_from_pdf(self, pdf_path: str | Path) -> OCRResult:
        """
        Extract text from a PDF using OCR.

        Converts each PDF page to an image and runs OCR.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            OCRResult with text from all pages

        Raises:
            ImportError: If pdf2image is not installed
            FileNotFoundError: If PDF doesn't exist
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image is required for PDF OCR. "
                "Install with: pip install pdf2image\n"
                "Also requires poppler: brew install poppler (macOS) "
                "or apt install poppler-utils (Linux)"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)

        # Process each page
        pages = []
        for page_num, image in enumerate(images, start=1):
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            # Run OCR
            page_result = self.extract_from_image_bytes(img_bytes, page_num)
            pages.append(page_result)

        return OCRResult(
            pages=pages,
            source_file=str(pdf_path)
        )

    def extract_from_pdf_streaming(
        self, 
        pdf_path: str | Path
    ) -> Iterator[OCRPage]:
        """
        Extract text from PDF pages one at a time (memory efficient).

        Yields OCRPage results as each page is processed.

        Args:
            pdf_path: Path to the PDF file

        Yields:
            OCRPage for each page
        """
        if not PDF2IMAGE_AVAILABLE:
            raise ImportError(
                "pdf2image is required for PDF OCR. "
                "Install with: pip install pdf2image"
            )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Get page count first
        from pdf2image import pdfinfo_from_path
        info = pdfinfo_from_path(pdf_path)
        page_count = info.get("Pages", 0)

        # Process one page at a time
        for page_num in range(1, page_count + 1):
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.dpi,
                first_page=page_num,
                last_page=page_num
            )

            if images:
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format="PNG")
                img_bytes = img_byte_arr.getvalue()

                yield self.extract_from_image_bytes(img_bytes, page_num)


def extract_text_with_ocr(pdf_path: str | Path, dpi: int = 300) -> str:
    """
    Convenience function to extract text from a PDF using OCR.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion

    Returns:
        Extracted text from all pages
    """
    ocr = GoogleOCR(dpi=dpi)
    result = ocr.extract_from_pdf(pdf_path)
    return result.full_text


# TODO: Add support for Google Cloud Vision async batch processing for large PDFs
# TODO: Add table detection using block positions
# TODO: Add support for handwriting recognition mode
# TODO: Add caching of OCR results to avoid re-processing
# TODO: Add support for other OCR providers (AWS Textract, Azure Computer Vision)
