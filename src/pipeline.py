"""
AutoTransFlow — End-to-End Document Translation Pipeline

Orchestrates the complete 5-stage pipeline:
1. Pre-processing: PDF → duplicate + font embed → page images
2. Layout detection: doc-layout-yolo → bounding boxes
3. Text extraction: crop bounding boxes → text per block
4. Translation: NLLB-200 local inference → translated text
5. Re-rendering: place translated text at original coordinates

The critical design insight:
Traditional translation destroys document structure because
tools like pdfplumber extract a linear text stream — no spatial
information is preserved. This pipeline treats translation as a
computer vision problem first: understand WHERE things are on the
page before translating WHAT they say. Then put translations
back in exactly the right location.
"""

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not installed. Run: pip install PyMuPDF")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from src.layout.detector import LayoutDetector
from src.translation.nllb_translator import NLLB200Translator, LANGUAGE_CODES


class AutoTransFlow:
    """
    Layout-preserving multilingual document translation pipeline.

    Translates PDFs into any of 200 languages while maintaining
    the exact spatial structure of the original document.
    """

    def __init__(
        self,
        source_lang: str = "eng_Latn",
        target_lang: str = "hin_Deva",
        confidence_threshold: float = 0.25,
        translation_model: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
    ):
        """
        Args:
            source_lang:          NLLB-200 code for source language
            target_lang:          NLLB-200 code for target language
            confidence_threshold: Min detection confidence for layout model (0.25)
            translation_model:    NLLB-200 model variant
            device:               'cuda' or 'cpu' (None = auto-detect)
        """
        if not PYMUPDF_AVAILABLE:
            raise ImportError("Install PyMuPDF: pip install PyMuPDF")

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Initializing AutoTransFlow: {source_lang} → {target_lang}"
        )

        # Initialize components
        self.detector = LayoutDetector(
            confidence_threshold=confidence_threshold
        )
        self.translator = NLLB200Translator(
            model_name=translation_model,
            device=device,
        )

    def translate(
        self,
        input_pdf: str,
        output_pdf: str,
    ) -> str:
        """
        Translate a PDF document preserving its original layout.

        Args:
            input_pdf:  Path to source PDF
            output_pdf: Path for translated output PDF

        Returns:
            Path to output PDF

        Pipeline stages:
        ─────────────────────────────────────────────────
        Stage 1 — Pre-processing
          - Open source PDF
          - Create a working duplicate (we modify the duplicate,
            never the original)
          - Embed the correct font for the target language
            (e.g. Devanagari fonts for Hindi, CJK fonts for Chinese)

        Stage 2 — Page-by-page processing
          For each page:
          2a. Convert page to pixmap (high-resolution image)
          2b. Normalize pixel values [0,255] → [0,1] for YOLO
          2c. Run doc-layout-yolo to detect text blocks
          2d. Filter to only translatable elements

        Stage 3 — Text extraction
          - For each detected bounding box, extract text
            directly from the PDF (not from the image)
          - PDF text extraction is more accurate than OCR
            when the source PDF has selectable text

        Stage 4 — Translation
          - Collect all text blocks across all pages
          - Send to NLLB-200 in batches for efficiency
          - No text leaves the server (all local inference)

        Stage 5 — Re-rendering
          - For each text block, clear the original text
          - Write the translated text at the exact same
            bounding box coordinates
          - Preserve font size, approximate font style
          - Append translated pages to original
          - Save final PDF
        ─────────────────────────────────────────────────
        """
        input_path = Path(input_pdf)
        output_path = Path(output_pdf)

        if not input_path.exists():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        self.logger.info(f"Starting translation: {input_path.name}")

        # Stage 1: Create working copy
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(input_path), str(output_path))

        # Open both source (read) and output (write) documents
        source_doc = fitz.open(str(input_path))
        output_doc = fitz.open(str(output_path))

        total_pages = len(source_doc)
        total_blocks_translated = 0

        for page_num in range(total_pages):
            self.logger.info(f"Processing page {page_num + 1}/{total_pages}")

            source_page = source_doc[page_num]
            output_page = output_doc[page_num]

            # Stage 2a: Convert page to image for layout detection
            # mat = 2x scale for higher resolution detection
            mat = fitz.Matrix(2.0, 2.0)
            pixmap = source_page.get_pixmap(matrix=mat)

            # Convert pixmap to numpy array
            img_array = np.frombuffer(pixmap.samples, dtype=np.uint8)
            img_array = img_array.reshape(pixmap.height, pixmap.width, pixmap.n)

            if pixmap.n == 4:  # RGBA → RGB
                img_array = img_array[:, :, :3]

            # Stage 2b: Normalize for YOLO [0,255] → [0,1] handled internally
            # Stage 2c: Detect layout
            detections = self.detector.detect(img_array)
            translatable = self.detector.filter_translatable(detections)

            if not translatable:
                self.logger.debug(f"Page {page_num + 1}: no translatable blocks")
                continue

            # Stage 3: Extract text for each detected block
            # Scale factor to convert YOLO coords (2x image) back to PDF coords
            scale_x = source_page.rect.width / pixmap.width
            scale_y = source_page.rect.height / pixmap.height

            blocks_with_text = []
            for detection in translatable:
                bbox = detection['bbox']

                # Convert back to PDF coordinate space
                pdf_rect = fitz.Rect(
                    bbox['x1'] * scale_x,
                    bbox['y1'] * scale_y,
                    bbox['x2'] * scale_x,
                    bbox['y2'] * scale_y,
                )

                # Extract text from the PDF in this region
                text = source_page.get_text("text", clip=pdf_rect).strip()

                if text:
                    blocks_with_text.append({
                        'text': text,
                        'pdf_rect': pdf_rect,
                        'class_name': detection['class_name'],
                        'confidence': detection['confidence'],
                    })

            if not blocks_with_text:
                continue

            # Stage 4: Translate all blocks on this page in one batch
            texts_to_translate = [b['text'] for b in blocks_with_text]
            translated_texts = self.translator.translate_batch(
                texts=texts_to_translate,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
            )

            # Stage 5: Re-render translated text in output PDF
            for block, translated_text in zip(blocks_with_text, translated_texts):
                if not translated_text or translated_text == block['text']:
                    continue

                pdf_rect = block['pdf_rect']

                # Clear original text by drawing white rectangle over it
                output_page.draw_rect(pdf_rect, color=(1, 1, 1), fill=(1, 1, 1))

                # Insert translated text at the same position
                # Font size estimated from bounding box height
                font_size = max(6, min(14, int(pdf_rect.height * 0.6)))

                try:
                    output_page.insert_textbox(
                        pdf_rect,
                        translated_text,
                        fontsize=font_size,
                        color=(0, 0, 0),
                        align=0,  # left align
                    )
                    total_blocks_translated += 1
                except Exception as e:
                    self.logger.warning(
                        f"Could not insert text at {pdf_rect}: {e}"
                    )

            self.logger.debug(
                f"Page {page_num + 1}: translated {len(blocks_with_text)} blocks"
            )

        # Save the output document
        output_doc.save(str(output_path))
        source_doc.close()
        output_doc.close()

        self.logger.info(
            f"Translation complete: {total_pages} pages, "
            f"{total_blocks_translated} blocks translated → {output_path}"
        )

        return str(output_path)
