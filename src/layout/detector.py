"""
Document Layout Detection using doc-layout-yolo

Detects text blocks in PDF page images and returns their
bounding box coordinates for structure-preserving translation.

Why computer vision over text extraction:
- Libraries like pdfplumber/PyMuPDF extract text as a linear stream
- They lose all spatial information (where on the page things are)
- Financial documents, contracts, and legal papers have rigid layouts
- Heading vs paragraph vs table cell vs footnote — all are "text" but
  mean different things in different positions
- A translated financial contract must look identical to the original
  for legal and compliance validity

Why doc-layout-yolo over a custom detector:
- Pre-trained on massive diverse document dataset
- Recognises heading, paragraph, table, figure, caption, footnote etc.
- Using it out-of-box saves months of annotation and training time
- Proven in research and production document processing pipelines
"""

import logging
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning(
        "ultralytics not installed. Run: pip install ultralytics"
    )

# Class IDs from doc-layout-yolo
# Maps numeric class IDs to human-readable element types
LAYOUT_CLASSES = {
    0: "title",
    1: "plain_text",
    2: "abandon",       # decorative or non-content elements
    3: "figure",
    4: "figure_caption",
    5: "table",
    6: "table_caption",
    7: "table_footnote",
    8: "isolate_formula",
    9: "formula_caption",
}

# Classes that contain translatable text
TRANSLATABLE_CLASSES = {
    "title", "plain_text", "figure_caption",
    "table_caption", "table_footnote"
}


class LayoutDetector:
    """
    Detects document layout elements in PDF page images.

    Returns bounding boxes with coordinates for each detected
    text block, filtered by confidence threshold.
    """

    def __init__(
        self,
        model_path: str = "juliozhao/DocLayout-YOLO-DocStructBench",
        confidence_threshold: float = 0.25,
    ):
        """
        Args:
            model_path:           Path or HuggingFace ID for doc-layout-yolo
            confidence_threshold: Minimum confidence to accept a detection.
                                  0.25 filters low-confidence noise while
                                  retaining all meaningful text blocks.
                                  Detections below this are structural noise
                                  (watermarks, borders, artifacts).
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Install ultralytics: pip install ultralytics"
            )

        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Loading doc-layout-yolo from: {model_path}")
        self.model = YOLO(model_path)
        self.logger.info("Layout detection model loaded.")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all layout elements in a PDF page image.

        Args:
            image: Page image as numpy array (H, W, C), pixel values [0, 255]
                   Obtained from pixmap conversion of PDF page.

        Returns:
            List of detected element dicts, sorted top-to-bottom, left-to-right:
            {
                'class_id':    int,    # numeric class from YOLO
                'class_name':  str,    # human-readable type
                'confidence':  float,  # detection confidence
                'bbox':        {       # pixel coordinates on page
                    'x1': int,         # top-left x
                    'y1': int,         # top-left y
                    'x2': int,         # bottom-right x
                    'y2': int,         # bottom-right y
                    'width': int,
                    'height': int,
                },
                'translatable': bool,  # whether this element should be translated
            }

        Why sort by position:
        - Reading order in legal documents is top-to-bottom, left-to-right
        - Maintaining reading order ensures translated text is inserted
          in the correct sequence when re-rendering the document
        """
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4) x1y1x2y2
            confidences = result.boxes.conf.cpu().numpy()  # (N,)
            class_ids = result.boxes.cls.cpu().numpy().astype(int)  # (N,)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)
                class_name = LAYOUT_CLASSES.get(cls_id, f"class_{cls_id}")

                detections.append({
                    'class_id': int(cls_id),
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': x1, 'y1': y1,
                        'x2': x2, 'y2': y2,
                        'width': x2 - x1,
                        'height': y2 - y1,
                    },
                    'translatable': class_name in TRANSLATABLE_CLASSES,
                })

        # Sort by vertical position (top-to-bottom), then horizontal (left-to-right)
        detections.sort(key=lambda d: (d['bbox']['y1'], d['bbox']['x1']))

        self.logger.debug(
            f"Detected {len(detections)} elements "
            f"({sum(1 for d in detections if d['translatable'])} translatable)"
        )

        return detections

    def filter_translatable(self, detections: List[Dict]) -> List[Dict]:
        """
        Return only detections that contain translatable text.

        Why filter:
        - Figure images, table borders, decorative elements should not
          be sent to the translation model
        - Only text-containing elements get translated and re-rendered
        - Non-text elements are preserved as-is from the original PDF
        """
        return [d for d in detections if d['translatable']]
