# AutoTransFlow

> Layout-preserving multilingual document translation system. Translates PDFs into 200 languages while maintaining exact spatial structure — zero external API calls, all local inference.

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/NLLB--200-Facebook-FF9D00?style=flat-square)](https://huggingface.co/facebook/nllb-200-distilled-600M)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

---

## The Problem

Standard Chartered handles financial documents, contracts, and legal papers in dozens of languages. These documents need to be translated quickly — but traditional translation approaches destroy the document structure.

A financial contract with tables, headings, and signature blocks rendered as a plain unformatted text block is **legally unusable**. Layout is not decoration — it is part of the document's legal meaning.

**Why tools like pdfplumber/PyMuPDF fail:**
These libraries extract text as a linear stream. They cannot differentiate a heading from a paragraph, a table cell from a caption, or understand where content sits spatially on the page. The result is a long wall of text with no structure.

**Why external APIs like Google Translate were not used:**
Standard Chartered has a strict data privacy policy. Financial documents cannot be sent to external cloud services. All processing must run on internal servers.

---

## The Solution

Treat document translation as a **computer vision problem first, not a text problem**.

1. Convert each PDF page to a high-resolution image
2. Use a YOLO-based document layout model to detect exactly where every text block sits
3. Extract text only from within detected bounding boxes
4. Translate each block using NLLB-200 (local inference)
5. Place translated text back at the exact original coordinates

The document looks identical to the original — just in a different language.

---

## Architecture

```
Input PDF
    │
    ▼
┌──────────────────────────┐
│  Pre-processing          │
│  • Create duplicate PDF  │
│  • BabelDoc font embed   │
│    (target language font)│
│  • Page → PIXMAP image   │
│  • Normalise [0,255]→[0,1]│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  doc-layout-yolo         │
│  Layout Detection Model  │
│  • Detects text blocks   │
│  • Returns bounding boxes│
│  • Confidence scores     │
│  • Filters < 0.25 conf.  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Text Extraction         │
│  • Crop each bounding box│
│  • Extract text content  │
│  • Preserve block order  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  NLLB-200 Translation    │
│  facebook/nllb-200-      │
│  distilled-600M          │
│  • Local inference only  │
│  • forced_bos_token_id   │
│    steers to target lang │
│  • 200 language pairs    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Re-rendering            │
│  • Translated text placed│
│    at original coordinates│
│  • Font-embedded duplicate│
│    PDF updated           │
│  • Original + translated │
│    appended together     │
└────────────┬─────────────┘
             │
             ▼
    Final PDF Output
    (original + translated
     side by side)
```

---

## Key Technical Decisions

### Why doc-layout-yolo instead of simple text extraction?
Document layout is a 2D problem, not a 1D problem. A heading at the top of a page and a footnote at the bottom are both "text" but have completely different meanings. YOLO-based detection understands the visual hierarchy of a page — it returns class IDs (heading, paragraph, table, caption) and precise bounding box coordinates for each element. This spatial understanding is what makes re-rendering possible.

### How NLLB-200 language steering works
NLLB-200 (No Language Left Behind) is a sequence-to-sequence translation model. Translation to a specific target language is controlled via `forced_bos_token_id` — forcing the first generated token to be the target language code (e.g. `eng_Latn` for English, `hin_Deva` for Hindi). All subsequent tokens are then generated in that language.

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def translate(text: str, source_lang: str, target_lang: str) -> str:
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt")
    target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=target_lang_id
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
```

### Language codes reference
```
English:   eng_Latn
Hindi:     hin_Deva
Arabic:    arb_Arab
Chinese:   zho_Hans
French:    fra_Latn
German:    deu_Latn
Spanish:   spa_Latn
Japanese:  jpn_Jpan
```

---

## Project Structure

```
autotransflow/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml              # Language codes, confidence thresholds
├── src/
│   ├── preprocessing/
│   │   ├── pdf_to_pixmap.py     # PDF page → image conversion
│   │   └── font_embedder.py    # BabelDoc font embedding per language
│   ├── layout/
│   │   └── detector.py          # doc-layout-yolo inference + filtering
│   ├── translation/
│   │   ├── nllb_translator.py  # NLLB-200 local translation
│   │   └── argos_translator.py # Alternative: Argos Translate backend
│   ├── rendering/
│   │   └── rerenderer.py        # Place translated text at coordinates
│   └── pipeline.py             # End-to-end pipeline orchestration
├── tests/
│   ├── test_detector.py
│   ├── test_translation.py
│   └── test_pipeline.py
├── notebooks/
│   ├── 01_layout_detection_demo.ipynb
│   └── 02_translation_quality_eval.ipynb
└── Dockerfile
```

---

## Installation

```bash
git clone https://github.com/harshalbhambhani/autotransflow
cd autotransflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Requirements

```
torch==2.0.0
transformers==4.30.0
sentencepiece==0.1.99
PyMuPDF==1.22.3
Pillow==9.5.0
ultralytics==8.0.120
numpy==1.24.3
tqdm==4.65.0
pyyaml==6.0
```

---

## Usage

```python
from src.pipeline import AutoTransFlow

pipeline = AutoTransFlow(
    source_lang="eng_Latn",
    target_lang="hin_Deva",
    confidence_threshold=0.25
)

output_path = pipeline.translate(
    input_pdf="financial_contract_english.pdf",
    output_pdf="financial_contract_hindi.pdf"
)
print(f"Translated document saved to: {output_path}")
```

---

## Supported Languages (sample)

English, Hindi, Arabic, Chinese (Simplified/Traditional), French, German, Spanish, Japanese, Korean, Russian, Portuguese, Turkish, Bengali, Urdu, Tamil, Telugu, and 180+ more.

---

*Built at Standard Chartered Global Business Services · Aug–Nov 2025*
*Harshal Bhambhani · BITS Hyderabad · 2026*