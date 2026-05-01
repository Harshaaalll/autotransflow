"""
NLLB-200 Translation Engine — Local Inference Only

Translates text blocks into 200 languages using Facebook's
No Language Left Behind (NLLB-200) model, running entirely
on internal servers with zero external API calls.

Why NLLB-200 over external APIs (Google Translate, DeepL):
- Standard Chartered data privacy policy prohibits sending financial
  documents to external cloud services
- Customer contracts, loan agreements, and legal documents cannot
  leave the bank's internal network
- NLLB-200 (facebook/nllb-200-distilled-600M) runs fully locally
- 200 languages including low-resource languages Google handles poorly
- The distilled-600M model is fast enough for batch document processing

How language steering works:
- NLLB-200 is a seq2seq transformer (encoder-decoder architecture)
- Target language is controlled via forced_bos_token_id
- The first generated token is forced to be the target language code
- All subsequent tokens are generated in that target language
- Example: forced_bos_token_id = tokenizer.convert_tokens_to_ids("hin_Deva")
  forces all output to be Hindi in Devanagari script

Supported language codes (sample):
  eng_Latn  →  English
  hin_Deva  →  Hindi
  arb_Arab  →  Arabic
  zho_Hans  →  Chinese (Simplified)
  fra_Latn  →  French
  deu_Latn  →  German
  spa_Latn  →  Spanish
  jpn_Jpan  →  Japanese
  kor_Hang  →  Korean
  rus_Cyrl  →  Russian
  urd_Arab  →  Urdu
  tam_Taml  →  Tamil
  tel_Telu  →  Telugu
"""

import logging
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Run: pip install transformers")


# Complete language code reference for NLLB-200
LANGUAGE_CODES: Dict[str, str] = {
    "english":              "eng_Latn",
    "hindi":                "hin_Deva",
    "arabic":               "arb_Arab",
    "chinese_simplified":   "zho_Hans",
    "chinese_traditional":  "zho_Hant",
    "french":               "fra_Latn",
    "german":               "deu_Latn",
    "spanish":              "spa_Latn",
    "japanese":             "jpn_Jpan",
    "korean":               "kor_Hang",
    "russian":              "rus_Cyrl",
    "portuguese":           "por_Latn",
    "turkish":              "tur_Latn",
    "bengali":              "ben_Beng",
    "urdu":                 "urd_Arab",
    "tamil":                "tam_Taml",
    "telugu":               "tel_Telu",
    "marathi":              "mar_Deva",
    "gujarati":             "guj_Gujr",
    "punjabi":              "pan_Guru",
}


class NLLB200Translator:
    """
    Local inference translation engine using Facebook NLLB-200.

    All processing happens on internal servers.
    No data is sent to external services.
    """

    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: NLLB-200 model variant.
                       Options:
                       - facebook/nllb-200-distilled-600M  (fast, good quality)
                       - facebook/nllb-200-distilled-1.3B  (slower, better quality)
                       - facebook/nllb-200-1.3B            (no distillation)
                       600M distilled chosen for production batch processing speed.

            device:    'cuda', 'cpu', or None (auto-detect).
                      GPU strongly recommended for batch processing.
                      CPU inference is ~10x slower.
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Install transformers: pip install transformers sentencepiece"
            )

        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Device selection
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.logger.info(
            f"Loading NLLB-200 ({model_name}) on {self.device}..."
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.logger.info("NLLB-200 loaded successfully.")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
    ) -> str:
        """
        Translate a single text block.

        Args:
            text:        Text to translate
            source_lang: NLLB-200 language code for source (e.g. "eng_Latn")
            target_lang: NLLB-200 language code for target (e.g. "hin_Deva")
            max_length:  Maximum output token length
            num_beams:   Beam search width.
                        4 beams gives good quality/speed tradeoff.
                        Greedy (num_beams=1) is faster but lower quality.

        Returns:
            Translated text string

        How forced_bos_token_id works:
        - NLLB-200 decoder generates one token at a time
        - The first token determines the target language script
        - By forcing the first token to be the target language code,
          all subsequent generation is constrained to that language
        - Without this, the model may default to English or mix languages
        """
        if not text or not text.strip():
            return text

        try:
            # Set source language for tokenizer
            self.tokenizer.src_lang = source_lang

            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True,
            ).to(self.device)

            # Get target language token ID for forced_bos_token_id
            target_lang_id = self.tokenizer.convert_tokens_to_ids(target_lang)

            if target_lang_id == self.tokenizer.unk_token_id:
                self.logger.warning(
                    f"Unknown language code: {target_lang}. "
                    f"Check LANGUAGE_CODES dict for valid codes."
                )
                return text

            # Generate translation
            with torch.no_grad():
                translated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                )

            # Decode
            translated_text = self.tokenizer.batch_decode(
                translated_tokens,
                skip_special_tokens=True,
            )[0]

            return translated_text

        except Exception as e:
            self.logger.error(f"Translation failed: {e}. Returning original text.")
            return text

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> List[str]:
        """
        Translate a list of text blocks in batches.

        Why batch processing:
        - GPU utilization increases with batch size
        - Sequential per-text translation wastes GPU parallelism
        - batch_size=8 is a good default for memory vs speed tradeoff

        Args:
            texts:       List of text strings to translate
            source_lang: Source language NLLB-200 code
            target_lang: Target language NLLB-200 code
            batch_size:  Number of texts per GPU batch
            max_length:  Maximum token length per translation

        Returns:
            List of translated strings in same order as input
        """
        if not texts:
            return []

        translated = []
        self.tokenizer.src_lang = source_lang
        target_lang_id = self.tokenizer.convert_tokens_to_ids(target_lang)

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # Filter empty strings, remember positions
            non_empty = [(j, t) for j, t in enumerate(batch) if t.strip()]

            if not non_empty:
                translated.extend(batch)
                continue

            indices, batch_texts = zip(*non_empty)

            try:
                inputs = self.tokenizer(
                    list(batch_texts),
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=target_lang_id,
                        max_length=max_length,
                        num_beams=4,
                    )

                batch_translated = self.tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

                # Re-assemble with original ordering
                result_batch = list(batch)
                for idx, translation in zip(indices, batch_translated):
                    result_batch[idx] = translation

                translated.extend(result_batch)

            except Exception as e:
                self.logger.error(f"Batch translation failed: {e}")
                translated.extend(batch)  # Return originals on failure

            self.logger.debug(
                f"Translated batch {i // batch_size + 1}: "
                f"{min(i + batch_size, len(texts))}/{len(texts)} texts"
            )

        return translated

    @staticmethod
    def get_language_code(language_name: str) -> Optional[str]:
        """
        Get NLLB-200 language code from human-readable name.

        Args:
            language_name: e.g. "hindi", "french", "arabic"

        Returns:
            NLLB-200 language code, or None if not found
        """
        return LANGUAGE_CODES.get(language_name.lower())
