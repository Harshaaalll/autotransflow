"""
AutoTransFlow — Command Line Interface

Run the document translation pipeline from the terminal.

Usage examples:
    # Translate English PDF to Hindi
    python main.py --input contract.pdf --output contract_hindi.pdf \
                   --source eng_Latn --target hin_Deva

    # Translate to Arabic
    python main.py --input agreement.pdf --output agreement_ar.pdf \
                   --source eng_Latn --target arb_Arab

    # List all supported languages
    python main.py --list-languages

    # Use GPU for faster translation
    python main.py --input doc.pdf --output doc_fr.pdf \
                   --source eng_Latn --target fra_Latn --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

from src.pipeline import AutoTransFlow
from src.translation.nllb_translator import LANGUAGE_CODES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def list_languages():
    """Print all supported language codes."""
    print("\nSupported languages:")
    print(f"{'Language':<25} {'NLLB-200 Code'}")
    print("─" * 40)
    for name, code in sorted(LANGUAGE_CODES.items()):
        print(f"{name:<25} {code}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AutoTransFlow: Layout-preserving multilingual PDF translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input contract.pdf --output contract_hi.pdf \\
                 --source eng_Latn --target hin_Deva

  python main.py --list-languages
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="Path to input PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path for translated output PDF"
    )
    parser.add_argument(
        "--source", "-s",
        default="eng_Latn",
        help="Source language NLLB-200 code (default: eng_Latn)"
    )
    parser.add_argument(
        "--target", "-t",
        help="Target language NLLB-200 code (e.g. hin_Deva, fra_Latn)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Inference device (default: auto-detect)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Layout detection confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List all supported language codes and exit"
    )

    args = parser.parse_args()

    if args.list_languages:
        list_languages()
        sys.exit(0)

    # Validate required arguments
    if not args.input:
        parser.error("--input is required")
    if not args.target:
        parser.error("--target language code is required")
    if not args.output:
        input_path = Path(args.input)
        args.output = str(
            input_path.parent / f"{input_path.stem}_{args.target}{input_path.suffix}"
        )
        logger.info(f"Output path auto-set to: {args.output}")

    # Validate input file
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    logger.info(
        f"AutoTransFlow starting:\n"
        f"  Input:   {args.input}\n"
        f"  Output:  {args.output}\n"
        f"  Source:  {args.source}\n"
        f"  Target:  {args.target}\n"
        f"  Device:  {args.device or 'auto'}"
    )

    # Run pipeline
    pipeline = AutoTransFlow(
        source_lang=args.source,
        target_lang=args.target,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    output_path = pipeline.translate(
        input_pdf=args.input,
        output_pdf=args.output,
    )

    logger.info(f"Done. Translated document saved to: {output_path}")


if __name__ == "__main__":
    main()
