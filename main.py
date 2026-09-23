"""
Main Entry Point for GPT-2 Mini Model Pipeline
Runs: Tokenization -> Embeddings -> Transformer -> Text Generation
"""

import sys
from pathlib import Path

# Ensure UTF-8 output encoding across platforms
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.main import run_pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run complete GPT-2 Mini Model Pipeline")
    parser.add_argument(
        "--text",
        type=str,
        default="I love machine learning",
        help="Input prompt for GPT-2 generation"
    )
    args = parser.parse_args()

    run_pipeline(args.text)
