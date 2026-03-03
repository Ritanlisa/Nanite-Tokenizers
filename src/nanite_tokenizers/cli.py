from __future__ import annotations

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(prog="nanite-tokenizers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("demo", help="Run the seq2seq demo")

    train_parser = subparsers.add_parser("train", help="Train the simplier compressor")
    train_parser.add_argument("--model-index", type=int, default=0)

    subparsers.add_parser("download", help="Download tokenizer assets")

    args = parser.parse_args()

    if args.command == "demo":
        from nanite_tokenizers.inference.seq2seq_demo import run_demo

        run_demo()
    elif args.command == "train":
        from nanite_tokenizers.training.simplier import train_simplier

        train_simplier(model_index=args.model_index)
    elif args.command == "download":
        from nanite_tokenizers.tools.download_tokenizer import download_tokenizer

        download_tokenizer()


if __name__ == "__main__":
    main()
