#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def main():
    """Main method"""
    parser = argparse.ArgumentParser(prog="check_transcriptions.py")
    parser.add_argument("wav_paths", help="WAV file paths on each line")
    parser.add_argument("expected_dir", help="Directory with expected transcriptions")
    args = parser.parse_args()

    expected_dir = Path(args.expected_dir)
    with open(args.wav_paths, "r") as wav_paths:
        for wav_path, line in zip(wav_paths, sys.stdin):
            wav_path = Path(wav_path.strip())
            actual_text = json.loads(line.strip())["text"]
            expected_text = (
                (expected_dir / (str(wav_path.stem) + ".txt")).read_text().strip()
            )

            assert (
                expected_text == actual_text
            ), f'Got "{actual_text}" but expected "${expected_text}" for {wav_path}'

            print(str(wav_path), "OK")


# ------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
