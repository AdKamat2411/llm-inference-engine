#!/usr/bin/env python3
"""Measure token timing from the existing inference binary's stdout."""

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path


def run_once(binary: Path, tokens: int) -> dict:
    command = ["timeout", "180s", "stdbuf", "-oL", str(binary)]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=binary.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    ids = []
    times = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if "Generated token ID: " not in line:
                continue
            ids.append(int(line.rsplit("Generated token ID: ", 1)[1]))
            times.append(time.perf_counter())
            if len(ids) == tokens:
                break
    finally:
        process.terminate()
        _, stderr = process.communicate()

    if len(ids) != tokens:
        raise RuntimeError(f"Expected {tokens} tokens, got {len(ids)}; stderr: {stderr}")

    digest = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()
    return {
        "first_token_s": times[0] - started,
        "decode_tokens_per_s": (tokens - 1) / (times[-1] - times[0]),
        "token_ids_sha256": digest,
        "first_8_ids": ids[:8],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if args.tokens < 2 or args.repeats < 1:
        parser.error("--tokens must be at least 2 and --repeats at least 1")

    run_once(binary, min(16, args.tokens))
    runs = [run_once(binary, args.tokens) for _ in range(args.repeats)]
    if len({run["token_ids_sha256"] for run in runs}) != 1:
        raise RuntimeError("Generated token IDs differed between repetitions")

    print(json.dumps({
        "binary": str(binary),
        "tokens": args.tokens,
        "repeats": args.repeats,
        "runs": runs,
        "median_first_token_s": statistics.median(run["first_token_s"] for run in runs),
        "median_decode_tokens_per_s": statistics.median(
            run["decode_tokens_per_s"] for run in runs
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
