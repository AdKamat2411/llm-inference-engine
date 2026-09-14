#!/usr/bin/env python3
"""Sample a running inference process's GPU memory until N tokens are emitted."""

import argparse
import subprocess
import threading
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()
    binary = args.binary.resolve()

    process = subprocess.Popen(
        ["stdbuf", "-oL", str(binary)],
        cwd=binary.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    token_ids = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            if "Generated token ID: " in line:
                token_ids.append(int(line.rsplit("Generated token ID: ", 1)[1]))
                if len(token_ids) == args.tokens:
                    process.terminate()
                    return

    reader = threading.Thread(target=read_output)
    reader.start()
    peak_mib = 0
    while process.poll() is None:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        for row in result.stdout.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) == process.pid:
                peak_mib = max(peak_mib, int(fields[1]))
        time.sleep(0.05)

    reader.join()
    _, stderr = process.communicate()
    if len(token_ids) != args.tokens or peak_mib == 0:
        raise RuntimeError(
            f"Got {len(token_ids)} tokens and {peak_mib} MiB peak; stderr: {stderr}"
        )
    print(f"tokens={len(token_ids)} peak_process_gpu_memory_mib={peak_mib}")


if __name__ == "__main__":
    main()
