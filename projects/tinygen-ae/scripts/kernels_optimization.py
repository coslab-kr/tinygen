import os
import re
from pathlib import Path
import argparse

GEN_DIR = Path("gen")
PARAM_KEYS = ["pad_y", "pad_x", "stride_y", "stride_x", "dilation_y", "dilation_x"]

OP_TO_EXISTING_FILE = {
    "avg_pool_2d": "arm_avgpool_s8.c",
    "conv2d_f3x3_s8": "arm_convolve_s8.c",
    "depthwise_conv2d_f3x3_s8": "arm_depthwise_conv_3x3_s8.c",
    "max_pool_2d": "arm_max_pool_s8.c",
}

def read_lines(p: Path):
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]

def patch_consts_inplace(filepath: Path, params: dict) -> int:
    code = filepath.read_text(encoding="utf-8")
    total = 0

    for key, val in params.items():
        pattern = rf"(^\s*const\s+[\w\d_]+\s+{re.escape(key)}\s*=\s*)([^;]+)(;)"
        repl = r"\g<1>" + str(val) + r"\g<3>"

        code, n = re.subn(pattern, repl, code, flags=re.MULTILINE)
        total += n

    if total:
        filepath.write_text(code, encoding="utf-8")
    return total

def build_params(nums):
    values = list(map(int, nums))
    k = min(len(values), len(PARAM_KEYS))
    params = {}
    for i in range(k):
        if values[i] != -1:
            params[PARAM_KEYS[i]] = values[i]
    return params

def process_one_tinygen_dir(tinygen_dir: Path) -> None:
    op_file = tinygen_dir / "model_ops_list.txt"

    if not op_file.exists():
        return

    lines = read_lines(op_file)

    for line in lines:
        parts = line.split()
        op = parts[0]
        nums = parts[1:]

        if op not in OP_TO_EXISTING_FILE:
            continue
        if not nums:
            continue

        params = build_params(nums)
        if not params:
            continue

        target = tinygen_dir / OP_TO_EXISTING_FILE[op]
        if not target.exists():
            raise FileNotFoundError(f"[{tinygen_dir.name}] target missing for {op}: {target.name}")

        n = patch_consts_inplace(target, params)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", help="Specific tinygen directory to optimize", required=True)
    args = parser.parse_args()

    if not GEN_DIR.exists():
        print(f"[ERROR] Missing gen dir: {GEN_DIR}")
        return

    tinygen_dirs = [Path(args.dir)]

    for d in tinygen_dirs:
        if not d.exists():
            continue
        process_one_tinygen_dir(d)

if __name__ == "__main__":
    main()
