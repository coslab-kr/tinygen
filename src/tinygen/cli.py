"""Command line entry point for the Python compilation pipeline."""
import argparse
import subprocess
import sys

from .api import compile_model, lower_model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate C/C++ from TOSA MLIR")
    parser.add_argument("model_path")
    parser.add_argument("-o", "--output-dir", default="gen")
    parser.add_argument("--tinyc", help="Path to the native tinyc executable")
    parser.add_argument("--mlir-translate", help="Path to LLVM 20.1 mlir-translate")
    parser.add_argument("--emitc-only", action="store_true", help="Only run tinyc")
    parser.add_argument("--no-optimize", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.emitc_only:
            result = lower_model(args.model_path, args.output_dir, tinyc=args.tinyc)
        else:
            result = compile_model(
                args.model_path, args.output_dir, tinyc=args.tinyc,
                mlir_translate=args.mlir_translate, optimize=not args.no_optimize,
            )
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"tinygen: {error}", file=sys.stderr)
        return 1
    print(result)
    return 0
