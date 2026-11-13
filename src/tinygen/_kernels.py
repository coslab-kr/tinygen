"""Collect CMSIS sources by symbol dependencies rather than a version-specific file list."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
import shutil

_SYMBOL = re.compile(r"\b(arm_\w+)\s*\(")
_DEFINITION = re.compile(r"\b(arm_\w+)\s*\([^;{}]*\)\s*\{", re.S)
_COMMENTS = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)


def _code(path: Path) -> str:
    return _COMMENTS.sub("", path.read_text(encoding="utf-8"))


def cmsis_sources(ops: list[str], kernels: Path, cmsis: Path) -> list[Path]:
    """Conservatively include dependencies from all architecture branches."""
    providers: dict[str, set[Path]] = defaultdict(set)
    symbols = {}
    for source in sorted((cmsis / "Source").rglob("*.c")):
        code = _code(source)
        symbols[source] = set(_SYMBOL.findall(code))
        for symbol in _DEFINITION.findall(code):
            providers[symbol].add(source)
    selected: set[Path] = set()
    pending = []
    for op in ops:
        wrapper = kernels / "cmsis-nn" / f"{op}.cpp"
        if wrapper.is_file():
            for symbol in _SYMBOL.findall(_code(wrapper)):
                if symbol not in providers:
                    raise RuntimeError(f"CMSIS-NN has no implementation of {symbol} (used by {op})")
                pending.extend(providers[symbol])
    while pending:
        source = pending.pop()
        if source in selected:
            continue
        selected.add(source)
        for symbol in symbols[source]:
            pending.extend(providers.get(symbol, ()))
    return sorted(selected)


def collect_kernels(op_list: Path, destination: Path, kernels: Path, cmsis: Path) -> None:
    if not (cmsis / "Include" / "arm_nnfunctions.h").is_file():
        raise FileNotFoundError("CMSIS-NN sources missing; run git submodule update --init --recursive")
    ops = [line.split()[0] for line in op_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    for op in ops:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", op):
            raise ValueError(f"Invalid operator name: {op!r}")
    # Resolve dependencies before writing any generated source files.
    sources = cmsis_sources(ops, kernels, cmsis)
    destination.mkdir(parents=True, exist_ok=True)
    defines = "\n".join(f"#define {op.upper()}" for op in dict.fromkeys(ops))
    (destination / "kernels.h").write_text(
        f"#ifndef TINYGEN_KERNELS_H\n#define TINYGEN_KERNELS_H\n\n{defines}\n\n#endif\n",
        encoding="utf-8",
    )
    for name in ("tensor.h", "model.h"):
        shutil.copy2(kernels / name, destination / name)
    for folder in ("cmsis-nn", "reference"):
        for source in sorted((kernels / folder).iterdir()):
            if source.suffix in (".cpp", ".h"):
                shutil.copy2(source, destination / source.name)
    shutil.copytree(cmsis / "Include", destination, dirs_exist_ok=True)
    shutil.copy2(cmsis / "LICENSE", destination / "CMSIS-NN-LICENSE.txt")
    names = set()
    for source in sources:
        if source.name in names:
            raise RuntimeError(f"CMSIS-NN source filename collision: {source.name}")
        names.add(source.name)
        shutil.copy2(source, destination / source.name)
