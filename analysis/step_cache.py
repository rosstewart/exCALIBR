"""
On-disk cache for the slow, pure computation steps of
analysis/analyze_pipeline_output.py Part A (ClinGen confusion, robustness,
bootstrap-count and restart sweeps, SpliceAI ablation, skew-locked val_ll).

A step's result is reused only when nothing it depends on changed. The key
combines:
  - the step name,
  - fingerprints of its input files: size + mtime of each path, or of every
    file matching (directory, glob),
  - a hash of the source of the code that computes it (module objects or
    file paths), so editing that code invalidates the cache,
  - a free-form key for parameters (e.g. the dataset list).

Results are pickled under config.STEP_CACHE_DIR. Set EXCALIBR_NO_STEP_CACHE=1
to ignore (and overwrite) every cache entry for a run.

    from analysis.step_cache import cached
    value, hit = cached("clingen_strip1", lambda: build(...),
                        inputs=[DATASET_TSV, (OUTPUT_DIR, "*/*_variants.csv")],
                        code=[analysis.clingen], key=sorted(datasets))
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Tuple

from analysis import config as cfg


def _file_fp(path: Path) -> list:
    st = path.stat()
    return [str(path), st.st_size, st.st_mtime_ns]


def fingerprint(inputs: Iterable) -> str:
    """Hash of (path, size, mtime) for each input: a path, or (dir, glob)."""
    h = hashlib.sha256()
    for item in inputs:
        if isinstance(item, tuple):
            root, pattern = Path(item[0]), item[1]
            files = sorted(p for p in root.glob(pattern) if p.is_file()) if root.exists() else []
            h.update(json.dumps([str(root), pattern, [_file_fp(p) for p in files]]).encode())
        else:
            p = Path(item)
            h.update(json.dumps(_file_fp(p) if p.exists() else [str(p), "missing"]).encode())
    return h.hexdigest()


def _code_digest(code) -> bytes:
    """Bytecode, names and constants of a code object (recursively), without
    line numbers, so edits elsewhere in the notebook don't change it."""
    h = hashlib.sha256(code.co_code)
    h.update(repr((code.co_names, code.co_varnames, code.co_freevars)).encode())
    for c in code.co_consts:
        h.update(_code_digest(c) if hasattr(c, "co_code") else repr(c).encode())
    return h.digest()


def _code_hash(code: Iterable) -> str:
    h = hashlib.sha256()
    for obj in code:
        if isinstance(obj, (str, Path)):
            h.update(Path(obj).read_bytes())
        else:
            try:
                h.update(inspect.getsource(obj).encode())
            except (OSError, TypeError):  # e.g. a function defined in a notebook cell
                h.update(_code_digest(obj.__code__))
    return h.hexdigest()


def cached(name: str, compute: Callable[[], Any], inputs: Iterable = (), code: Iterable = (),
           key: Any = None) -> Tuple[Any, bool]:
    """(value, hit): the cached value for this step when its inputs, code and
    key are unchanged, else compute() (and store it)."""
    t0 = time.time()
    digest = hashlib.sha256(json.dumps(
        [name, fingerprint(inputs), _code_hash(code), repr(key)]).encode()).hexdigest()[:20]
    cache_dir = Path(cfg.STEP_CACHE_DIR)
    path = cache_dir / f"{name}-{digest}.pkl"
    if path.exists() and os.environ.get("EXCALIBR_NO_STEP_CACHE", "0") != "1":
        with open(path, "rb") as f:
            value = pickle.load(f)
        print(f"  [step cache] {name}: reused {path.name} ({time.time() - t0:.0f}s)")
        return value, True
    value = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in cache_dir.glob(f"{name}-*.pkl"):  # keep one entry per step
        old.unlink()
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    print(f"  [step cache] {name}: computed and stored ({time.time() - t0:.0f}s)")
    return value, False
