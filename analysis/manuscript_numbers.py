"""
Manuscript number/table/figure registry for the Nature Genetics manuscript
(`ExCALIBR_latex_src/main.tex`, `supplement.tex`).

`analyze_pipeline_output.py` fills one `ManuscriptRegistry` (`REG`) as it
computes, then `REG.write()` emits, under `MANUSCRIPT_FIGURE_DIR/generated/`
(= the Overleaf `Figures/generated/` folder; `REG.export_figures()` copies the
included figures there from the `FIGURE_DIR` working tree):

- `macros.tex`: one `\\newcommand{\\Name}{value}` per registered number, sorted
  by the macro's first use in the .tex files, with `% file:Lnn` comments.
  Macros are plain (no trailing space / xspace), so the .tex uses `\\Name{}`.
- `tables/<label>.tex`: tabular bodies only (`\\begin{tabular}...\\end{tabular}`);
  captions/labels/table environments stay hand-written in the .tex.
- `manifest.json`: provenance for every number/table/figure/input.
- `build_report.md`: old -> new values, staleness, table-template diffs,
  unregistered macros, failed prose claims, Nature Genetics checks.

The manuscript is the source of truth for all non-quantitative text: table
bodies are compared to reference snapshots of the manuscript's own tables
(`analysis/manuscript_table_templates/`) with numbers masked, and any
wording/structure difference is reported as a failure.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

MANAGED_PREFIXES = ("Num", "Data", "Meth", "Res", "ED", "Supp")
MANAGED_MACRO_RE = re.compile(r"\\((?:%s)[A-Za-z]+)" % "|".join(MANAGED_PREFIXES))
TEMPLATE_DIR = Path(__file__).resolve().parent / "manuscript_table_templates"
# Literal values the .tex carried before each was replaced by a macro.
OLD_VALUES_JSON = Path(__file__).resolve().parent / "manuscript_curated" / "manuscript_old_values.json"

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


class ManuscriptBuildError(RuntimeError):
    pass


def number_to_words(n: int) -> str:
    if not 0 <= n < 100:
        return f"{n:,}"
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _git_info(repo: Path) -> Dict[str, Any]:
    try:
        # cwd= rather than `git -C` (not supported by this machine's git).
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo),
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(repo),
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}


def _file_info(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    info = {"path": str(path), "exists": True, "mtime": st.st_mtime,
            "mtime_iso": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
    if path.is_file():
        info["size"] = st.st_size
        if st.st_size <= 200 * 1024 * 1024:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            info["sha256"] = h.hexdigest()
    return info


def newest_mtime(path: Path, pattern: str = "*") -> float:
    """Newest mtime of a file, or of any file matching `pattern` under a directory."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_mtime
    mtimes = [p.stat().st_mtime for p in path.rglob(pattern) if p.is_file()]
    return max(mtimes) if mtimes else 0.0


def extract_tabular(latex: str) -> str:
    """The `\\begin{tabular}...\\end{tabular}` block of a full table string."""
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", latex, flags=re.S)
    if not m:
        raise ValueError("no tabular environment found")
    return m.group(0)


_NUM_TOKEN_RE = re.compile(r"(?<![A-Za-z\\])[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def mask_numbers(text: str) -> str:
    return _NUM_TOKEN_RE.sub("#", text)


def _row_shape(row: str) -> str:
    row = re.sub(r"\\textit\{[^}]*\}", r"\\textit{*}", row)
    row = re.sub(r"\\texttt\{[^}]*\}", r"\\texttt{*}", row)
    return re.sub(r"\s+", " ", mask_numbers(row)).strip()


def _split_rows(tabular: str) -> List[str]:
    body = re.sub(r"^\\begin\{tabular\}\{.*?\}\s*|\\end\{tabular\}\s*$", "", tabular.strip(), flags=re.S)
    rows = []
    for raw in body.split("\n"):
        raw = raw.strip()
        if raw:
            rows.append(raw)
    return rows


def compare_to_template(generated: str, template: str, data_row_label_re: Optional[str] = None) -> List[str]:
    """Differences between a generated tabular body and the manuscript's own
    table, ignoring numbers. Fixed rows (headers, rules, metric-name rows) must
    match exactly; data rows whose label matches `data_row_label_re` (e.g. gene
    rows) may differ in count/content as long as each one's masked shape is a
    shape the template already uses."""
    issues = []
    gen_spec = re.match(r"\\begin\{tabular\}\{(.*?)\}\s", generated.strip() + "\n", flags=re.S)
    tpl_spec = re.match(r"\\begin\{tabular\}\{(.*?)\}\s", template.strip() + "\n", flags=re.S)
    if gen_spec and tpl_spec and gen_spec.group(1) != tpl_spec.group(1):
        issues.append(f"column spec differs: generated {{{gen_spec.group(1)}}} vs manuscript {{{tpl_spec.group(1)}}}")

    def is_data(row: str) -> bool:
        return bool(data_row_label_re) and re.match(data_row_label_re, row) is not None

    def fixed_rows(rows: List[str]) -> List[str]:
        # \midrule separators between data-row blocks scale with the data
        # (e.g. one per gene), so only rules outside the data span are fixed.
        data_idx = [i for i, r in enumerate(rows) if is_data(r)]
        lo, hi = (data_idx[0], data_idx[-1]) if data_idx else (len(rows), -1)
        return [_row_shape(r) for i, r in enumerate(rows)
                if not is_data(r) and not (lo < i < hi and r.startswith("\\midrule"))]

    gen_rows, tpl_rows = _split_rows(generated), _split_rows(template)
    gen_fixed = fixed_rows(gen_rows)
    tpl_fixed = fixed_rows(tpl_rows)
    if gen_fixed != tpl_fixed:
        import difflib
        diff = list(difflib.unified_diff(tpl_fixed, gen_fixed, "manuscript", "generated", lineterm="", n=0))
        issues.append("fixed rows differ (numbers masked):\n" + "\n".join(diff))
    tpl_shapes = {_row_shape(r) for r in tpl_rows if is_data(r)}
    bad = sorted({_row_shape(r) for r in gen_rows if is_data(r)} - tpl_shapes)
    if bad:
        issues.append("data-row shapes not used by the manuscript table:\n" + "\n".join(bad[:10]))
    return issues


@dataclass
class _Entry:
    name: str
    rendered: str
    value: Any
    kind: str
    source: str
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    part: str = "A"


class ManuscriptRegistry:
    def __init__(self, tex_paths: Sequence[Path], out_dir: Path, repo_root: Path,
                 strict: Optional[bool] = None, waivers: Optional[Dict[str, str]] = None):
        self.tex_paths = [Path(p) for p in tex_paths]
        self.out_dir = Path(out_dir)
        self.tables_dir = self.out_dir / "tables"
        self.repo_root = Path(repo_root)
        self.strict = (os.environ.get("EXCALIBR_STRICT", "0") == "1") if strict is None else strict
        self.waivers = dict(waivers or {})
        self.started = _dt.datetime.now().timestamp()
        self.macros: Dict[str, _Entry] = {}
        self.tables: Dict[str, Dict[str, Any]] = {}
        self.figures: Dict[str, Dict[str, Any]] = {}
        self.statics: Dict[str, Dict[str, Any]] = {}
        self.requirements: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []
        self.outputs: List[Dict[str, Any]] = []
        self.notes: List[str] = []
        self.carried_from: Optional[str] = None  # set by carry_over()

    # -- partial rebuilds --------------------------------------------------------
    def carry_over(self, manifest_path: Path):
        """Seed this registry from a previous build's manifest.json, for a
        partial rebuild (scripts/rebuild_sections.py) that reruns only some
        sections: every entry those sections register replaces its carried
        copy; the rest keep the previous build's values and are reported as
        carried over. Word forms (<name>Word/WordCap) are re-rendered from
        their base value."""
        m = json.loads(Path(manifest_path).read_text())
        stamp = m.get("generated", "?")
        self.carried_from = f"{manifest_path} ({stamp})"
        for name, e in m.get("macros", {}).items():
            self.macros[name] = _Entry(name, e["rendered"], e["value"], e["kind"], e["source"],
                                       e.get("inputs", []), "carried")
        for name, e in list(self.macros.items()):
            for suffix in ("WordCap", "Word"):
                base = self.macros.get(name[:-len(suffix)]) if name.endswith(suffix) else None
                if base is not None and isinstance(base.value, (int, float)) and not isinstance(base.value, bool):
                    # Only fix words rendered with the old truncation
                    # (int(value)); leave custom text (e.g. 0 -> "none") alone.
                    old, new = number_to_words(int(base.value)), number_to_words(int(round(base.value)))
                    cap = lambda w: w[:1].upper() + w[1:]
                    if e.rendered in (old, cap(old)):
                        e.rendered = cap(new) if suffix == "WordCap" else new
                    break
        for key in ("tables", "figures", "statics"):
            for label, rec in m.get(key, {}).items():
                getattr(self, key)[label] = {**rec, "carried": stamp}
        for rec in self.tables.values():  # templates may have been refreshed since
            body, tpl = Path(rec["path"]), Path(rec.get("template", ""))
            # (older manifests lack data_row_label_re: re-check only those with issues)
            if body.exists() and tpl.is_file() and ("data_row_label_re" in rec or rec.get("issues")):
                rec["issues"] = compare_to_template(body.read_text(), tpl.read_text(),
                                                    rec.get("data_row_label_re"))
        self.requirements = [{**r, "carried": stamp} for r in m.get("requirements", [])]
        self.checks = [{**c, "carried": stamp} for c in m.get("checks", [])]
        print(f"  Carried over {len(self.macros)} macros, {len(self.tables)} tables, {len(self.figures)} figures "
              f"from {self.carried_from}")

    # -- numbers ---------------------------------------------------------------
    def _add(self, name: str, rendered: str, value: Any, kind: str, source: str,
             inputs: Iterable = (), part: str = "A"):
        if not MANAGED_MACRO_RE.fullmatch("\\" + name):
            raise ValueError(f"macro name {name!r} must start with one of {MANAGED_PREFIXES}")
        self.macros[name] = _Entry(name, rendered, value, kind, source,
                                   [_file_info(Path(p)) for p in inputs], part)

    def num(self, name: str, value, fmt: str = ",d", kind: str = "data", source: str = "",
            inputs: Iterable = (), words: bool = False):
        """Register a number. `fmt` is a Python format spec (",d", ".1f", ".3f").
        `words=True` also registers `<name>Word` ("fourteen") and
        `<name>WordCap` ("Fourteen") for small integers."""
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            self._fail("missing-value", f"{name}: value is {value!r} ({source})")
            rendered = "??"
        else:
            if fmt.endswith("d"):
                value = int(round(value))
            rendered = format(value, fmt)
        self._add(name, rendered, value, kind, source, inputs)
        if words and rendered != "??":
            w = number_to_words(int(round(value)))  # match the rendered (rounded) number
            self._add(name + "Word", w, value, kind, source, inputs)
            self._add(name + "WordCap", w[:1].upper() + w[1:], value, kind, source, inputs)
        return value

    def pct(self, name: str, numer, denom, digits: int = 1, kind: str = "data", source: str = "",
            inputs: Iterable = (), words: bool = False):
        """Register 100*numer/denom (rendered without the % sign; the .tex keeps `\\%`)."""
        value = (100.0 * numer / denom) if denom else None
        self.num(name, value, fmt=f".{digits}f", kind=kind,
                 source=f"{source} [{numer}/{denom}]", inputs=inputs)
        if words and value is not None:
            w = number_to_words(int(round(value)))
            self._add(name + "Word", w, value, kind, source, inputs)
            self._add(name + "WordCap", w[:1].upper() + w[1:], value, kind, source, inputs)
        return value

    def text(self, name: str, latex: str, kind: str = "data", source: str = "", inputs: Iterable = ()):
        self._add(name, latex, latex, kind, source, inputs)

    def check(self, claim_id: str, ok: bool, message: str, location: str = ""):
        """Record whether a quantitative claim in the prose still holds. Never
        raises: rewording is the author's call, so failures go to the report."""
        self.checks = [c for c in self.checks if not (c.get("carried") and c["claim"] == claim_id)]
        self.checks.append({"claim": claim_id, "ok": bool(ok), "message": message, "location": location})
        if not ok:
            print(f"  CLAIM CHECK FAILED [{claim_id}] {location}: {message}")

    # -- tables / figures --------------------------------------------------------
    def table(self, label: str, latex: str, inputs: Iterable = (), data_row_label_re: Optional[str] = None,
              part: str = "A"):
        """Write the tabular body of `latex` (full table or tabular) to
        `tables/<label_without_tab:>.tex` and compare it to the manuscript template."""
        body = extract_tabular(latex)
        stem = label.split(":", 1)[-1]
        path = self.tables_dir / f"{stem}.tex"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n")
        tpl_path = TEMPLATE_DIR / f"{stem}.tex"
        issues = (compare_to_template(body, tpl_path.read_text(), data_row_label_re)
                  if tpl_path.exists() else [f"no manuscript template at {tpl_path}"])
        self.tables[label] = {"path": str(path), "template": str(tpl_path), "issues": issues, "part": part,
                              "data_row_label_re": data_row_label_re,
                              "inputs": [_file_info(Path(p)) for p in inputs]}
        print(f"  Table {label} -> {path}" + (f"  [{len(issues)} template issue(s)]" if issues else ""))
        for issue in issues:
            print("    " + issue.replace("\n", "\n    "))
        return path

    def figure(self, label: str, path: Path, inputs: Iterable = (), part: str = "A"):
        path = Path(path)
        info = _file_info(path)
        fresh = info.get("exists") and info["mtime"] >= self.started - 1
        self.figures[label] = {**info, "fresh_this_run": bool(fresh), "part": part,
                               "inputs": [_file_info(Path(p)) for p in inputs]}
        if not fresh:
            self._fail("figure-not-written", f"{label}: {path} was not (re)written this run")

    def static(self, path: str, note: str = ""):
        self.statics[path] = {"path": path, "note": note}

    def output(self, path: Path, part: str, what: str = ""):
        """Record a non-manuscript output (Parts B-D) so the preservation check can see it."""
        self.outputs.append({**_file_info(Path(path)), "part": part, "what": what})

    # -- inputs ----------------------------------------------------------------
    def require(self, path, what: str, newer_than: Sequence = (), expect_datasets: Optional[set] = None,
                found_datasets: Optional[set] = None, subset_ok: bool = False, regen: str = "",
                own_pattern: str = "*") -> bool:
        """Check an upstream input: exists, not older than `newer_than` paths,
        and covers exactly `expect_datasets` (or a subset if `subset_ok`).
        Returns True when usable. Problems raise in strict mode unless waived."""
        path = Path(path) if path else None
        rec = {"what": what, "path": str(path), "problems": [], "regen": regen}
        if path is None or not path.exists():
            rec["problems"].append("missing")
        else:
            mtime = newest_mtime(path, own_pattern)
            for dep in newer_than:
                # dep: a path, or (directory, glob) to compare against only matching files
                # (e.g. (OUTPUT_DIR, "*_calibration.json"), never the figures under it).
                dep_path, dep_pat = (Path(dep[0]), dep[1]) if isinstance(dep, tuple) else (Path(dep), "*")
                if dep_path.exists() and newest_mtime(dep_path, dep_pat) > mtime + 1:
                    rec["problems"].append(
                        f"older than {dep_path}/{dep_pat if dep_path.is_dir() else ''} "
                        f"({_dt.datetime.fromtimestamp(mtime):%Y-%m-%d %H:%M} < "
                        f"{_dt.datetime.fromtimestamp(newest_mtime(dep_path, dep_pat)):%Y-%m-%d %H:%M})")
        if expect_datasets is not None and found_datasets is not None:
            missing = sorted(set(expect_datasets) - set(found_datasets))
            extra = sorted(set(found_datasets) - set(expect_datasets))
            if (missing and not subset_ok) or extra:
                rec["problems"].append(
                    f"dataset set differs: {len(found_datasets)} found vs {len(expect_datasets)} expected; "
                    f"missing={missing[:12]}{'...' if len(missing) > 12 else ''} "
                    f"extra={extra[:12]}{'...' if len(extra) > 12 else ''}")
        rec["waived"] = self.waivers.get(str(path))
        self.requirements = [r for r in self.requirements if not (r.get("carried") and r["what"] == what)]
        self.requirements.append(rec)
        if rec["problems"]:
            msg = f"{what} ({path}): " + "; ".join(rec["problems"]) + (f"\n    regenerate: {regen}" if regen else "")
            if rec["waived"]:
                print(f"  WAIVED {msg}\n    reason: {rec['waived']}")
            else:
                self._fail("stale-input", msg)
        return path is not None and path.exists()

    def _fail(self, kind: str, msg: str):
        self.notes.append(f"[{kind}] {msg}")
        if self.strict:
            raise ManuscriptBuildError(msg)
        print(f"  WARNING [{kind}] {msg}")

    # -- tex scanning ------------------------------------------------------------
    def _scan_tex(self):
        uses: Dict[str, tuple] = {}
        literals: Dict[str, List[str]] = {}
        includes: List[tuple] = []
        for order, tex in enumerate(self.tex_paths):
            if not tex.exists():
                continue
            for lineno, line in enumerate(tex.read_text().splitlines(), 1):
                code = re.sub(r"(?<!\\)%.*$", "", line)
                if code.lstrip().startswith("\\newcommand"):
                    continue
                for m in MANAGED_MACRO_RE.finditer(code):
                    uses.setdefault(m.group(1), (order, lineno, m.start(), tex.name, line.strip()[:90]))
                for m in re.finditer(r"\\(?:includegraphics(?:\[[^\]]*\])?|input)\{(Figures/[^}]*)\}", code):
                    includes.append((tex.name, lineno, m.group(1)))
        return uses, includes

    # -- export --------------------------------------------------------------------
    def export_figures(self, src_root: Path, dest_root: Path) -> List[str]:
        """Copy every `Figures/...` file the .tex includes from the working tree
        `src_root` to the Overleaf folder `dest_root`, keeping relative paths.
        `generated/` (written straight into dest_root) and static hand-made
        assets are skipped; copies whose source is unchanged are skipped too.
        Returns the relative paths copied."""
        import shutil
        _, includes = self._scan_tex()
        copied = []
        for tex, lineno, rel in includes:
            rel_path = rel[len("Figures/"):]
            if rel in self.statics or rel_path.startswith("generated/"):
                continue
            src = Path(src_root) / rel_path
            if not src.exists():
                continue  # reported as a missing include by write()
            dest = Path(dest_root) / rel_path
            if dest.exists() and dest.stat().st_size == src.stat().st_size and dest.stat().st_mtime >= src.stat().st_mtime:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel_path)
        print(f"  Exported {len(copied)} manuscript figure(s) {src_root} -> {dest_root}")
        return copied

    # -- write -------------------------------------------------------------------
    def write(self, figure_root: Path, nature_checks: Optional[Dict[str, Any]] = None,
              old_values: Optional[Dict[str, str]] = None, highlight_changed: bool = True,
              placeholder_unregistered: bool = False, generator: str = "analysis/analyze_pipeline_output.py"):
        """`placeholder_unregistered`: also define every managed macro the .tex
        uses but this run never registered, as a red XX, so a not-yet-computed
        number stays visible (and the document still compiles) rather than
        printing nothing. Off by default -- the univariate paper treats an
        unregistered macro as a wiring error, reported below."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        uses, includes = self._scan_tex()

        # macros.tex, in order of first use; unused-but-registered at the end
        def sort_key(name):
            return uses.get(name, (99, 0, 0, "", ""))[:3]

        lines = [f"% Generated by {generator} -- do not edit by hand.",
                 f"% {_dt.datetime.now():%Y-%m-%d %H:%M}  git {_git_info(self.repo_root)}",
                 "% Use as \\Name{} in the .tex.", ""]
        old_values = old_values or {}
        for name in sorted(self.macros, key=sort_key):
            e = self.macros[name]
            where = uses.get(name)
            loc = f"{where[3]}:L{where[1]}" if where else "UNUSED"
            value = e.rendered
            changed = old_values.get(name, "").strip() != e.rendered
            if highlight_changed and changed:
                # Review aid: numbers that differ from the pre-automation .tex
                # (or are new) render red; flattening strips the colour.
                value = f"\\textcolor{{red}}{{{e.rendered}}}"
            status = "new" if name not in old_values else ("changed" if changed else "same")
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {loc} | {status} | {e.kind} | {e.source}")
        unregistered = sorted(set(uses) - set(self.macros))
        if placeholder_unregistered and unregistered:
            lines += ["", "% Used in the .tex but not computed by this run -- placeholders."]
            for name in sorted(unregistered, key=sort_key):
                lines.append(f"\\newcommand{{\\{name}}}{{\\textcolor{{red}}{{XX}}}}  "
                             f"% {uses[name][3]}:L{uses[name][1]} | NOT COMPUTED")
        (self.out_dir / "macros.tex").write_text("\n".join(lines) + "\n")

        for name in unregistered:
            self._fail_soft("unregistered-macro", f"\\{name} used at {uses[name][3]}:L{uses[name][1]} "
                                                 f"but never registered")
        # A carried table the .tex no longer \input's was dropped from the
        # manuscript (e.g. a table folded into the text): don't report it.
        _included = {Path(rel).stem for _, _, rel in includes}
        self.tables = {label: rec for label, rec in self.tables.items()
                       if not rec.get("carried") or Path(rec["path"]).stem in _included}
        missing_includes = []
        for tex, lineno, rel in includes:
            if rel in self.statics:  # hand-made asset maintained on Overleaf
                continue
            target = Path(figure_root) / rel[len("Figures/"):]
            candidates = [target] + [target.with_suffix(s) for s in (".tex", ".png", ".pdf")]
            if not any(c.exists() for c in candidates):
                missing_includes.append(f"{tex}:L{lineno} {rel}")
        for item in missing_includes:
            self._fail_soft("missing-include", item)

        manifest = {
            "generated": _dt.datetime.now().isoformat(timespec="seconds"),
            "git": _git_info(self.repo_root),
            "strict": self.strict,
            "macros": {n: {"rendered": e.rendered, "value": e.value if not isinstance(e.value, float) else round(e.value, 10),
                           "kind": e.kind, "source": e.source, "first_use": uses.get(n, [None] * 5)[3:5],
                           "inputs": e.inputs} for n, e in self.macros.items()},
            "tables": self.tables, "figures": self.figures, "statics": self.statics,
            "requirements": self.requirements, "checks": self.checks, "outputs": self.outputs,
            "notes": self.notes, "unregistered_macros": unregistered, "missing_includes": missing_includes,
            "carried_over_from": self.carried_from,
            "carried_macros": sorted(n for n, e in self.macros.items() if e.part == "carried"),
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        self._write_report(uses, unregistered, missing_includes, nature_checks or {}, old_values or {})
        print(f"  Wrote {self.out_dir / 'macros.tex'} ({len(self.macros)} macros), manifest.json, build_report.md")
        problems = len(unregistered) + len(missing_includes) + sum(bool(t["issues"]) for t in self.tables.values())
        if self.strict and problems:
            raise ManuscriptBuildError(f"{problems} manuscript wiring problem(s); see {self.out_dir / 'build_report.md'}")
        return manifest

    def _fail_soft(self, kind: str, msg: str):
        self.notes.append(f"[{kind}] {msg}")
        print(f"  WARNING [{kind}] {msg}")

    def _write_report(self, uses, unregistered, missing_includes, nature_checks, old_values):
        r = [f"# Manuscript build report ({_dt.datetime.now():%Y-%m-%d %H:%M})", "",
             f"git: {_git_info(self.repo_root)}; strict={self.strict}", ""]
        if self.carried_from:
            carried = sorted(n for n, e in self.macros.items() if e.part == "carried")
            r += [f"**Partial rebuild.** {len(carried)} of {len(self.macros)} macros, and the tables, figures, "
                  f"checks and inputs of sections not rerun, are carried over from {self.carried_from} "
                  "(marked `carried` below).", ""]
        r += ["## Upstream inputs", "", "| input | path | status | regenerate |", "|---|---|---|---|"]
        for q in self.requirements:
            status = "OK" if not q["problems"] else ("WAIVED: " + q["waived"] if q["waived"] else "; ".join(q["problems"]))
            r.append(f"| {q['what']}{' (carried)' if q.get('carried') else ''} | `{q['path']}` | {status} | {q['regen'] or ''} |")
        r += ["", "## Tables (generated body vs. manuscript, numbers masked)", ""]
        for label, t in self.tables.items():
            r.append(f"- `{label}` -> `{t['path']}`: " + ("matches manuscript" if not t["issues"] else "**DIFFERS**"))
            for issue in t["issues"]:
                r += ["  ```", "  " + issue.replace("\n", "\n  "), "  ```"]
        r += ["", "## Figures", ""]
        for label, f in self.figures.items():
            r.append(f"- `{label}` `{f['path']}`: " + ("written this run" if f["fresh_this_run"] else "**NOT written this run**"))
        for p, s in self.statics.items():
            r.append(f"- static `{p}`: {s['note']}")
        r += ["", "## Macros (old literal in .tex -> new value)", "", "| macro | first use | old | new | source |",
              "|---|---|---|---|---|"]
        for name in sorted(self.macros, key=lambda n: uses.get(n, (99, 0, 0))[:3]):
            e = self.macros[name]
            where = uses.get(name)
            loc = f"{where[3]}:L{where[1]}" if where else "unused"
            old = old_values.get(name, "")
            flag = " **changed**" if old and old.strip() != e.rendered else ""
            r.append(f"| `\\{name}` | {loc} | {old} | {e.rendered}{flag} | {e.source} |")
        if unregistered:
            r += ["", "## Unregistered macros used in the .tex", ""] + [f"- `\\{n}`" for n in unregistered]
        if missing_includes:
            r += ["", "## Missing \\includegraphics / \\input targets", ""] + [f"- {m}" for m in missing_includes]
        r += ["", "## Prose claims checked against current data (rewording needs author approval)", ""]
        for c in self.checks:
            r.append(f"- [{'OK' if c['ok'] else 'FAILS'}] `{c['claim']}` {c['location']}: {c['message']}")
        if nature_checks:
            r += ["", "## Nature Genetics checks", "", "| rule | value | status |", "|---|---|---|"]
            for rule, (value, ok) in nature_checks.items():
                r.append(f"| {rule} | {value} | {'OK' if ok else '**CHECK**'} |")
        if self.outputs:
            r += ["", "## Non-manuscript outputs (Parts B-D)", ""]
            for o in self.outputs:
                r.append(f"- Part {o['part']}: `{o['path']}` {'' if o.get('exists') else '**MISSING**'} {o['what']}")
        if self.notes:
            r += ["", "## Warnings", ""] + [f"- {n}" for n in self.notes]
        (self.out_dir / "build_report.md").write_text("\n".join(r) + "\n")


def _strip_tex(s: str) -> str:
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    s = re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " EQ ", s, flags=re.S)
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\(ref\*?|label)\{[^}]*\}", "X", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{\})?", " ", s)
    return re.sub(r"[{}$~]", " ", s)


def _words(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][\w.\-%']*", _strip_tex(s)))


def nature_genetics_checks(main_tex: Path, supp_tex: Path) -> Dict[str, tuple]:
    """Article limits from nature.com/ng (content types; AIP formatting)."""
    t = Path(main_tex).read_text()
    s = Path(supp_tex).read_text() if Path(supp_tex).exists() else ""

    def section(a, b):
        i = t.find(a)
        j = t.find(b, i + 1)
        return t[i:j] if i >= 0 and j > i else ""

    _nc = lambda x: re.sub(r"(?<!\\)%.*", "", x)
    _labels_t = set(re.findall(r"\\label\{([^}]+)\}", _nc(t)))
    _labels_s = set(re.findall(r"\\label\{([^}]+)\}", _nc(s)))
    bad_xrefs = sorted(
        {f"main:{l}" for l in re.findall(r"\\ref\{([^}]+)\}", _nc(t)) if l in _labels_s and l not in _labels_t}
        | {f"supplement:{l}" for l in re.findall(r"\\ref\{([^}]+)\}", _nc(s)) if l in _labels_t and l not in _labels_s})
    ed_start = t.find(r"\section*{Extended Data}")
    main_part = t[:ed_start] if ed_start > 0 else t
    ed_part = re.sub(r"(?m)^\s*%.*$", "", t[ed_start:]) if ed_start > 0 else ""
    n_fig = len(re.findall(r"\\begin\{figure\}", re.sub(r"(?m)^\s*%.*$", "", main_part)))
    n_tab = len(re.findall(r"\\begin\{table\}", re.sub(r"(?m)^\s*%.*$", "", main_part)))
    n_ed = len(re.findall(r"\\begin\{(figure|table)\}", ed_part))
    cites = {k.strip() for c in re.findall(r"\\cite\{([^}]*)\}", re.sub(r"(?m)(?<!\\)%.*$", "", t))
             for k in c.split(",")}
    labels_main = re.findall(r"\\label\{([^}]*)\}", re.sub(r"(?m)^\s*%.*$", "", t))
    dup = sorted({l for l in labels_main if labels_main.count(l) > 1})
    ed_labels = re.findall(r"\\label\{((?:fig|tab):[^}]*)\}", ed_part)
    uncited_ed = [l for l in ed_labels if not re.search(r"\\ref\*?\{%s\}" % re.escape(l), t)]
    supp_labels = re.findall(r"\\label\{(tab:[^}]*|fig:[^}]*)\}", re.sub(r"(?m)^\s*%.*$", "", s))
    uncited_supp = [l for l in supp_labels
                    if not re.search(r"\\ref\*?\{%s\}" % re.escape(l), t + s.replace(r"\label{%s}" % l, ""))]
    abstract = _words(section(r"\begin{abstract}", r"\end{abstract}"))
    body = _words(section(r"\section*{Introduction}", r"\section*{Methods}"))
    return {
        "abstract <= 150 words": (abstract, abstract <= 150),
        "main text <= 4,000 words (excl. abstract, Methods, refs, legends)": (body, body <= 4000),
        "display items <= 8": (f"{n_fig} figures, {n_tab} tables", n_fig + n_tab <= 8),
        "Extended Data items <= 10": (n_ed, n_ed <= 10),
        "references ~50 (guideline)": (len(cites), len(cites) <= 60),
        "unique labels": (", ".join(dup) or "none duplicated", not dup),
        "every Extended Data item cited": (", ".join(uncited_ed) or "all cited", not uncited_ed),
        "every Supplementary table/figure cited": (", ".join(uncited_supp) or "all cited", not uncited_supp),
        "no personal macros in submission copy": ("see submission/ + flatten warnings", True),
        # A plain \ref to a label in the other document becomes a hyperlink to a
        # target that does not exist in this PDF; cross-document refs use \ref*.
        "cross-document refs use \\ref*": (", ".join(bad_xrefs) or "all \\ref*", not bad_xrefs),
    }


def literal_macro_values(tex_paths: Sequence[Path]) -> Dict[str, str]:
    """`\\newcommand{\\Name}{value}` literals still defined inline in the .tex
    (pre-automation values), for the old -> new report column."""
    out = {}
    for tex in tex_paths:
        if Path(tex).exists():
            for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", Path(tex).read_text()):
                out.setdefault(m.group(1), m.group(2).strip())
    return out
