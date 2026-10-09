"""
Write Nature-Genetics-compliant copies of the manuscript sources (final
LaTeX submissions may not contain personal macros): every registry macro
(`Figures/generated/macros.tex`) and `\\excalibr` expanded to literal text,
generated table bodies inlined, `\\mycomment{...}` dropped, and
`\\textcolor{blue}{...}` revision highlighting unwrapped. The working .tex
files keep their macros; only the copies under `out_dir` are flattened.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

from analysis.manuscript_numbers import MANAGED_MACRO_RE


def read_macros(macros_tex: Path) -> Dict[str, str]:
    out = {}
    for line in Path(macros_tex).read_text().splitlines():
        m = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*%", line) or \
            re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line)
        if m:
            value = m.group(2)
            hl = re.fullmatch(r"\\textcolor\{red\}\{(.*)\}", value)
            out[m.group(1)] = hl.group(1) if hl else value
    return out


def _matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError(f"unbalanced brace at {open_idx}")


def unwrap_command(text: str, command: str, drop_content: bool = False) -> str:
    r"""Replace `<command>{...}` by the braced argument's content (or drop it).
    `command` includes any fixed leading arguments, e.g. r"\textcolor{blue}"."""
    pattern = re.compile(re.escape(command) + r"(?![A-Za-z])")
    out, pos = [], 0
    while True:
        m = pattern.search(text, pos)
        if not m:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:m.start()])
        i = m.end()
        while i < len(text) and text[i] in " \n\t":
            i += 1
        if i >= len(text) or text[i] != "{":
            out.append(text[m.start():m.end()])
            pos = m.end()
            continue
        j = _matching_brace(text, i)
        if not drop_content:
            out.append(text[i + 1:j])
        pos = j + 1


def _uncommented(tex: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", tex)


def _uses(tex: str, name: str) -> bool:
    """Is \\name used outside comments and its own \\newcommand line?"""
    body = re.sub(r"\\newcommand\{\\%s\}.*" % name, "", _uncommented(tex))
    return re.search(r"\\%s(?![A-Za-z])" % name, body) is not None


def expand_personal_macros(tex: str) -> str:
    """Expand the author's own 0- and 1-argument \\newcommand macros wherever
    they are used outside comments, and drop their definitions. Macros with
    more arguments are left for the leftover check to report."""
    defs = {}
    for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}(?:\[(\d)\])?\{", tex):
        line_start = tex.rfind("\n", 0, m.start()) + 1
        if "%" in re.sub(r"\\%", "", tex[line_start:m.start()]):
            continue  # commented-out definition
        nargs = int(m.group(2) or 0)
        end = _matching_brace(tex, m.end() - 1)
        if nargs <= 1:
            defs[m.group(1)] = (nargs, tex[m.end():end], m.start(), end + 1)
    for name, (nargs, body, start, stop) in sorted(defs.items(), key=lambda kv: -kv[1][2]):
        if not _uses(tex, name):
            continue
        definition = tex[start:stop]
        tex = tex[:start] + tex[stop:]
        out, pos = [], 0
        for m in re.finditer(r"\\%s(?![A-Za-z])" % name, tex):
            if m.start() < pos:
                continue
            line_start = tex.rfind("\n", 0, m.start()) + 1
            if "%" in re.sub(r"\\%", "", tex[line_start:m.start()]):
                continue  # inside a comment; leave as written
            out.append(tex[pos:m.start()])
            if nargs == 0:
                out.append(body)
                pos = m.end()
                if tex[pos:pos + 2] == "{}":
                    pos += 2
            else:
                i = m.end()
                while i < len(tex) and tex[i] in " \t":
                    i += 1
                if i < len(tex) and tex[i] == "{":
                    j = _matching_brace(tex, i)
                    arg = tex[i + 1:j]
                    pos = j + 1
                else:  # single-token argument, e.g. \mb R
                    arg = tex[i]
                    pos = i + 1
                out.append("{" + body.replace("#1", arg) + "}")
        out.append(tex[pos:])
        tex = "".join(out)
    return tex


def flatten_tex(tex: str, macros: Dict[str, str], figure_root: Path) -> Tuple[str, List[str]]:
    problems = []
    tex = re.sub(r"[ \t]*\\input\{Figures/generated/macros(?:\.tex)?\}[ \t]*\n?", "", tex)

    def inline_table(m):
        rel = m.group(1)
        path = Path(figure_root) / rel[len("Figures/"):]
        if path.suffix != ".tex":
            path = path.with_suffix(".tex")
        if not path.exists():
            problems.append(f"missing table body {path}")
            return m.group(0)
        return path.read_text().rstrip("\n")

    tex = re.sub(r"\\input\{(Figures/generated/tables/[^}]*)\}", inline_table, tex)

    def expand(m):
        name = m.group(1)
        if name not in macros:
            problems.append(f"\\{name} has no generated value")
            return m.group(0)
        return macros[name]

    tex = re.sub(MANAGED_MACRO_RE.pattern + r"(?:\{\})?", expand, tex)
    tex = re.sub(r"[ \t]*\\newcommand\{\\excalibr\}\{[^\n]*\n", "", tex)
    tex = re.sub(r"\\excalibr(?![A-Za-z])", "ExCALIBR", tex)
    tex = unwrap_command(tex, r"\mycomment", drop_content=True)
    tex = re.sub(r"[ \t]*\\newcommand\{\\mycomment\}\[1\]\{\}[ \t]*\n", "", tex)
    tex = unwrap_command(tex, r"\textcolor{blue}")
    tex = unwrap_command(tex, r"\textcolor{red}")
    todo = re.findall(r"\[TO (?:WRITE|FILL)[^\]]*\]", tex)
    problems += [f"unresolved author marker: {m[:100]}" for m in todo]

    tex = expand_personal_macros(tex)

    leftovers = sorted({m.group(1) for m in MANAGED_MACRO_RE.finditer(re.sub(r"(?<!\\)%.*", "", tex))})
    problems += [f"\\{n} still present after flattening" for n in leftovers]
    used_personal = sorted({
        name for name in re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", tex)
        if _uses(tex, name)
    })
    if used_personal:
        problems.append("personal macros still defined and used (expand by hand or extend flatten): "
                        + ", ".join("\\" + n for n in used_personal))
    return tex, problems


def flatten_manuscript(tex_paths, macros_tex: Path, figure_root: Path, out_dir: Path) -> List[str]:
    macros = read_macros(macros_tex)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_problems = []
    for tex_path in tex_paths:
        flat, problems = flatten_tex(Path(tex_path).read_text(), macros, figure_root)
        (out_dir / Path(tex_path).name).write_text(flat)
        all_problems += [f"{Path(tex_path).name}: {p}" for p in problems]
        print(f"  Flattened {tex_path} -> {out_dir / Path(tex_path).name}")
    for p in all_problems:
        print(f"  FLATTEN: {p}")
    return all_problems
