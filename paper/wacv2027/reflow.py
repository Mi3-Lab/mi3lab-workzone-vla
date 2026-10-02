#!/usr/bin/env python3
"""Reflow a .tex file so each prose paragraph is a single (long) source line,
removing mid-paragraph hard wraps that are annoying to edit in Overleaf.
Protects tables, math, listings, and structural lines so the compiled output
is unchanged.  Usage: python3 reflow.py main.tex
"""
import re
import sys

PROTECT_ENVS = {"tabular", "tabular*", "array", "align", "align*", "equation",
                "equation*", "matrix", "bmatrix", "pmatrix", "cases",
                "lstlisting", "verbatim", "tikzpicture"}

# a source line that must stay on its own line (flush the paragraph buffer)
STRUCT = re.compile(r"^\s*(\\begin|\\end|\\section|\\subsection|\\subsubsection|"
                    r"\\item|\\label|\\includegraphics|\\centering|\\maketitle|"
                    r"\\toprule|\\midrule|\\bottomrule|\\cmidrule|\\hline|"
                    r"\\bibliography|\\bibliographystyle|\\appendix|\\def|"
                    r"\\usepackage|\\documentclass|\\renewcommand|\\newcommand|"
                    r"\\setcounter|\\setlength|\\IfFileExists|\{)")


def env_name(line, kind):
    m = re.search(r"\\%s\{([^}]*)\}" % kind, line)
    return m.group(1) if m else None


def reflow(text):
    lines = text.split("\n")
    out = []
    buf = []
    protect_depth = 0

    def flush():
        if buf:
            out.append(" ".join(s.strip() for s in buf))
            buf.clear()

    for line in lines:
        stripped = line.strip()
        # inside a protected environment: emit verbatim
        if protect_depth > 0:
            out.append(line)
            e = env_name(line, "end")
            if e in PROTECT_ENVS:
                protect_depth -= 1
            b = env_name(line, "begin")
            if b in PROTECT_ENVS:
                protect_depth += 1
            continue
        # entering a protected environment
        b = env_name(line, "begin")
        if b in PROTECT_ENVS:
            flush(); out.append(line); protect_depth += 1
            continue
        # blank line -> paragraph break
        if stripped == "":
            flush(); out.append("")
            continue
        # comment line -> own line
        if stripped.startswith("%"):
            flush(); out.append(line)
            continue
        # structural line -> own line
        if STRUCT.match(stripped):
            flush(); out.append(line)
            continue
        # a table/hard-break row (ends with \\) -> keep boundary
        if stripped.endswith(r"\\"):
            buf.append(stripped); flush()
            continue
        # otherwise: prose, accumulate. A trailing bare % is a line-continuation
        # comment; drop it before joining (a literal percent is written \%).
        if stripped.endswith("%") and not stripped.endswith(r"\%"):
            stripped = stripped[:-1].rstrip()
        buf.append(stripped)
    flush()
    # collapse 3+ blank lines to max 1
    res = []
    blank = 0
    for l in out:
        if l == "":
            blank += 1
            if blank <= 1:
                res.append(l)
        else:
            blank = 0
            res.append(l)
    return "\n".join(res) + "\n"


if __name__ == "__main__":
    path = sys.argv[1]
    src = open(path).read()
    open(path, "w").write(reflow(src))
    print("reflowed", path)
