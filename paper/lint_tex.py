"""Static checks for paper/main.tex (no LaTeX compiler on this machine; the paper is compiled on Overleaf).
Checks: brace balance, begin/end nesting, refs vs labels, cites vs refs.bib, figure files, stray % & _ #,
non-ASCII characters (pdfLaTeX), and math-mode balance per paragraph.
usage: python paper/lint_tex.py
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
tex = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(HERE, "refs.bib"), encoding="utf-8").read()
problems = []


def strip_comments(s):
    out = []
    for line in s.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        out.append(line[:m.start()] if m else line)
    return "\n".join(out)


body = strip_comments(tex)
preamble_end = body.find("\\begin{document}")


def blank_math_envs(s):
    # display-math environments count as math for the underscore check
    return re.sub(r"\\begin\{(equation|align)\*?\}.*?\\end\{\1\*?\}",
                  lambda m: "\n" * m.group(0).count("\n"), s, flags=re.S)

# 1. braces
depth = 0
for i, ch in enumerate(body):
    if ch in "{}" and (i == 0 or body[i - 1] != "\\"):
        depth += 1 if ch == "{" else -1
        if depth < 0:
            problems.append(f"unbalanced '}}' near line {body[:i].count(chr(10)) + 1}")
            depth = 0
if depth:
    problems.append(f"{depth} unclosed '{{'")

# 2. environments
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    line = body[:m.start()].count("\n") + 1
    if m.group(1) == "begin":
        stack.append((m.group(2), line))
    elif not stack or stack[-1][0] != m.group(2):
        problems.append(f"\\end{{{m.group(2)}}} at line {line} does not match {stack[-1] if stack else 'nothing'}")
    else:
        stack.pop()
if stack:
    problems.append(f"unclosed environments: {stack}")

# 3. refs and labels
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for r in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", body):
    if r not in labels:
        problems.append(f"undefined ref: {r}")
for l in labels:
    if not re.search(r"\\(?:ref|eqref|autoref)\{" + re.escape(l) + r"\}", body):
        print("note: label never referenced:", l)

# 4. citations
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for grp in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", body):
    cited |= {k.strip() for k in grp.split(",")}
for k in sorted(cited - bibkeys):
    problems.append(f"citation key not in refs.bib: {k}")
print("cited", len(cited), "of", len(bibkeys), "bib entries; uncited:", sorted(bibkeys - cited))

# 5. figures
for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
    if not os.path.exists(os.path.join(HERE, f)):
        problems.append(f"missing figure file: {f}")

# 6. stray special characters outside math / commands
for n, line in enumerate(blank_math_envs(body).split("\n"), 1):
    if re.search(r"(?<![\\$])#", line) and "\\newcommand" not in line:
        problems.append(f"line {n}: unescaped #")
    in_tab = False
    # underscores outside math and outside \label/\ref/\cite/\includegraphics/\texttt arguments
    clean = re.sub(r"\\(label|ref|eqref|cite[pt]?|includegraphics|bibliography(style)?|url|href)(\[[^\]]*\])?\{[^}]*\}", "", line)
    clean = re.sub(r"\$[^$]*\$", "", clean)
    clean = re.sub(r"\\\(.*?\\\)", "", clean)
    if re.search(r"(?<!\\)_", clean):
        problems.append(f"line {n}: underscore outside math: {line.strip()[:80]}")
    for ch in line:
        if ord(ch) > 127:
            problems.append(f"line {n}: non-ASCII character {ch!r}")
            break

# 7. ampersands outside tabular-like environments
envs = re.split(r"(\\begin\{(?:tabular|align\*?|array|matrix)\}|\\end\{(?:tabular|align\*?|array|matrix)\})", body[preamble_end:])
inside = False
for chunk in envs:
    if chunk.startswith("\\begin{"):
        inside = True
        continue
    if chunk.startswith("\\end{"):
        inside = False
        continue
    if not inside:
        for m in re.finditer(r"(?<!\\)&", chunk):
            problems.append(f"unescaped & outside tabular: ...{chunk[max(0, m.start() - 40):m.start() + 20]!r}")

# 8. dollar balance per paragraph
for p in re.split(r"\n\s*\n", body):
    if len(re.findall(r"(?<!\\)\$", p)) % 2:
        problems.append(f"odd number of $ in paragraph starting: {p.strip()[:70]!r}")

print("\n".join(problems) if problems else "no problems found")
sys.exit(1 if problems else 0)
