"""Build the web reader docs/paper/index.html from paper/main.tex. Run from repo root: python docs/build_web.py"""
import re, shutil, subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P, OUT = REPO / "paper", REPO / "docs" / "paper"
(OUT / "figures").mkdir(parents=True, exist_ok=True)
tex = (P / "main.tex").read_text()
tex = re.sub(r"figures/(\w+)\.pdf", r"figures/\1.png", tex)
tex = tex.replace(r"\usepackage{arxiv}", "")
tex = re.sub(r"\\rowcolor\{\w+\}", "", tex)
tex = re.sub(r"\\begin\{tabularx\}\{[^}]*\}", r"\\begin{tabular}", tex).replace(r"\end{tabularx}", r"\end{tabular}")
tex = re.sub(r"[LS]\{[^}]*\}|S\[[^\]]*\]|@\{\}", lambda m: "" if m.group(0) == "@{}" else "l", tex)
tex = re.sub(r"(\\begin\{tabular\}\{)([^}]*)\}", lambda m: m.group(1) + m.group(2).replace("Y", "l") + "}", tex)
tex = re.sub(r"\\(begin|end)\{threeparttable\}", "", tex)
tex = re.sub(r"\\begin\{tablenotes\}(.*?)\\end\{tablenotes\}",
             lambda m: m.group(1).replace(r"\item", "").replace(r"\footnotesize", ""), tex, flags=re.S)
tex = re.sub(r"\\ang\{([^}]*)\}", r"\1°", tex)
tex = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}",
             r"\\textit{[Diagram: see the PDF version.]}", tex, flags=re.S)
def cref(m):
    keys = m.group(2).split(",")
    return " and ".join(r"\%s{%s}" % (m.group(1), k) for k in keys)
tex = tex.replace(r"\label{eq:lp}", "")
tex = re.sub(r"\\cref\{eq:lp\}", "the local-projection equation", tex)
tex = re.sub(r"\\([cC]ref)\{([^}]*,[^}]*)\}", cref, tex)
tmp = P / "_web.tex"
tmp.write_text(tex)
subprocess.run(["pandoc", tmp.name, "-f", "latex", "-t", "html5", "-s", "--mathml", "--citeproc",
                "--bibliography", "refs.bib", "--template", str(REPO / "docs" / "web_template.html"),
                "--metadata", "link-citations=true", "-o", str(OUT / "index.html")], cwd=P, check=True)
tmp.unlink()
for png in (P / "figures").glob("*.png"):
    shutil.copy(png, OUT / "figures" / png.name)
shutil.copy(P / "paper.pdf", OUT / "paper.pdf")
print("wrote", OUT / "index.html")
