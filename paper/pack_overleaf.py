"""Zip the files Overleaf needs (main.tex, refs.bib, figures/*.pdf) into paper/overleaf_upload.zip.
Upload on Overleaf with New Project -> Upload Project; compiler pdfLaTeX (default), BibTeX runs automatically.
usage: python paper/pack_overleaf.py
"""
import glob
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "overleaf_upload.zip")
files = ["main.tex", "refs.bib"] + sorted(os.path.relpath(p, HERE) for p in glob.glob(os.path.join(HERE, "figures", "*.pdf")))
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(os.path.join(HERE, f), f.replace(os.sep, "/"))
print(out, os.path.getsize(out) // 1024, "KB:", files)
