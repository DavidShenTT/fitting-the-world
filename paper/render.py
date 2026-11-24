"""Render paper/preprint.md to a standalone HTML file, and a fallback PDF from it.

The main PDFs (preprint.pdf, summary.pdf) are built with Typst by paper/render_typst.py; this is the HTML route.

Markdown -> HTML: pandoc (pypandoc_binary in this folder's .venv), images embedded.
HTML -> preprint_html.pdf: the installed Microsoft Edge in headless mode, with its profile kept in .cache/edge (nothing is installed).
    python paper/render.py
"""
import os
import subprocess
import sys

import pypandoc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC, HTML, PDF = (os.path.join(HERE, f) for f in ("preprint.md", "preprint.html", "preprint_html.pdf"))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


def main():
    pypandoc.convert_file(SRC, "html5", outputfile=HTML,
                          extra_args=["--standalone", "--embed-resources", f"--css={os.path.join(HERE, 'style.css')}",
                                      f"--resource-path={HERE}", "--metadata=lang:en"])
    prof = os.path.join(ROOT, ".cache", "edge")
    os.makedirs(prof, exist_ok=True)
    cmd = [EDGE, "--headless", "--disable-gpu", "--no-first-run", "--no-pdf-header-footer", f"--user-data-dir={prof}",
           f"--print-to-pdf={PDF}", "file:///" + HTML.replace("\\", "/")]
    subprocess.run(cmd, check=True, timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from pypdf import PdfReader
    print(f"wrote {HTML} and {PDF} ({len(PdfReader(PDF).pages)} pages)")


if __name__ == "__main__":
    sys.exit(main())
