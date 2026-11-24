"""Render the preprint and the one-page summary to PDF with Typst.

Markdown -> Typst: pandoc (pypandoc_binary in this folder's .venv), with paper/typst/preprint.lua
                   (layout only: no wording or number changes) and paper/typst/template.typ.
Typst -> PDF:      the `typst` Python package, with the design in paper/typst/style.typ and the fonts
                   in paper/typst/fonts (fetch them once with `python tools/get_fonts.py`).

    python paper/render_typst.py              # preprint.pdf and summary.pdf
    python paper/render_typst.py preprint     # just one

The HTML route (paper/render.py) still works and stays as a fallback.
"""
import os
import sys

import pypandoc
import typst

HERE = os.path.dirname(os.path.abspath(__file__))
TYP = os.path.join(HERE, "typst")
FONTS = os.path.join(TYP, "fonts")
NEEDED = ["Inter-Regular.otf", "Inter-Medium.otf", "Inter-SemiBold.otf", "Inter-Bold.otf", "Inter-Italic.otf",
          "InterDisplay-Bold.otf", "SourceSerif4-Regular.otf", "SourceSerif4-It.otf", "SourceSerif4-Semibold.otf"]

JOBS = {
    "preprint": [],
    "summary": ["--metadata=date:One-page summary, November 2026", "--metadata=compact:true"],
}


def build(name):
    src, typ, pdf = (os.path.join(HERE, name + ext) for ext in (".md", ".typ", ".pdf"))
    args = ["--wrap=none", "--shift-heading-level-by=-1",
            f"--lua-filter={os.path.join(TYP, 'preprint.lua')}",
            f"--template={os.path.join(TYP, 'template.typ')}"] + JOBS[name]
    pypandoc.convert_file(src, "typst", format="markdown", outputfile=typ, extra_args=args)
    typst.compile(typ, output=pdf, root=HERE, font_paths=[FONTS])
    from pypdf import PdfReader
    print(f"wrote {pdf} ({len(PdfReader(pdf).pages)} pages)")
    return pdf


def main(argv):
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(FONTS, f))]
    if missing:
        sys.exit(f"fonts missing in {FONTS}: {missing}\nrun: python tools/get_fonts.py")
    for name in (argv or list(JOBS)):
        build(name)


if __name__ == "__main__":
    main(sys.argv[1:])
