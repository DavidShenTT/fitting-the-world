"""Fill the author name into the paper, summary, README and licence files, then rebuild the PDFs and the HTML.

Replaces every "[Author]" placeholder in the files below with the given name. Run once, before publishing:
    python tools/set_author.py "Firstname Lastname"
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ["LICENSE", "LICENSE-text.md", "README.md", "paper/preprint.md", "paper/summary.md"]
PLACEHOLDER = "[Author]"


def main():
    if len(sys.argv) != 2 or not sys.argv[1].strip() or "[" in sys.argv[1]:
        sys.exit('usage: python tools/set_author.py "Firstname Lastname"')
    name = sys.argv[1].strip()
    for rel in FILES:
        path = os.path.join(ROOT, rel)
        raw = open(path, "rb").read().decode("utf-8")
        n = raw.count(PLACEHOLDER)
        open(path, "w", encoding="utf-8", newline="").write(raw.replace(PLACEHOLDER, name))
        print(f"{rel}: {n} replaced")
    subprocess.run([sys.executable, os.path.join(ROOT, "paper", "render_typst.py")], check=True)   # preprint.pdf, summary.pdf
    subprocess.run([sys.executable, os.path.join(ROOT, "paper", "render.py")], check=True)          # preprint.html


if __name__ == "__main__":
    main()
