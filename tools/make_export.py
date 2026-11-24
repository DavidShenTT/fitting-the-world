"""Build a clean copy of the repository for public release in export/ (git-ignored). Nothing is pushed or uploaded.

INCLUDE lists what goes out; everything else stays. After copying, the script scans the export for personal details
(first names, local user paths, email addresses) and lists every hit for review.
    python tools/make_export.py
"""
import fnmatch
import os
import re
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "export")

INCLUDE = [
    "README.md", "LICENSE", "LICENSE-text.md", "requirements.txt", "env.sh", "HISTORY.md", "CITATION.cff", ".zenodo.json",
    "BRIEF.md", "BRIEF_v2_OLD.md", "PLAN.md", "PREREGISTRATION.md", "DEVIATIONS.md", "REPORT.md", "REVIEW_1.md",
    "CHECKIN_3.md", "LOG.md", "PILOT*.md", "PILOT_GIT_LOG.txt",
    "src/*.py", "runs/*.py", "runs/*.sh", "analysis/*.py", "analysis/figures/*.py", "tests/*.py", "tools/*.py",
    "results/**/*.csv", "results/**/*.txt", "results/**/*.json",
    "pilot*/*.py", "pilot*/results/*",
    "paper/preprint.md", "paper/preprint.pdf", "paper/preprint.html", "paper/summary.md", "paper/style.css",
    "paper/render.py", "paper/verify_numbers.py", "paper/figures/*",
    "paper/summary.pdf", "paper/render_typst.py", "paper/typst/*.typ", "paper/typst/*.lua",
]
EXCLUDE = [  # never exported, even if a pattern above matches
    ".venv/*", ".cache/*", "export/*", "logs/*", "results/**/parts/*", "**/__pycache__/*",
    "CLAUDE.md",          # instructions for the coding agent; machine-specific
    "STATUS.md", "PAPER_BRIEF.md", "paper/OUTLINE.md", "paper/reading_notes.md",   # internal working notes
    "paper/TYPST_BRIEF.md", "paper/typst/fonts/*", "paper/preview/*", "PUBLISH_BRIEF.md", "_backup/*",   # build brief; fetched fonts; local previews
]
# First names in the decision records are kept by the author's decision (23 Nov); local paths and emails are scanned.
SCAN = [r"C:\\Users\\", r"/c/Users/", r"[A-Za-z]:\\ML\\", r"/[a-z]/ML/", r"AppData", r"[\w.+-]+@[\w-]+\.[\w.]+"]


def match(rel, pats):
    rel = rel.replace("\\", "/")
    return any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p.replace("**/", "")) for p in pats)


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    files = []
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in (".git", ".venv", ".cache", "export", "__pycache__", "_backup")]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), ROOT).replace("\\", "/")
            if match(rel, INCLUDE) and not match(rel, EXCLUDE):
                files.append(rel)
    for rel in files:
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, rel), dst)
    size = sum(os.path.getsize(os.path.join(OUT, r)) for r in files)
    print(f"exported {len(files)} files, {size / 1e6:.1f} MB, to {OUT}")
    hits = {}
    for rel in files:
        if rel.endswith((".png", ".pdf", ".svg")):
            continue
        try:
            txt = open(os.path.join(OUT, rel), encoding="utf-8").read()
        except UnicodeDecodeError:
            continue
        for pat in SCAN:
            n = len(re.findall(pat, txt))
            if n:
                hits.setdefault(pat, []).append(f"{rel} ({n})")
    print("personal-detail scan (review before publishing):")
    for pat, where in hits.items():
        print(f"  {pat}: {len(where)} files: " + ", ".join(where[:12]) + (" ..." if len(where) > 12 else ""))
    if not hits:
        print("  no hits")


if __name__ == "__main__":
    main()
