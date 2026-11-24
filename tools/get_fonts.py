"""Fetch the two fonts the Typst build uses into paper/typst/fonts (gitignored).

Both are under the SIL Open Font License; their licence files are copied alongside.
    Inter 4.1            https://github.com/rsms/inter
    Source Serif 4.005   https://github.com/adobe-fonts/source-serif

    python tools/get_fonts.py
"""
import io
import os
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "paper", "typst", "fonts")

SOURCES = [
    ("https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip",
     {"extras/otf/" + f: f for f in [
         "Inter-Regular.otf", "Inter-Italic.otf", "Inter-Medium.otf", "Inter-MediumItalic.otf",
         "Inter-SemiBold.otf", "Inter-SemiBoldItalic.otf", "Inter-Bold.otf",
         "InterDisplay-SemiBold.otf", "InterDisplay-Bold.otf"]} | {"LICENSE.txt": "Inter-LICENSE.txt"}),
    ("https://github.com/adobe-fonts/source-serif/releases/download/4.005R/source-serif-4.005_Desktop.zip",
     {"source-serif-4.005_Desktop/OTF/" + f: f for f in [
         "SourceSerif4-Regular.otf", "SourceSerif4-It.otf", "SourceSerif4-Semibold.otf",
         "SourceSerif4-SemiboldIt.otf", "SourceSerif4-Bold.otf", "SourceSerif4-BoldIt.otf"]}
     | {"source-serif-4.005_Desktop/LICENSE.md": "SourceSerif4-LICENSE.md"}),
]


def main():
    os.makedirs(DEST, exist_ok=True)
    for url, members in SOURCES:
        if all(os.path.exists(os.path.join(DEST, out)) for out in members.values()):
            print(f"already present: {url.rsplit('/', 1)[-1]}")
            continue
        print(f"downloading {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        for member, out in members.items():
            with z.open(member) as src, open(os.path.join(DEST, out), "wb") as dst:
                dst.write(src.read())
    print(f"fonts in {DEST}")


if __name__ == "__main__":
    main()
