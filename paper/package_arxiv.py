#!/usr/bin/env python3
"""Package the long report's checked LaTeX sources without repository data."""
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT.parent / "output" / "arxiv" / "articulate3d-source.tar.gz"


def main():
    sources = [ROOT / "main.tex", ROOT / "refs.bib", ROOT / "main.bbl"]
    sources += sorted((ROOT / "figs").glob("*.pdf"))
    for source in sources:
        if not source.is_file():
            raise SystemExit(f"Missing {source.name}; compile with tectonic --keep-intermediates main.tex first.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(OUTPUT, "w:gz") as archive:
        for source in sources:
            archive.add(source, arcname=source.relative_to(ROOT), recursive=False)
    print(OUTPUT)
    print(f"Packaged {len(sources)} source files; no data, credentials or build logs included.")


if __name__ == "__main__":
    main()
