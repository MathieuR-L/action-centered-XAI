import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEX_PATH = ROOT / "paper" / "main.tex"
BIB_PATH = ROOT / "paper" / "references.bib"


def main() -> None:
    tex = TEX_PATH.read_text(encoding="utf-8")
    bib = BIB_PATH.read_text(encoding="utf-8")

    cite_keys = set()
    for match in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for key in match.group(1).split(","):
            cite_keys.add(key.strip())

    bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", bib))
    missing = sorted(cite_keys - bib_keys)

    print(f"TeX cite keys: {len(cite_keys)}")
    print(f"Bib entries: {len(bib_keys)}")
    print(f"Missing keys: {len(missing)}")
    for key in missing:
        print(key)

    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
