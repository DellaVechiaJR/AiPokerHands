import argparse
import os
import sys
import re

SUITS = {"♣": "CLUBS", "♥": "HEARTS", "♦": "DIAMONDS", "♠": "SPADES"}

MOJIBAKE_TO_SYMBOL = {
    "â™£": "♣",
    "â™¥": "♥",
    "â™¦": "♦",
    "â™¤": "♦",
    "â™\xa0": "♠",
    "â™ ": "♠",
    "â™¬": "♣",
    "â™¨": "♦",
}

MOJIBAKE_MARKERS = ("â™", "Ã", "Â", "â€“", "â€”", "â€", "â€™", "â€œ", "â€\x9d")

def looks_mojibake(s: str) -> bool:
    return isinstance(s, str) and any(m in s for m in MOJIBAKE_MARKERS)

def reverse_mojibake(s: str) -> str:
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin1", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return s

def normalize_suits_symbols(s: str) -> str:
    if not isinstance(s, str):
        return s

    if looks_mojibake(s):
        s = reverse_mojibake(s)

    for bad, sym in MOJIBAKE_TO_SYMBOL.items():
        s = s.replace(bad, sym)

    return s

def textify_cell(s: str) -> str:
    if not isinstance(s, str):
        return s

    s = normalize_suits_symbols(s)

    s = re.sub(r"(♣|♥|♦|♠)(?=[AKQJ]|10|[2-9])", r"\1 ", s)

    for sym, word in SUITS.items():
        s = s.replace(sym, word)

    return s

def process_text_file(in_path: str, out_path: str = None):
    with open(in_path, "rb") as f:
        raw = f.read()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin1", errors="replace")

    lines = text.splitlines()
    fixed = [textify_cell(ln) for ln in lines]
    result = "\n".join(fixed)

    if not out_path:
        root, ext = os.path.splitext(in_path)
        out_path = f"{root}_textified{ext or '.csv'}"

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(result)

    print(f"[OK] Salvo: {out_path}")

def process_xlsx(in_path: str, out_path: str = None):
    try:
        import pandas as pd
    except ImportError:
        print("[ERRO] Para XLSX, instale: pip install pandas openpyxl")
        sys.exit(1)

    xls = pd.ExcelFile(in_path)
    sheets = {}
    for sheet in xls.sheet_names:
        df = xls.parse(sheet, dtype=object)
        for col in df.columns:
            df[col] = df[col].map(lambda x: textify_cell(x))
        sheets[sheet] = df

    if not out_path:
        root, _ = os.path.splitext(in_path)
        out_path = f"{root}_textified.xlsx"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    print(f"[OK] Salvo: {out_path}")

def main():
    in_path = r"C:\Python_Projects\poker_dataset.csv"
    
    ap = argparse.ArgumentParser(description="Troca símbolos de naipes por texto em inglês (CLUBS/HEARTS/DIAMONDS/SPADES).")
    ap.add_argument("--out", dest="out_path", help="Arquivo de saída (opcional)")
    args = ap.parse_args()

    out_path = args.out_path

    if not os.path.exists(in_path):
        print(f"[ERRO] Arquivo não encontrado: {in_path}")
        sys.exit(1)

    ext = os.path.splitext(in_path)[1].lower()

    if ext in (".csv", ".txt", ".tsv"):
        process_text_file(in_path, out_path)
    elif ext == ".xlsx":
        process_xlsx(in_path, out_path)
    else:
        with open(in_path, "rb") as f:
            if f.read(4) == b"PK\x03\x04":
                process_xlsx(in_path, out_path)
            else:
                process_text_file(in_path, out_path)

if __name__ == "__main__":
    main()
