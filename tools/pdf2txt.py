"""pdf2txt.py SRC.pdf DEST.txt [start_char end_char] -- extract text from a PDF and optionally print a window."""
import sys, pypdf
src, dst = sys.argv[1], sys.argv[2]
r = pypdf.PdfReader(src)
txt = "\n".join((p.extract_text() or "") for p in r.pages)
open(dst, "w", encoding="utf-8").write(txt)
sys.stdout.reconfigure(encoding="utf-8")
print(f"[{len(r.pages)} pages, {len(txt)} chars]")
if len(sys.argv) > 4:
    print(txt[int(sys.argv[3]):int(sys.argv[4])])
