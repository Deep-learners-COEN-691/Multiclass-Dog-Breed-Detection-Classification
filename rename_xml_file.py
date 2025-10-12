from pathlib import Path

ann_root = Path("Dogbreeds/annotations/Annotation")  # <-- change me
count = 0
for p in ann_root.rglob("*"):
    if p.is_file() and p.suffix == "":
        try:
            head = p.read_bytes()[:128].decode("utf-8", errors="ignore")
            if "<annotation" in head:
                new = p.with_name(p.name + ".xml")
                if not new.exists():
                    p.rename(new)
                    count += 1
        except Exception:
            pass
print("Renamed", count, "files to .xml")
