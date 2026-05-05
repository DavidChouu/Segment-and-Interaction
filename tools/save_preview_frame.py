from pathlib import Path
import shutil
import sys

if len(sys.argv) != 3:
    print('usage: python tools/save_preview_frame.py <source_png> <gallery_dir>')
    sys.exit(1)

source = Path(sys.argv[1])
gallery = Path(sys.argv[2])
if not source.exists():
    raise FileNotFoundError(source)

gallery.mkdir(parents=True, exist_ok=True)
existing = sorted(gallery.glob('*.png'))
next_index = 0
if existing:
    names = []
    for p in existing:
        try:
            names.append(int(p.stem))
        except ValueError:
            pass
    if names:
        next_index = max(names) + 1

dst = gallery / f'{next_index:04d}.png'
shutil.copy2(source, dst)
print(dst)
