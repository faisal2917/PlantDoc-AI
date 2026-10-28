"""Print every folder that contains images, with counts. Run: python inspect_data.py"""

from collections import Counter

from config import DATASET_DIR, VALID_EXTENSIONS

image_counts: Counter = Counter()
other_files: Counter = Counter()

for path in DATASET_DIR.rglob("*"):
    if not path.is_file():
        continue
    if path.suffix.lower() in VALID_EXTENSIONS:
        image_counts[path.parent.relative_to(DATASET_DIR)] += 1
    else:
        other_files[path.suffix.lower() or "(no extension)"] += 1

for folder in sorted(image_counts):
    print(f"{image_counts[folder]:6d}  {folder}")

print(f"\nFolders with images: {len(image_counts)}")
print(f"Total images: {sum(image_counts.values())}")
if other_files:
    print(f"Non-image files by extension: {dict(other_files)}")