"""Verify SHA256SUMS.txt without modifying the release."""
from pathlib import Path
import hashlib

root = Path(__file__).resolve().parents[1]
errors = []
checked = 0
for line in (root / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
    expected, rel = line.split('  ', 1)
    path = root / rel
    if not path.is_file():
        errors.append('missing: ' + rel)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append('changed: ' + rel)
    checked += 1
for problem in errors:
    print(problem)
print(f'Checked {checked} files; {len(errors)} mismatches.')
raise SystemExit(1 if errors else 0)
