"""Score project Countix predictions using separately obtained source labels."""
from pathlib import Path
import argparse, csv, json, statistics

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', required=True, type=Path, help='Local CSV: unit_id,count')
    args = parser.parse_args()
    with args.labels.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    gold = {}
    for row in rows:
        uid = row['unit_id']
        if uid in gold:
            raise ValueError('Duplicate source label ID: ' + uid)
        gold[uid] = int(row['count'])
    root = Path(__file__).resolve().parents[1]
    seeds = []
    for path in sorted((root / '02_predictions').glob('countix*.jsonl')):
        pred = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        if len({r['unit_id'] for r in pred}) != len(pred):
            raise ValueError('Duplicate prediction IDs: ' + path.name)
        missing = [r['unit_id'] for r in pred if r['unit_id'] not in gold]
        if missing:
            raise ValueError(f'{path.name}: {len(missing)} source labels missing')
        if path.name.startswith('countix_refit_seed') and len(pred) != 476:
            raise ValueError('Expected 476 final-refit predictions: ' + path.name)
        hits = sum(type(r.get('pred')) is int and min(r['pred'], 4) == min(gold[r['unit_id']], 4) for r in pred)
        acc = 100 * hits / len(pred)
        print(f'{path.name}: n={len(pred)} exact4={acc:.6f}%')
        if path.name.startswith('countix_refit_seed'):
            seeds.append(acc)
    if len(seeds) == 5:
        print(f'Refit five-seed mean={statistics.mean(seeds):.6f}%; population SD={statistics.pstdev(seeds):.6f} pp; sample SD={statistics.stdev(seeds):.6f} pp')

if __name__ == '__main__':
    main()
