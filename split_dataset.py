import csv
import random
from collections import Counter, defaultdict
from datetime import datetime

INPUT_CSV       = "combined_filtered.csv"

TRAIN_CSV       = "train.csv"
TEST_CSV        = "test.csv"
TRAIN_TIME_CSV  = "train_time.csv"
TEST_TIME_CSV   = "test_time.csv"
SUMMARY_TXT     = "split_summary.txt"

RANDOM_SEED     = 42
TEST_FRACTION   = 0.20

random.seed(RANDOM_SEED)

# ------------------------------------------------------------------
# 1. Load the filtered dataset
# ------------------------------------------------------------------
rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["length"] = int(r["length"])
        r["n_count"] = int(r["n_count"])
        rows.append(r)

print(f"Loaded {len(rows)} records from {INPUT_CSV}")

fieldnames = ["accession","lineage","sequence","length","n_count",
              "collection_date","geographic_region"]

# ------------------------------------------------------------------
# 2. Random stratified 80:20 split
# ------------------------------------------------------------------
by_lineage = defaultdict(list)
for r in rows:
    by_lineage[r["lineage"]].append(r)

train_rows, test_rows = [], []

for lin, items in by_lineage.items():
    random.shuffle(items)
    n = len(items)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    test_rows.extend(items[:n_test])
    train_rows.extend(items[n_test:])

random.shuffle(train_rows)
random.shuffle(test_rows)

def write_csv(path, data):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in data:
            writer.writerow({k: r[k] for k in fieldnames})

write_csv(TRAIN_CSV, train_rows)
write_csv(TEST_CSV, test_rows)

# ------------------------------------------------------------------
# 3. Time-based 80:20 split
#    Strategy: within each lineage, sort by collection_date and take
#    the earliest 80% as train, latest 20% as test. This keeps every
#    class present in both sets while preserving chronological order.
# ------------------------------------------------------------------
def parse_date(s):
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

train_time_rows, test_time_rows = [], []
skipped_no_date = 0

for lin, items in by_lineage.items():
    dated = []
    for r in items:
        d = parse_date(r.get("collection_date", ""))
        if d is None:
            skipped_no_date += 1
            continue
        r["_date"] = d
        dated.append(r)

    dated.sort(key=lambda r: r["_date"])
    n = len(dated)
    n_test = max(1, int(round(n * TEST_FRACTION)))
    test_part = dated[-n_test:]
    train_part = dated[:-n_test]

    for r in train_part:
        r.pop("_date", None)
    for r in test_part:
        r.pop("_date", None)

    train_time_rows.extend(train_part)
    test_time_rows.extend(test_part)

random.shuffle(train_time_rows)
random.shuffle(test_time_rows)

write_csv(TRAIN_TIME_CSV, train_time_rows)
write_csv(TEST_TIME_CSV, test_time_rows)

# ------------------------------------------------------------------
# 4. Summary report
# ------------------------------------------------------------------
def counts(data):
    return Counter(r["lineage"] for r in data)

lines = []
lines.append("TRAIN / TEST SPLIT SUMMARY")
lines.append("=" * 60)
lines.append(f"Input          : {INPUT_CSV}")
lines.append(f"Random seed    : {RANDOM_SEED}")
lines.append(f"Test fraction  : {TEST_FRACTION}")
lines.append(f"Skipped (no date, time split): {skipped_no_date}")
lines.append("")

lines.append("RANDOM STRATIFIED SPLIT")
lines.append(f"  Train total: {len(train_rows)}")
lines.append(f"  Test total : {len(test_rows)}")
lines.append("  Per-class counts (train / test):")
c_train = counts(train_rows)
c_test = counts(test_rows)
for lin in sorted(set(c_train) | set(c_test)):
    lines.append(f"    {lin:15s} : {c_train.get(lin,0):5d} / {c_test.get(lin,0):5d}")
lines.append("")

lines.append("TIME-BASED SPLIT (per-lineage chronological 80:20)")
lines.append(f"  Train total: {len(train_time_rows)}")
lines.append(f"  Test total : {len(test_time_rows)}")
lines.append("  Per-class counts (train / test):")
ct_train = counts(train_time_rows)
ct_test = counts(test_time_rows)
for lin in sorted(set(ct_train) | set(ct_test)):
    lines.append(f"    {lin:15s} : {ct_train.get(lin,0):5d} / {ct_test.get(lin,0):5d}")
lines.append("")

lines.append("Collection-date range per split:")
def date_range(data):
    dates = [parse_date(r.get("collection_date","")) for r in data]
    dates = [d for d in dates if d is not None]
    if not dates:
        return "(none)"
    return f"{min(dates).date()} to {max(dates).date()}"

lines.append(f"  Random train: {date_range(train_rows)}")
lines.append(f"  Random test : {date_range(test_rows)}")
lines.append(f"  Time train  : {date_range(train_time_rows)}")
lines.append(f"  Time test   : {date_range(test_time_rows)}")

summary_text = "\n".join(lines)
print()
print(summary_text)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nWrote {TRAIN_CSV}")
print(f"Wrote {TEST_CSV}")
print(f"Wrote {TRAIN_TIME_CSV}")
print(f"Wrote {TEST_TIME_CSV}")
print(f"Wrote {SUMMARY_TXT}")