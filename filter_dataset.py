import csv
from collections import Counter, defaultdict

INPUT_CSV       = "combined_dataset.csv"
OUTPUT_CSV      = "combined_filtered.csv"
REMOVED_CSV     = "removed_sequences.csv"
SUMMARY_TXT     = "combined_filtered_summary.txt"

MIN_LENGTH      = 29000
MAX_LENGTH      = 30000
MAX_N_FRACTION  = 0.05     # 5% N cutoff

# ------------------------------------------------------------------
# 1. Load the combined dataset
# ------------------------------------------------------------------
rows = []
with open(INPUT_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["length"] = int(r["length"])
        r["n_count"] = int(r["n_count"])
        rows.append(r)

total_in = len(rows)
print(f"Loaded {total_in} records from {INPUT_CSV}")

# ------------------------------------------------------------------
# 2. Track removals
# ------------------------------------------------------------------
kept = []
removed = []

seen_sequences = set()

before_counts = Counter(r["lineage"] for r in rows)

for r in rows:
    reason = None
    seq = r["sequence"]

    # length check
    if r["length"] < MIN_LENGTH or r["length"] > MAX_LENGTH:
        reason = f"length_out_of_range ({r['length']})"

    # N-count check
    elif r["n_count"] / r["length"] > MAX_N_FRACTION:
        frac = r["n_count"] / r["length"]
        reason = f"too_many_Ns ({r['n_count']} / {r['length']} = {frac:.3f})"

    # duplicate check
    elif seq in seen_sequences:
        reason = "duplicate_sequence"

    if reason:
        removed.append({**r, "reason": reason})
    else:
        seen_sequences.add(seq)
        kept.append(r)

# ------------------------------------------------------------------
# 3. Write the filtered CSV
# ------------------------------------------------------------------
fieldnames = ["accession","lineage","sequence","length","n_count",
              "collection_date","geographic_region"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in kept:
        writer.writerow({k: r[k] for k in fieldnames})

# ------------------------------------------------------------------
# 4. Write the removed CSV
# ------------------------------------------------------------------
with open(REMOVED_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames + ["reason"])
    writer.writeheader()
    for r in removed:
        writer.writerow({k: r[k] for k in fieldnames + ["reason"]})

# ------------------------------------------------------------------
# 5. Build the summary report
# ------------------------------------------------------------------
after_counts = Counter(r["lineage"] for r in kept)
removed_counts = Counter(r["lineage"] for r in removed)

lines = []
lines.append("FILTERED DATASET SUMMARY")
lines.append("=" * 60)
lines.append(f"Input file            : {INPUT_CSV}")
lines.append(f"Output file           : {OUTPUT_CSV}")
lines.append(f"Length range kept     : [{MIN_LENGTH}, {MAX_LENGTH}]")
lines.append(f"Max N fraction allowed: {MAX_N_FRACTION}")
lines.append("")
lines.append(f"Total in              : {total_in}")
lines.append(f"Total kept            : {len(kept)}")
lines.append(f"Total removed         : {len(removed)}")
lines.append("")

lines.append("Per-lineage counts (before -> after):")
for lin in sorted(before_counts):
    lines.append(
        f"  {lin:15s} : {before_counts[lin]:5d} -> {after_counts.get(lin, 0):5d}"
        f"   (removed: {removed_counts.get(lin, 0)})"
    )
lines.append("")

lines.append("Removal reasons (overall):")
reason_counter = Counter(r["reason"].split(" ")[0] for r in removed)
for reason, n in reason_counter.most_common():
    lines.append(f"  {reason:25s} : {n}")

lines.append("")
lines.append("Final class sizes:")
for lin in sorted(after_counts):
    lines.append(f"  {lin:15s} : {after_counts[lin]}")

summary_text = "\n".join(lines)
print()
print(summary_text)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nWrote {OUTPUT_CSV}")
print(f"Wrote {REMOVED_CSV}")
print(f"Wrote {SUMMARY_TXT}")