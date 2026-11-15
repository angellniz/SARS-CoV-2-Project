import os
import json
import csv
import re

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
FASTA_FILES = {
    "Alpha":        "alpha.fasta",
    "Beta":         "beta.fasta",
    "Gamma":        "gamma.fasta",
    "Delta":        "delta.fasta",
    "Omicron_BA1":  "omicron.fasta",
}

META_FILES = {
    "Alpha":        os.path.join("meta", "alpha_meta.jsonl"),
    "Beta":         os.path.join("meta", "beta_meta.jsonl"),
    "Gamma":        os.path.join("meta", "gamma_meta.jsonl"),
    "Delta":        os.path.join("meta", "delta_meta.jsonl"),
    "Omicron_BA1":  os.path.join("meta", "omicron_meta.jsonl"),
}

OUTPUT_CSV = "combined_dataset.csv"
SUMMARY_TXT = "combined_dataset_summary.txt"

# ------------------------------------------------------------------
# 1. Load metadata: accession -> (collection_date, region)
# ------------------------------------------------------------------
def load_metadata(path):
    meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            acc = rec.get("accession")
            if not acc:
                continue
            isolate = rec.get("isolate", {}) or {}
            date = isolate.get("collectionDate", "")
            location = rec.get("location", {}) or {}
            region = location.get("geographicRegion", "")
            meta[acc] = (date, region)
    return meta

# ------------------------------------------------------------------
# 2. Parse a FASTA file: returns list of (accession, sequence)
# ------------------------------------------------------------------
def parse_fasta(path):
    records = []
    current_acc = None
    current_seq = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current_acc is not None:
                    records.append((current_acc, "".join(current_seq)))
                header = line[1:].strip()
                # NCBI FASTA headers look like:
                # >OY958312.1 Severe acute respiratory syndrome coronavirus 2 ...
                current_acc = header.split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_acc is not None:
            records.append((current_acc, "".join(current_seq)))
    return records

# ------------------------------------------------------------------
# 3. Build the combined dataset
# ------------------------------------------------------------------
all_rows = []

for lineage, fasta_path in FASTA_FILES.items():
    if not os.path.exists(fasta_path):
        print(f"WARNING: missing {fasta_path}, skipping")
        continue

    print(f"Loading metadata for {lineage}...")
    meta = load_metadata(META_FILES[lineage])

    print(f"Parsing {fasta_path}...")
    records = parse_fasta(fasta_path)

    for acc, seq in records:
        seq_upper = seq.upper()
        length = len(seq_upper)
        n_count = seq_upper.count("N")
        date, region = meta.get(acc, ("", ""))

        all_rows.append({
            "accession": acc,
            "lineage": lineage,
            "sequence": seq_upper,
            "length": length,
            "n_count": n_count,
            "collection_date": date,
            "geographic_region": region,
        })

# ------------------------------------------------------------------
# 4. Write the combined CSV
# ------------------------------------------------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "accession",
            "lineage",
            "sequence",
            "length",
            "n_count",
            "collection_date",
            "geographic_region",
        ],
    )
    writer.writeheader()
    for row in all_rows:
        writer.writerow(row)

print(f"\nWrote {len(all_rows)} records to {OUTPUT_CSV}")

# ------------------------------------------------------------------
# 5. Write a summary report
# ------------------------------------------------------------------
from collections import Counter, defaultdict

class_counts = Counter(r["lineage"] for r in all_rows)
lengths = defaultdict(list)
n_counts = defaultdict(list)
region_counts = defaultdict(Counter)
date_missing = 0

for r in all_rows:
    lengths[r["lineage"]].append(r["length"])
    n_counts[r["lineage"]].append(r["n_count"])
    region_counts[r["lineage"]][r["geographic_region"]] += 1
    if not r["collection_date"]:
        date_missing += 1

lines = []
lines.append("COMBINED DATASET SUMMARY")
lines.append("=" * 60)
lines.append(f"Total records: {len(all_rows)}")
lines.append(f"Records with missing collection_date: {date_missing}")
lines.append("")
lines.append("Per-lineage counts:")
for lin in sorted(class_counts):
    lines.append(f"  {lin:15s} : {class_counts[lin]}")
lines.append("")
lines.append("Per-lineage sequence length (min / mean / max):")
for lin in sorted(lengths):
    L = lengths[lin]
    lines.append(f"  {lin:15s} : {min(L)} / {sum(L)/len(L):.1f} / {max(L)}")
lines.append("")
lines.append("Per-lineage N-count (min / mean / max):")
for lin in sorted(n_counts):
    N = n_counts[lin]
    lines.append(f"  {lin:15s} : {min(N)} / {sum(N)/len(N):.1f} / {max(N)}")
lines.append("")
lines.append("Top geographic regions per lineage:")
for lin in sorted(region_counts):
    top = region_counts[lin].most_common(5)
    lines.append(f"  {lin:15s} : " + ", ".join(f"{r or '(blank)'}={c}" for r, c in top))

summary_text = "\n".join(lines)
print()
print(summary_text)

with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nWrote summary to {SUMMARY_TXT}")