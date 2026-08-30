"""
LEADFORGE SPRINT — Stage 4 validation, step 2
Compares your scorecard's ranking of the 20 sample leads against 3
teammates' independent human rankings, using Spearman correlation.

Expects a CSV with a lead_id column plus 3 columns whose names match the
pattern human<N>_rank_1_to_20 (e.g. human1_rank_1_to_20, human2_rank_1_to_20,
human3_rank_1_to_20) — this matches what you get when you merge 3 filled-in
copies of the ranking sheet into one file.

Run:
    python stages/04b_validate_ranking.py --scored data/04_scored.jsonl --ranks data/human_ranks.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

from scipy.stats import spearmanr

# Matches human1_rank_1_to_20, human2_rank_1_to_20, human_rank_teammate1, etc.
RANK_COLUMN_PATTERN = re.compile(r"^human.*rank", re.IGNORECASE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="data/04_scored.jsonl")
    parser.add_argument("--ranks", default="data/human_ranks.csv",
                         help="CSV with lead_id + human rank columns (e.g. human1_rank_1_to_20)")
    args = parser.parse_args()

    scores_by_id = {}
    with open(args.scored, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                scores_by_id[rec["lead_id"]] = rec["score"]

    lead_ids, our_scores, avg_human_ranks = [], [], []
    with open(args.ranks, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rank_cols = [c for c in reader.fieldnames if c and RANK_COLUMN_PATTERN.match(c)]
        if not rank_cols:
            print(f"ERROR: no columns matching 'human*rank*' found. "
                  f"Columns present: {reader.fieldnames}")
            return
        print(f"Using rank columns: {rank_cols}")

        for row in reader:
            lid = row["lead_id"]
            if lid not in scores_by_id:
                continue
            teammate_ranks = [
                float(row[c]) for c in rank_cols
                if row.get(c) and row[c].strip()
            ]
            if len(teammate_ranks) < len(rank_cols):
                print(f"WARNING: {lid} has only {len(teammate_ranks)}/{len(rank_cols)} rankings filled in — skipping")
                continue
            lead_ids.append(lid)
            our_scores.append(scores_by_id[lid])
            avg_human_ranks.append(sum(teammate_ranks) / len(teammate_ranks))

    if len(lead_ids) < 5:
        print(f"ERROR: only {len(lead_ids)} leads had complete rankings — need more before this number means anything")
        return

    # Human ranks: 1 = best. Our scores: higher = better. Flip sign on
    # score so both are "lower = better" before correlating, OR just let
    # spearmanr handle it — correlation should come out strongly NEGATIVE
    # if we agree (high score <-> low/best human rank number). We report
    # the sign-corrected version so "close to +1" always means "we agree".
    rho, p_value = spearmanr(our_scores, avg_human_ranks)
    agreement = -rho  # flip so +1 = perfect agreement, -1 = totally backwards

    print(f"Compared {len(lead_ids)} leads")
    print(f"Spearman correlation (sign-corrected, +1 = perfect agreement): {agreement:.3f}")
    print(f"p-value: {p_value:.4f}")
    if agreement < 0.3:
        print(">> Weak/no agreement with human judgement — report this honestly, "
              "then consider re-weighting CHECK_WEIGHTS / CONVERSION_POINTS_PER_FINDING in 04_score.py")


if __name__ == "__main__":
    main()