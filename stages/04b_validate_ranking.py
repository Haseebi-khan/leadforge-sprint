"""
LEADFORGE SPRINT — Stage 4 validation, step 2
Compares your scorecard's ranking of the 20 sample leads against 3
teammates' independent human rankings, using Spearman correlation.

Expects a CSV with columns:
    lead_id, ... , human_rank_teammate1, human_rank_teammate2, human_rank_teammate3
(collect the 3 filled-in ranking sheets and merge their rank columns into one
file with these column names before running this.)

Run:
    python stages/04b_validate_ranking.py --scored data/04_scored.jsonl --ranks data/human_ranks.csv
"""
import argparse
import csv
import json
from pathlib import Path

from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", default="data/04_scored.jsonl")
    parser.add_argument("--ranks", default="data/human_ranks.csv",
                         help="CSV with lead_id + human_rank_teammate1/2/3 columns")
    args = parser.parse_args()

    scores_by_id = {}
    with open(args.scored, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                scores_by_id[rec["lead_id"]] = rec["score"]

    lead_ids, our_scores, avg_human_ranks = [], [], []
    with open(args.ranks, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid = row["lead_id"]
            if lid not in scores_by_id:
                continue
            teammate_ranks = [
                float(row[c]) for c in row
                if c.startswith("human_rank_teammate") and row[c].strip()
            ]
            if len(teammate_ranks) < 3:
                print(f"WARNING: {lid} has only {len(teammate_ranks)}/3 rankings filled in — skipping")
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
