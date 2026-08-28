"""
LEADFORGE SPRINT — Stage 4 validation, step 1
Generates a CSV of 20 leads (name + a few key signals, NO score shown) for
3 teammates to independently rank 1 (best lead) to 20 (worst lead) by eye.

Run:
    python stages/04a_make_ranking_sheet.py --input data/04_scored.jsonl --n 20
Then send data/ranking_sheet.csv to 3 teammates and collect their 1-20 ranks
in extra columns (see stages/04b_validate_ranking.py for the expected shape).
"""
import argparse
import csv
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/04_scored.jsonl")
    parser.add_argument("--output", default="data/ranking_sheet.csv")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    leads = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                leads.append(json.loads(line))

    random.seed(args.seed)
    sample = random.sample(leads, min(args.n, len(leads)))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Deliberately do NOT include score/band — teammates must rank blind
        writer.writerow(["lead_id", "name", "domain", "has_contact_method",
                          "mobile_friendly", "loads_under_5s", "num_findings",
                          "human_rank_1_to_20"])
        for lead in sample:
            writer.writerow([
                lead.get("lead_id"), lead.get("name"), lead.get("domain"),
                lead.get("has_contact_method"), lead.get("mobile_friendly"),
                lead.get("loads_under_5s"), len(lead.get("findings", []) or []),
                "",  # blank for teammate to fill in
            ])

    print(f"Wrote {len(sample)} leads to {args.output}")
    print("Send this to 3 teammates, each fills their own copy of the "
          "human_rank_1_to_20 column (1 = best lead to contact, 20 = worst).")


if __name__ == "__main__":
    main()
