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

# Leads known to have bad/spam site_text from Stage 1 scraping — exclude
# from the ranking sheet so they don't confuse human rankers or pollute
# the correlation. Add to this list as more are found.
KNOWN_BAD_LEADS = {"sd_0014"}


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
                lead = json.loads(line)
                if lead.get("lead_id") not in KNOWN_BAD_LEADS:
                    leads.append(lead)

    random.seed(args.seed)
    sample = random.sample(leads, min(args.n, len(leads)))

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Deliberately do NOT include score/band — teammates must rank blind.
        # Column names match the REAL Stage 2 (data/02_visual.jsonl) fields.
        writer.writerow(["lead_id", "name", "domain", "phone_visible",
                          "contact_form", "horizontal_scroll_mobile",
                          "loads_under_5_seconds", "meta_description_present",
                          "site_status", "human_rank_1_to_20"])
        for lead in sample:
            writer.writerow([
                lead.get("lead_id"), lead.get("name"), lead.get("domain"),
                lead.get("phone_visible"), lead.get("contact_form"),
                lead.get("horizontal_scroll_mobile"),
                lead.get("loads_under_5_seconds"),
                lead.get("meta_description_present"),
                lead.get("status"),
                "",  # blank for teammate to fill in
            ])

    print(f"Wrote {len(sample)} leads to {args.output} (excluded {len(KNOWN_BAD_LEADS)} known-bad leads)")
    print("Send this to 3 teammates, each fills their own copy of the "
          "human_rank_1_to_20 column (1 = best lead to contact, 20 = worst).")


if __name__ == "__main__":
    main()