"""
LEADFORGE SPRINT — Stage 4: Score the leads
Owner: Azlan

Reads:  data/03_research.jsonl   (written by Haseeb — Stage 3)
Writes: data/04_scored.jsonl

Every input field is copied through unchanged; this script only ADDS:
    score          -> int 0-100
    band           -> "A" | "B" | "C" | "D"
    score_reasons  -> list of the top 3 signals that drove the score

Run:
    python stages/04_score.py --input data/03_research.jsonl --output data/04_scored.jsonl
    python stages/04_score.py --limit 5      # test on first 5 rows only
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Scoring config — the weights are the thing you'll tune once on Wednesday
# after seeing the Spearman correlation against human ranking.
# ---------------------------------------------------------------------------

# Set this to whatever category the team is targeting this sprint
# (e.g. "real_estate_agency"). Leave as None to disable the category bonus.
TARGET_CATEGORY = "restaurant"

# Site fully failed to load — this is the strongest possible signal, worth
# more than any single check below. Comes from Stage 2's `status` field.
SITE_DOWN_POINTS = 15

# Each failed visual/technical check is a sales hook — a site with more
# broken things is a BETTER lead for an agency that fixes websites, so
# failures ADD to the score rather than subtract from it.
# NOTE: these are the REAL Stage 2 (data/02_visual.jsonl) field names,
# confirmed against an actual sample entry on 2026-08-28.
CHECK_WEIGHTS = {
    "no_contact_method":  ("no visible phone or contact form", 12),
    "not_mobile_friendly": ("mobile horizontal-scroll issue", 10),
    "slow_load":          ("slow page load (5s+)", 10),
    "no_meta_description": ("missing meta description", 8),
}
MAX_CHECK_SCORE = sum(w for _, w in CHECK_WEIGHTS.values())  # 40

CONVERSION_POINTS_PER_FINDING = 12
MAX_CONVERSION_SCORE = 25

CATEGORY_MATCH_BONUS = 15

MAX_TEXT_SCORE = 20
TEXT_LEN_SUBSTANTIAL = 500   # chars -> full text score
TEXT_LEN_MINIMAL = 150       # chars -> partial text score


def score_lead(lead: dict) -> dict:
    """Return {score, band, score_reasons} for one lead record."""
    contributions = []  # list of (label, points)

    # --- Signal 0: site failed to load entirely (Stage 2 `status`) --------
    # Strongest possible signal — worse than any individual check failing.
    if lead.get("status") == "error":
        contributions.append((f"site failed to load ({lead.get('error', 'unknown error')})", SITE_DOWN_POINTS))

    # --- Signal 1: failed visual/technical checks (up to 40 pts) ----------
    # Derive our booleans from Stage 2's real field names.
    derived_checks = {
        "no_contact_method": not (lead.get("phone_visible") or lead.get("contact_form")),
        "not_mobile_friendly": bool(lead.get("horizontal_scroll_mobile")),  # True = broken
        "slow_load": lead.get("loads_under_5_seconds") is False,
        "no_meta_description": lead.get("meta_description_present") is False,
    }
    for field, (label, weight) in CHECK_WEIGHTS.items():
        # Only score a check if the underlying Stage 2 data is actually present,
        # so a lead never gets penalized just because Stage 2 didn't run on it.
        if field == "no_contact_method" and ("phone_visible" not in lead and "contact_form" not in lead):
            continue
        if field == "not_mobile_friendly" and "horizontal_scroll_mobile" not in lead:
            continue
        if field == "slow_load" and "loads_under_5_seconds" not in lead:
            continue
        if field == "no_meta_description" and "meta_description_present" not in lead:
            continue
        if derived_checks[field]:
            contributions.append((label, weight))

    # --- Signal 2: verified conversion-related findings (up to 25 pts) ----
    findings = lead.get("findings", []) or []
    conversion_findings = [
        f for f in findings
        if f.get("category") == "conversion" and f.get("quote_verified")
    ]
    if conversion_findings:
        pts = min(len(conversion_findings) * CONVERSION_POINTS_PER_FINDING, MAX_CONVERSION_SCORE)
        contributions.append((f"{len(conversion_findings)} verified conversion issue(s)", pts))

    # --- Signal 3: site text length as a proxy for a "real", active business
    #     (thin/near-empty site content is a data-quality red flag, not a lead) (up to 20 pts)
    text_len = len(lead.get("site_text", "") or "")
    if text_len >= TEXT_LEN_SUBSTANTIAL:
        contributions.append(("substantial site content", MAX_TEXT_SCORE))
    elif text_len >= TEXT_LEN_MINIMAL:
        contributions.append(("some site content", MAX_TEXT_SCORE // 2))
    else:
        contributions.append(("thin site content (low confidence)", 0))

    # --- Signal 4: category match bonus (up to 15 pts) ---------------------
    if TARGET_CATEGORY and lead.get("category") == TARGET_CATEGORY:
        contributions.append(("target category match", CATEGORY_MATCH_BONUS))

    raw_score = sum(pts for _, pts in contributions)
    score = max(0, min(100, raw_score))

    if score >= 75:
        band = "A"
    elif score >= 50:
        band = "B"
    elif score >= 25:
        band = "C"
    else:
        band = "D"

    # Top 3 reasons by point contribution, dropping zero-point ones first
    ranked = sorted([c for c in contributions if c[1] > 0], key=lambda c: -c[1])
    score_reasons = [label for label, _ in ranked[:3]]
    if not score_reasons:
        score_reasons = ["no strong signals detected"]

    return {"score": score, "band": band, "score_reasons": score_reasons}


def process(input_path: Path, output_path: Path, limit: int | None) -> None:
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        print("(This is expected if Haseeb hasn't pushed data/03_research.jsonl "
              "or data/sample_10.jsonl yet — test on a mock file in the meantime.)",
              file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    read_count = 0
    written_count = 0
    band_counts = {"A": 0, "B": 0, "C": 0, "D": 0}

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            read_count += 1
            if limit and read_count > limit:
                break

            lead = json.loads(line)
            result = score_lead(lead)

            # Copy every input field through untouched, then add ours —
            # this is the one rule that protects everyone downstream.
            out_record = {**lead, **result}
            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            written_count += 1
            band_counts[result["band"]] += 1

    print(f"Read {read_count} leads, wrote {written_count} scored leads -> {output_path}")
    print(f"Band distribution: {band_counts}")


def main():
    parser = argparse.ArgumentParser(description="LeadForge Stage 4 — Score the leads")
    parser.add_argument("--input", default="data/03_research.jsonl",
                         help="Path to Stage 3 output (default: data/03_research.jsonl)")
    parser.add_argument("--output", default="data/04_scored.jsonl",
                         help="Path to write scored leads (default: data/04_scored.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N leads (for quick testing)")
    args = parser.parse_args()

    process(Path(args.input), Path(args.output), args.limit)


if __name__ == "__main__":
    main()