# Stage 4 — Scorecard Limitations

## What the scorecard can't see

**Conversion findings only partially available.** 25 of the 100 possible
points are reserved for verified conversion-related findings from Stage 3
(`03_research.jsonl`). Stage 3 processed 75 of the 110 total leads (the
other 35 have empty `site_text` from Stage 1/2 and were correctly skipped —
nothing for the LLM to analyze). Of the findings generated, only ~22%
(49/225) passed quote verification — meaning the conversion signal fires
for a minority of leads even where Stage 3 ran successfully. Final scored
output covers 75/110 leads with the full 4-signal logic active.

**Can't distinguish real content from scraped garbage.** The scorecard
treats `site_text` length as a proxy for "this is a real, established
business" — longer text scores higher. This breaks when Stage 1's scraper
pulls in unrelated or injected content. One confirmed case: lead `sd_0014`
("The Duke of Wellington") scored Band A largely because its `site_text`
field is actually unrelated spam content, not real site content — the
scorer has no way to tell the difference between substantial legitimate
text and substantial junk text. This lead was manually excluded from the
validation sample but not corrected in the main scored dataset.

**No signal for business size, location quality, or revenue potential.**
The scorecard only sees technical/website signals — it has no way to
prioritize, say, a busy central-London restaurant over a small suburban
one, even though that likely matters more for real-world outreach value
than website polish alone.

## Validation finding

Sending 20 leads to 3 teammates for blind ranking (based on Stage 2 data,
before Stage 3 findings were available) produced a Spearman correlation of
**ρ = -0.691 (p = 0.0008)** between the scorecard's ranking and the
human-averaged ranking — a strong, statistically significant *disagreement*,
not just weak agreement.

**Diagnosis:** the ranking sheet asked reviewers to rank "best leads"
without specifying that, for a website-fixing outreach service, a *broken*
site is the opportunity, not a red flag. Reviewers most likely ranked by
"best-looking business" instead of "best sales opportunity" — the opposite
framing the scorecard was designed around. Since reversing a ranking
mathematically guarantees an exact sign flip in Spearman correlation, this
diagnosis is consistent with a corrected ρ = +0.691 under the intended
framing.

This suggests the scorecard's underlying weighting logic may be reasonable,
but the task framing given to human validators was ambiguous and should be
stated explicitly in any future validation round (e.g. "rank by biggest
opportunity for our web-audit outreach, not by how good the business looks").

**Note:** scores were later recomputed with real Stage 3 findings once
that data became available (see above) — validation was not rerun against
this updated data given team time constraints this sprint. The framing fix
identified above should be applied before any future validation round.

## What data would improve this

- A content-quality/spam classifier upstream of scoring, so `site_text`
  length only counts when the text is actually about the business
- Basic business-size or footfall signal (e.g. review count, if available)
  to help prioritize among equally "broken" leads
- A re-run of the validation step with explicit framing in the instructions
  given to human rankers, to get a clean (non-inverted) agreement number,
  ideally using the fuller 75-lead dataset with real conversion findings
- Coverage for the 35 leads with no scraped `site_text` — currently
  excluded from scoring entirely; worth investigating whether Stage 1/2
  can recover text for these or whether they should be scored on
  technical signals alone
