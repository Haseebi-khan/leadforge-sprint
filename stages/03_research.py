"""
Reads a JSONL file (Stage 2 output, or the shared sample file), sends each
lead's site_text + visual checks to an LLM, extracts exactly 3 structured
findings, verifies every quote is a literal substring of site_text, and
writes data/03_research.jsonl.

One LLM call per lead. Gemini first, Groq as fallback if Gemini fails or
GEMINI_API_KEY isn't set.

Usage:
    python stages/03_research.py --input data/sample_10.jsonl --limit 5
    python stages/03_research.py --input data/02_visual.jsonl --output data/03_research.jsonl
"""

import argparse
import json
import os
import re
import sys
from typing import List, Literal

from pydantic import BaseModel, ValidationError

# "Pydantic: is a popular Python library used for data validation, parsing, and serialization using standard Python type hints."

from dotenv import load_dotenv

load_dotenv()

CATEGORIES = ["design", "content", "conversion", "trust"]

# errors from stage2 are naming convention usally like 
# stages 1 visual_ok  stage 2 status ("success"/"error") + error
# has_contact_method to split into phone_visible and contact_form
# loads_under_5s to loads_under_5_seconds
# mobile_friendly to horizontal_scroll_mobile
# has_meta_description to meta_description_present
# screenshot_desktop/screenshot_mobile  to desktop_screenshot/mobile_screenshot

# and also new feilds  are  website_url, title_present, load_time_seconds

# i am going to change below or write the scrpit which will take care of the below comment

"""
Fixed the field-name mismatch — 03_research.py now reads Stage 2's real fields (phone_visible, contact_form, loads_under_5_seconds, horizontal_scroll_mobile, title_present, meta_description_present, status, error) instead of the never-matching schema names it was written against before Stage 2's actual code existed.
Updated lead_schema.json to match Intern 5's real output, including website_url, load_time_seconds, and the status/error pair Stage 2 uses instead of a single visual_ok boolean.
Added the empty-site_text pre-check — a lead with under 20 characters of usable text is now skipped before the LLM call, logged separately as "skipped (no site_text)" rather than counted as a failure.
"""

SYSTEM_PROMPT = """You are auditing a small local business's website for a lead-generation \
research pipeline. You will be given the extracted site text and some automated checks. \
Produce EXACTLY 3 findings.

Rules:
- Each finding must have: claim, quote, category.
- "quote" MUST be copied verbatim, character-for-character, from the site text provided. \
Do not paraphrase, shorten, or fix typos in the quote. If you cannot find a supporting quote \
in the text, use the automated check results instead and set quote to an empty string.
- "category" must be exactly one of: design, content, conversion, trust.
- "claim" must be one specific, concrete observation - never a generic statement like \
"the website could be improved."
- Never invent a fact that isn't supported by the site text or the check results.

Return ONLY a JSON object of the form:
{"findings": [{"claim": "...", "quote": "...", "category": "..."}, {...}, {...}]}
No prose, no markdown fences, no extra keys.
"""


class Finding(BaseModel):
    claim: str
    quote: str
    category: Literal["design", "content", "conversion", "trust"]


class FindingsResponse(BaseModel):
    findings: List[Finding]

    def validate_length(self):
        if len(self.findings) != 3:
            raise ValueError(f"expected exactly 3 findings, got {len(self.findings)}")


# Minimum usable length for site_text. Below this there's nothing to ground
# a quote in - calling the LLM anyway just wastes quota on a lead that was
# never going to produce a real finding.
MIN_SITE_TEXT_CHARS = 20

# Stage 2's actual field names (confirmed against Intern 5's real script and
# real 02_visual.jsonl output - these do NOT match the field names originally
# drafted into contracts/lead_schema.json, which has been corrected to match).
STAGE_2_CHECK_FIELDS = (
    "phone_visible",            # visible phone number found in page text
    "contact_form",             # a visible <form> with input/textarea/select fields
    "loads_under_5_seconds",
    "horizontal_scroll_mobile", # True here means NOT mobile friendly
    "title_present",
    "meta_description_present",
    "status",                   # "success" or "error" - Stage 2's own fetch outcome
    "error",                    # error message if status == "error", else None
)


def build_user_prompt(lead: dict) -> str:
    checks = {k: lead.get(k) for k in STAGE_2_CHECK_FIELDS if k in lead}
    return (
        f"Business: {lead.get('name')}\n"
        f"Category: {lead.get('category')}\n"
        f"Automated checks: {json.dumps(checks)}\n\n"
        f"Site text:\n{lead.get('site_text', '')}"
    )


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"```$", "", text)
    return text.strip()


def call_gemini(user_prompt: str) -> dict:
    import google.generativeai as genai

    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"},
    )
    resp = model.generate_content(user_prompt)
    return json.loads(_strip_code_fence(resp.text))


def call_groq(user_prompt: str) -> dict:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(_strip_code_fence(resp.choices[0].message.content))


def get_findings(lead: dict) -> List[Finding]:
    """One LLM call. Tries Gemini, falls back to Groq. Raises on total failure."""
    user_prompt = build_user_prompt(lead)
    raw = None
    last_err = None

    if os.environ.get("GEMINI_API_KEY"):
        try:
            raw = call_gemini(user_prompt)
        except Exception as e:  # noqa: BLE001
            last_err = e

    if raw is None and os.environ.get("GROQ_API_KEY"):
        try:
            raw = call_groq(user_prompt)
        except Exception as e:  # noqa: BLE001
            last_err = e

    if raw is None:
        raise RuntimeError(f"no LLM backend succeeded for lead {lead.get('lead_id')}: {last_err}")

    parsed = FindingsResponse.model_validate(raw)
    parsed.validate_length()
    return parsed.findings


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verify_quote(quote: str, site_text: str) -> bool | None:
    if not quote:
        return False
    return normalize(quote) in normalize(site_text)


class NoSiteTextError(ValueError):
    """Raised when a lead has no usable site_text - never worth an LLM call."""


def process_lead(lead: dict) -> dict:
    site_text = lead.get("site_text", "") or ""
    if len(site_text.strip()) < MIN_SITE_TEXT_CHARS:
        raise NoSiteTextError(
            f"site_text is empty/too short ({len(site_text.strip())} chars) - skipped before LLM call"
        )

    findings = get_findings(lead)

    lead_out = dict(lead)  # copy every field through untouched
    lead_out["findings"] = [
        {
            "claim": f.claim,
            "quote": f.quote,
            "category": f.category,
            "quote_verified": verify_quote(f.quote, site_text),
        }
        for f in findings
    ]
    return lead_out


"""
    module for parsing command-line arguments. It lets you define the arguments your program accepts, automatically generates help messages, and validates user input.

    parser = argparse.ArgumentParser(description="Greet a user")

    parser.add_argument("name", help="Name of the user")
    parser.add_argument("--age", type=int, help="User's age")

    args = parser.parse_args()

    print(f"Hello, {args.name}!")
    if args.age:
    print(f"You are {args.age} years old.")
"""

def main():
    
    ap = argparse.ArgumentParser(description="Stage 3 - Research: extract & verify findings | Haseeb Khan")
    ap.add_argument("--input", default="data/02_visual.jsonl", help="Input JSONL from Stage 2 (Ishmal)")
    ap.add_argument("--output", default="data/03_research.jsonl", help="Output JSONL for Stage 4 (Azlan)")
    
    # use --limit=N or --limit N but the defualt is three for now       (comment for umer) 
    
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N (which is 3) leads (for testing)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    if args.limit:
        lines = lines[: args.limit]

    written = 0
    total_findings = 0
    verified_findings = 0
    skipped_no_text = []
    failures = []

    with open(args.output, "w", encoding="utf-8") as out:
        for lead in lines:
            lead_id = lead.get("lead_id", "?")
            try:
                result = process_lead(lead)
            except NoSiteTextError as e:
                print(f"  [skip - no text] {lead_id}: {e}", file=sys.stderr)
                skipped_no_text.append(lead_id)
                continue
            except (ValidationError, ValueError, RuntimeError) as e:
                print(f"  [skip - failed] {lead_id}: {e}", file=sys.stderr)
                failures.append(lead_id)
                continue

            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
            for fnd in result["findings"]:
                total_findings += 1
                if fnd["quote_verified"]:
                    verified_findings += 1

    print(f"read: {len(lines)}  written: {written}  "
          f"skipped (no site_text): {len(skipped_no_text)}  failed (LLM/validation): {len(failures)}")
    if skipped_no_text:
        print(f"skipped lead_ids: {skipped_no_text}")
    if failures:
        print(f"failed lead_ids: {failures}")
    if total_findings:
        rate = verified_findings / total_findings * 100
        print(f"quote verification pass rate: {verified_findings}/{total_findings} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
    

    
    
# Outputs

    
    
    
# Outputs

    
    
    
# Outputs

    
    
    
# Outputs

    
    
    
# Outputs



"""
(leadforge) noneo@noneo-Precision-5530:~/Codes/Sprin/leadforge-sprint$ python stages/03_research.py --input data/02_visual.jsonl
/home/noneo/Codes/Sprin/leadforge-sprint/stages/03_research.py:123: FutureWarning: 

All support for the `google.generativeai` package has ended. It will no longer be receiving 
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
  [skip - no text] sd_0002: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0006: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0007: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0010: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0011: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0012: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0015: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0016: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0018: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0025: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0032: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0036: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0039: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0042: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0044: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0045: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0046: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0052: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0053: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0061: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0067: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0068: no LLM backend succeeded for lead sd_0068: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 6113, Requested 1961. Please try again in 554.999999ms. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0069: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0075: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0078: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0080: no LLM backend succeeded for lead sd_0080: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per minute (TPM): Limit 8000, Used 6251, Requested 1847. Please try again in 735ms. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0081: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0082: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0084: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0086: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0090: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0096: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0104: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0105: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0106: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0107: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0109: site_text is empty/too short (0 chars) - skipped before LLM call
read: 110  written: 73  skipped (no site_text): 35  failed (LLM/validation): 2
skipped lead_ids: ['sd_0002', 'sd_0006', 'sd_0007', 'sd_0010', 'sd_0011', 'sd_0012', 'sd_0015', 'sd_0016', 'sd_0018', 'sd_0025', 'sd_0032', 'sd_0036', 'sd_0039', 'sd_0042', 'sd_0044', 'sd_0045', 'sd_0046', 'sd_0052', 'sd_0053', 'sd_0061', 'sd_0067', 'sd_0069', 'sd_0075', 'sd_0078', 'sd_0081', 'sd_0082', 'sd_0084', 'sd_0086', 'sd_0090', 'sd_0096', 'sd_0104', 'sd_0105', 'sd_0106', 'sd_0107', 'sd_0109']
failed lead_ids: ['sd_0068', 'sd_0080']
quote verification pass rate: 55/219 (25.1%)

"""



"""
02_visual.jsonl Output


(leadforge) noneo@noneo-Precision-5530:~/Codes/Sprin/leadforge-sprint$ python stages/03_research.py --input data/02_visual.jsonl
/home/noneo/Codes/Sprin/leadforge-sprint/stages/03_research.py:123: FutureWarning: 

All support for the `google.generativeai` package has ended. It will no longer be receiving 
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
  [skip - no text] sd_0002: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0006: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0007: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0010: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0011: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0012: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0015: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0016: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0018: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0025: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0032: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0036: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0039: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0042: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0044: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0045: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0046: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0052: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0053: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0061: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0067: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0069: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0075: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0078: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0081: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0082: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0084: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0086: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0090: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0096: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0104: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0105: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0106: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0107: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0109: site_text is empty/too short (0 chars) - skipped before LLM call
read: 110  written: 75  skipped (no site_text): 35  failed (LLM/validation): 0
skipped lead_ids: ['sd_0002', 'sd_0006', 'sd_0007', 'sd_0010', 'sd_0011', 'sd_0012', 'sd_0015', 'sd_0016', 'sd_0018', 'sd_0025', 'sd_0032', 'sd_0036', 'sd_0039', 'sd_0042', 'sd_0044', 'sd_0045', 'sd_0046', 'sd_0052', 'sd_0053', 'sd_0061', 'sd_0067', 'sd_0069', 'sd_0075', 'sd_0078', 'sd_0081', 'sd_0082', 'sd_0084', 'sd_0086', 'sd_0090', 'sd_0096', 'sd_0104', 'sd_0105', 'sd_0106', 'sd_0107', 'sd_0109']
quote verification pass rate: 57/225 (25.3%)
(leadforge) noneo@noneo-Precision-5530:~/Codes/Sprin/leadforge-sprint$ 
"""



"""
(leadforge) noneo@noneo-Precision-5530:~/Codes/Sprin/leadforge-sprint$ python stages/03_research.py --input data/02_visual.jsonl
  [skip - no text] sd_0002: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0006: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0007: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0010: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0011: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0012: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0015: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0016: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0018: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0022: no LLM backend succeeded for lead sd_0022: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199400, Requested 1421. Please try again in 5m54.672s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0024: no LLM backend succeeded for lead sd_0024: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199998, Requested 443. Please try again in 3m10.512s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0025: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0026: no LLM backend succeeded for lead sd_0026: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199995, Requested 521. Please try again in 3m42.911999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0027: no LLM backend succeeded for lead sd_0027: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199993, Requested 655. Please try again in 4m39.936s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0028: no LLM backend succeeded for lead sd_0028: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199991, Requested 1081. Please try again in 7m43.104s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0029: no LLM backend succeeded for lead sd_0029: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199988, Requested 576. Please try again in 4m3.648s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0030: no LLM backend succeeded for lead sd_0030: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199986, Requested 923. Please try again in 6m32.688s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0031: no LLM backend succeeded for lead sd_0031: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199984, Requested 824. Please try again in 5m49.056s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0032: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0033: no LLM backend succeeded for lead sd_0033: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199981, Requested 586. Please try again in 4m4.944s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0034: no LLM backend succeeded for lead sd_0034: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199979, Requested 1054. Please try again in 7m26.256s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0035: no LLM backend succeeded for lead sd_0035: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199976, Requested 402. Please try again in 2m43.296s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0036: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0037: no LLM backend succeeded for lead sd_0037: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199974, Requested 729. Please try again in 5m3.696s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0038: no LLM backend succeeded for lead sd_0038: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199971, Requested 1442. Please try again in 10m10.416s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0039: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0040: no LLM backend succeeded for lead sd_0040: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199969, Requested 911. Please try again in 6m20.16s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0041: no LLM backend succeeded for lead sd_0041: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199966, Requested 1264. Please try again in 8m51.36s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0042: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0043: no LLM backend succeeded for lead sd_0043: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199964, Requested 622. Please try again in 4m13.152s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0044: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0045: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0046: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0047: no LLM backend succeeded for lead sd_0047: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199961, Requested 1549. Please try again in 10m52.32s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0048: no LLM backend succeeded for lead sd_0048: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199959, Requested 1229. Please try again in 8m33.216s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0049: no LLM backend succeeded for lead sd_0049: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199956, Requested 568. Please try again in 3m46.368s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0050: no LLM backend succeeded for lead sd_0050: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199954, Requested 619. Please try again in 4m7.536s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0051: no LLM backend succeeded for lead sd_0051: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199952, Requested 1251. Please try again in 8m39.696s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0052: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0053: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0054: no LLM backend succeeded for lead sd_0054: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199949, Requested 1339. Please try again in 9m16.416s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0055: no LLM backend succeeded for lead sd_0055: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199949, Requested 2321. Please try again in 16m20.64s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0056: no LLM backend succeeded for lead sd_0056: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199949, Requested 1477. Please try again in 10m16.032s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0057: no LLM backend succeeded for lead sd_0057: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199946, Requested 972. Please try again in 6m36.576s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0058: no LLM backend succeeded for lead sd_0058: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199943, Requested 1327. Please try again in 9m8.64s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0059: no LLM backend succeeded for lead sd_0059: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199941, Requested 1242. Please try again in 8m31.056s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0060: no LLM backend succeeded for lead sd_0060: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199939, Requested 718. Please try again in 4m43.824s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0061: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0062: no LLM backend succeeded for lead sd_0062: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199936, Requested 1189. Please try again in 8m6s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0063: no LLM backend succeeded for lead sd_0063: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199934, Requested 1228. Please try again in 8m21.984s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0064: no LLM backend succeeded for lead sd_0064: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199931, Requested 519. Please try again in 3m14.399999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0065: no LLM backend succeeded for lead sd_0065: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199929, Requested 1269. Please try again in 8m37.535999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0066: no LLM backend succeeded for lead sd_0066: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199926, Requested 1096. Please try again in 7m21.504s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0067: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0068: no LLM backend succeeded for lead sd_0068: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199924, Requested 928. Please try again in 6m8.063999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0069: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0070: no LLM backend succeeded for lead sd_0070: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199921, Requested 1250. Please try again in 8m25.872s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0071: no LLM backend succeeded for lead sd_0071: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199919, Requested 589. Please try again in 3m39.456s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0072: no LLM backend succeeded for lead sd_0072: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199916, Requested 795. Please try again in 5m7.152s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0073: no LLM backend succeeded for lead sd_0073: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199914, Requested 1352. Please try again in 9m6.911999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0074: no LLM backend succeeded for lead sd_0074: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199911, Requested 1318. Please try again in 8m50.928s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0075: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0076: no LLM backend succeeded for lead sd_0076: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199909, Requested 1262. Please try again in 8m25.872s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0077: no LLM backend succeeded for lead sd_0077: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199906, Requested 867. Please try again in 5m33.936s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0078: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0079: no LLM backend succeeded for lead sd_0079: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199904, Requested 1303. Please try again in 8m41.424s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0080: no LLM backend succeeded for lead sd_0080: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199901, Requested 636. Please try again in 3m51.983999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0081: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0082: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0083: no LLM backend succeeded for lead sd_0083: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199899, Requested 629. Please try again in 3m48.096s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0084: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0085: no LLM backend succeeded for lead sd_0085: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199897, Requested 1883. Please try again in 12m48.96s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0086: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0087: no LLM backend succeeded for lead sd_0087: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199894, Requested 487. Please try again in 2m44.592s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0088: no LLM backend succeeded for lead sd_0088: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199891, Requested 724. Please try again in 4m25.68s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0089: no LLM backend succeeded for lead sd_0089: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199888, Requested 807. Please try again in 5m0.24s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0090: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0091: no LLM backend succeeded for lead sd_0091: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199886, Requested 1184. Please try again in 7m42.239999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0092: no LLM backend succeeded for lead sd_0092: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199884, Requested 2442. Please try again in 16m44.832s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0093: no LLM backend succeeded for lead sd_0093: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199881, Requested 666. Please try again in 3m56.304s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0094: no LLM backend succeeded for lead sd_0094: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199879, Requested 679. Please try again in 4m1.055999999s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0095: no LLM backend succeeded for lead sd_0095: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199876, Requested 544. Please try again in 3m1.44s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0096: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0097: no LLM backend succeeded for lead sd_0097: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199874, Requested 1188. Please try again in 7m38.784s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0098: no LLM backend succeeded for lead sd_0098: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199872, Requested 547. Please try again in 3m1.008s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0099: no LLM backend succeeded for lead sd_0099: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199869, Requested 1143. Please try again in 7m17.184s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0100: no LLM backend succeeded for lead sd_0100: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199867, Requested 523. Please try again in 2m48.48s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0101: no LLM backend succeeded for lead sd_0101: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199864, Requested 1244. Please try again in 7m58.656s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0102: no LLM backend succeeded for lead sd_0102: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199862, Requested 1489. Please try again in 9m43.632s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - failed] sd_0103: no LLM backend succeeded for lead sd_0103: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199859, Requested 1009. Please try again in 6m14.976s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0104: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0105: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0106: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - no text] sd_0107: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0108: no LLM backend succeeded for lead sd_0108: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199857, Requested 975. Please try again in 5m59.424s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
  [skip - no text] sd_0109: site_text is empty/too short (0 chars) - skipped before LLM call
  [skip - failed] sd_0110: no LLM backend succeeded for lead sd_0110: Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` in organization `org_01m13qysesegxs4s7504kd8kx2` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199854, Requested 1173. Please try again in 7m23.664s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
read: 110  written: 13  skipped (no site_text): 35  failed (LLM/validation): 62
skipped lead_ids: ['sd_0002', 'sd_0006', 'sd_0007', 'sd_0010', 'sd_0011', 'sd_0012', 'sd_0015', 'sd_0016', 'sd_0018', 'sd_0025', 'sd_0032', 'sd_0036', 'sd_0039', 'sd_0042', 'sd_0044', 'sd_0045', 'sd_0046', 'sd_0052', 'sd_0053', 'sd_0061', 'sd_0067', 'sd_0069', 'sd_0075', 'sd_0078', 'sd_0081', 'sd_0082', 'sd_0084', 'sd_0086', 'sd_0090', 'sd_0096', 'sd_0104', 'sd_0105', 'sd_0106', 'sd_0107', 'sd_0109']
failed lead_ids: ['sd_0022', 'sd_0024', 'sd_0026', 'sd_0027', 'sd_0028', 'sd_0029', 'sd_0030', 'sd_0031', 'sd_0033', 'sd_0034', 'sd_0035', 'sd_0037', 'sd_0038', 'sd_0040', 'sd_0041', 'sd_0043', 'sd_0047', 'sd_0048', 'sd_0049', 'sd_0050', 'sd_0051', 'sd_0054', 'sd_0055', 'sd_0056', 'sd_0057', 'sd_0058', 'sd_0059', 'sd_0060', 'sd_0062', 'sd_0063', 'sd_0064', 'sd_0065', 'sd_0066', 'sd_0068', 'sd_0070', 'sd_0071', 'sd_0072', 'sd_0073', 'sd_0074', 'sd_0076', 'sd_0077', 'sd_0079', 'sd_0080', 'sd_0083', 'sd_0085', 'sd_0087', 'sd_0088', 'sd_0089', 'sd_0091', 'sd_0092', 'sd_0093', 'sd_0094', 'sd_0095', 'sd_0097', 'sd_0098', 'sd_0099', 'sd_0100', 'sd_0101', 'sd_0102', 'sd_0103', 'sd_0108', 'sd_0110']
quote verification pass rate: 11/39 (28.2%)
"""
