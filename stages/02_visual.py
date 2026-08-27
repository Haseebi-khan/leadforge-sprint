import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


# ============================================================
# FILE LOCATIONS
# ============================================================

INPUT_FILE = Path("data/01_leads.jsonl")
OUTPUT_FILE = Path("data/02_visual.jsonl")
SCREENSHOT_DIR = Path("screenshots")

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ STAGE 1 LEADS
# ============================================================

def load_leads():

    leads = []

    with INPUT_FILE.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                lead = json.loads(line)
                leads.append(lead)

            except json.JSONDecodeError as error:

                print(
                    f"Skipping invalid JSON on line "
                    f"{line_number}: {error}"
                )

    return leads


# ============================================================
# CREATE WEBSITE URL
# ============================================================

def build_url(domain):

    if not domain:
        return None

    domain = domain.strip()

    if domain.startswith("http://") or domain.startswith("https://"):
        return domain

    return "https://" + domain


# ============================================================
# CHECK PHONE NUMBER
# ============================================================

def check_phone_visible(page):

    try:

        body_text = page.locator("body").inner_text(timeout=5000)

        phone_pattern = re.compile(
            r"(?:\+?\d[\d\s().-]{7,}\d)"
        )

        matches = phone_pattern.findall(body_text)

        return len(matches) > 0

    except Exception:

        return False


# ============================================================
# CHECK CONTACT FORM
# ============================================================

def check_contact_form(page):

    try:

        forms = page.locator("form")

        for i in range(forms.count()):

            form = forms.nth(i)

            if not form.is_visible():
                continue

            fields = form.locator(
                "input, textarea, select"
            )

            if fields.count() > 0:
                return True

        return False

    except Exception:

        return False


# ============================================================
# CHECK MOBILE HORIZONTAL SCROLL
# ============================================================

def check_horizontal_scroll(page):

    try:

        return page.evaluate(
            """
            () => document.documentElement.scrollWidth >
                  document.documentElement.clientWidth
            """
        )

    except Exception:

        return False


# ============================================================
# CHECK TITLE
# ============================================================

def check_title(page):

    try:

        title = page.title()

        return bool(title.strip())

    except Exception:

        return False


# ============================================================
# CHECK META DESCRIPTION
# ============================================================

def check_meta_description(page):

    try:

        description = page.locator(
            'meta[name="description"]'
        ).get_attribute("content")

        return bool(
            description and description.strip()
        )

    except Exception:

        return False


# ============================================================
# PROCESS ONE LEAD
# ============================================================

def process_lead(browser, lead):

    lead_id = lead.get("lead_id", "unknown")

    domain = lead.get("domain")

    url = build_url(domain)

    print()
    print("=" * 60)
    print(f"Lead: {lead_id}")
    print(f"Business: {lead.get('name')}")
    print(f"Website: {url}")

    # Copy every Stage 1 field
    result = dict(lead)

    # Add Stage 2 fields
    result.update({
        "website_url": url,
        "phone_visible": False,
        "contact_form": False,
        "load_time_seconds": None,
        "loads_under_5_seconds": False,
        "horizontal_scroll_mobile": False,
        "title_present": False,
        "meta_description_present": False,
        "desktop_screenshot": None,
        "mobile_screenshot": None,
        "status": "success",
        "error": None
    })

    if not url:

        result["status"] = "error"
        result["error"] = "No domain provided"

        return result

    # ========================================================
    # DESKTOP
    # ========================================================

    desktop = browser.new_page(
        viewport={
            "width": 1440,
            "height": 900
        }
    )

    try:

        start_time = time.perf_counter()

        desktop.goto(
            url,
            wait_until="load",
            timeout=10000
        )

        load_time = time.perf_counter() - start_time

        result["load_time_seconds"] = round(
            load_time,
            3
        )

        result["loads_under_5_seconds"] = (
            load_time < 5
        )

        # Four website checks
        result["phone_visible"] = check_phone_visible(
            desktop
        )

        result["contact_form"] = check_contact_form(
            desktop
        )

        result["title_present"] = check_title(
            desktop
        )

        result["meta_description_present"] = (
            check_meta_description(desktop)
        )

        # Desktop screenshot
        desktop_path = (
            SCREENSHOT_DIR /
            f"{lead_id}_desktop.png"
        )

        desktop.screenshot(
            path=str(desktop_path),
            full_page=True
        )

        result["desktop_screenshot"] = str(
            desktop_path
        )

        print(
            f"Load time: {load_time:.2f}s"
        )

        print(
            f"Phone visible: "
            f"{result['phone_visible']}"
        )

        print(
            f"Contact form: "
            f"{result['contact_form']}"
        )

        print(
            f"Title present: "
            f"{result['title_present']}"
        )

        print(
            f"Meta description: "
            f"{result['meta_description_present']}"
        )

        print(
            f"Desktop screenshot saved."
        )

    except PlaywrightTimeoutError:

        result["status"] = "error"
        result["error"] = "Desktop website load timed out"

        print("Desktop load timed out.")

    except Exception as error:

        result["status"] = "error"
        result["error"] = str(error)

        print(f"Desktop error: {error}")

    finally:

        desktop.close()

    # ========================================================
    # MOBILE
    # ========================================================

    mobile = browser.new_page(
        viewport={
            "width": 390,
            "height": 844
        }
    )

    try:

        mobile.goto(
            url,
            wait_until="domcontentloaded",
            timeout=10000
        )

        result["horizontal_scroll_mobile"] = (
            check_horizontal_scroll(mobile)
        )

        mobile_path = (
            SCREENSHOT_DIR /
            f"{lead_id}_mobile.png"
        )

        mobile.screenshot(
            path=str(mobile_path),
            full_page=True
        )

        result["mobile_screenshot"] = str(
            mobile_path
        )

        print(
            f"Horizontal mobile scroll: "
            f"{result['horizontal_scroll_mobile']}"
        )

        print("Mobile screenshot saved.")

    except PlaywrightTimeoutError:

        print("Mobile load timed out.")

    except Exception as error:

        print(f"Mobile error: {error}")

    finally:

        mobile.close()

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    if not INPUT_FILE.exists():

        print(
            f"ERROR: {INPUT_FILE} was not found."
        )

        print(
            "Run this script from the repository root."
        )

        return

    leads = load_leads()

    print(
        f"Found {len(leads)} leads."
    )

    if not leads:

        print(
            "01_leads.jsonl is empty."
        )

        return

    results = []

    with sync_playwright() as playwright:

        browser = playwright.chromium.launch(
            headless=False
        )

        for number, lead in enumerate(
            leads,
            start=1
        ):

            print()
            print(
                f"PROCESSING {number}/{len(leads)}"
            )

            result = process_lead(
                browser,
                lead
            )

            results.append(result)

        browser.close()

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )

    print()
    print("=" * 60)
    print("STAGE 2 COMPLETE")
    print("=" * 60)

    print(
        f"Output file: {OUTPUT_FILE}"
    )

    print(
        f"Screenshots: {SCREENSHOT_DIR}"
    )

    print(
        f"Leads processed: {len(results)}"
    )


if __name__ == "__main__":
    main()