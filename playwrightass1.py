# search_scorecard_fixed.py
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from urllib.parse import unquote, urlparse, parse_qs
import time
import sys

QUERY = "sa vs ind final scorecard"

# try multiple selectors for the search box / results
SEARCH_BOX_SELECTORS = [
    "input[name='q']",      # Google
    "#sb_form_q",           # Bing common id
    "input[type='search']",
    "input[aria-label='Search']",
    "input[role='search']"
]

RESULT_LINK_SELECTORS = [
    "li.b_algo h2 a",       # Bing organic
    ".b_algo a",            # fallback Bing
    "a[href*='cricbuzz.com']",
    "a[href*='espncricinfo.com']"
]

def extract_bing_redirect(href: str) -> str:
    """
    If href is a Bing redirect like /ck/a?...&u=<encodedURL>&..., extract and return decoded target.
    Otherwise return original href.
    """
    if not href:
        return href
    try:
        parsed = urlparse(href)
        qs = parse_qs(parsed.query)
        # look for 'u' param (common for Bing ck redirect)
        if 'u' in qs and qs['u']:
            return unquote(qs['u'][0])
        # sometimes href contains 'u=' as substring
        if 'u=' in href:
            # crude fallback: find u= and split until & or end
            start = href.find('u=') + 2
            rest = href[start:]
            end = rest.find('&')
            if end != -1:
                val = rest[:end]
            else:
                val = rest
            return unquote(val)
    except Exception:
        pass
    return href

def main(headless: bool = False):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, slow_mo=50)
        context = browser.new_context(
            viewport={"width": 1250, "height": 778},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"),
            locale="en-US",
            accept_downloads=True
        )
        page = context.new_page()

        try:
            search_url = "https://www.bing.com/search?q=" + QUERY.replace(" ", "+")
            print("Going to", search_url)
            page.goto(search_url, wait_until="domcontentloaded", timeout=60000)

            # Accept cookie/consent if present (try a few texts)
            try:
                consent_selectors = [
                    'button:has-text("I agree")',
                    'button:has-text("I Accept")',
                    'button:has-text("Accept all")',
                    'button:has-text("Accept")',
                    'button:has-text("Ok")',
                    "button#bnp_btn_accept"
                ]
                for sel in consent_selectors:
                    loc = page.locator(sel)
                    if loc.count() > 0:
                        print("Clicking consent button:", sel)
                        loc.first.click(timeout=5000)
                        page.wait_for_timeout(800)
                        break
            except Exception:
                pass

            # Wait for results container or at least for some result links
            results_found = False
            for sel in RESULT_LINK_SELECTORS:
                try:
                    page.wait_for_selector(sel, timeout=30000)
                    results_found = True
                    break
                except PlaywrightTimeoutError:
                    continue

            if not results_found:
                # fallback: wait a bit and grab the page HTML for debugging
                print("No results selector matched. Saving HTML for debugging.")
                ts = int(time.time())
                html_file = f"bing_noresults_{ts}.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(page.content())
                print("Saved:", html_file)
                browser.close()
                return

            # Collect candidate links (use multiple selectors)
            results = page.locator(", ".join(RESULT_LINK_SELECTORS))
            count = results.count()
            print("Found result links:", count)
            chosen_index = None
            chosen_href = None
            chosen_title = None

            # Heuristics to pick a scorecard link
            for i in range(count):
                link = results.nth(i)
                try:
                    title = link.inner_text().strip()
                except Exception:
                    title = ""
                try:
                    href = link.get_attribute("href") or ""
                except Exception:
                    href = ""
                low_href = (href or "").lower()
                low_title = (title or "").lower()

                if ("scorecard" in low_href) or ("scorecard" in low_title) \
                   or ("espncricinfo" in low_href) or ("cricbuzz" in low_href) \
                   or ("cricket" in low_title) or ("cricket" in low_href):
                    chosen_index = i
                    chosen_href = href
                    chosen_title = title
                    break

            # fallback to first result
            if chosen_index is None and count > 0:
                chosen_index = 0
                link0 = results.nth(0)
                chosen_title = link0.inner_text().strip() if link0 else ""
                chosen_href = link0.get_attribute("href") if link0 else ""

            if chosen_index is None:
                print("No suitable result found.")
                browser.close()
                return

            print(f"Chosen result #{chosen_index + 1}: {chosen_title}")
            print("Raw href:", chosen_href)

            # If Bing uses ck/a redirect, extract actual target
            target = extract_bing_redirect(chosen_href)
            print("Resolved target:", target)

            # If we have a resolved absolute URL, navigate directly (more reliable)
            if target and (target.startswith("http://") or target.startswith("https://")):
                try:
                    page.goto(target, wait_until="networkidle", timeout=45000)
                except PlaywrightTimeoutError:
                    print("Navigation timeout on direct target; continuing to try clicking the link.")
            else:
                # otherwise click the link element (some links navigate via JS)
                try:
                    results.nth(chosen_index).scroll_into_view_if_needed()
                    results.nth(chosen_index).click(timeout=20000)
                except Exception as e:
                    print("Click failed:", e)
                    # fallback: try javascript navigation using href
                    if chosen_href:
                        try:
                            page.evaluate("href => window.location.href = href", chosen_href)
                        except Exception:
                            pass

            # Wait for page to load a bit
            try:
                page.wait_for_load_state("networkidle", timeout=30000)
            except PlaywrightTimeoutError:
                # still fine, continue to snapshot
                pass

            # give dynamic JS a moment
            page.wait_for_timeout(1200)

            ts = int(time.time())
            screenshot_path = f"scorecard_{ts}.png"
            page.screenshot(path=screenshot_path, full_page=True)
            print("Screenshot saved to:", screenshot_path)

            page_title = page.title()
            page_url = page.url
            print("Final page title:", page_title)
            print("Final page URL:", page_url)

            html_path = f"scorecard_{ts}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(page.content())
            print("HTML saved to:", html_path)

        except PlaywrightTimeoutError as e:
            print("A timeout occurred while performing steps:", str(e))
        except Exception as e:
            print("An unexpected error occurred:", str(e))
        finally:
            browser.close()


if __name__ == "__main__":
    headless_flag = False
    if "--headless" in sys.argv:
        headless_flag = True
    main(headless=headless_flag)
