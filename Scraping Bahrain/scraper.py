import csv
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

BASE_URL = "https://findbahrain.com"
DIRECTORY_URL = f"{BASE_URL}/directory"
OUTPUT_FILE = "Scraping Bahrain/businesses.csv"
MAP_FILE = "Scraping Bahrain/category_map.json"

EMAIL_REGEX = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Thread locks to ensure thread-safe file writing and safe Google query rates
csv_lock = threading.Lock()
google_lock = threading.Lock()

def discover_categories_and_subcategories():
    print(f"Discovering categories from directory home page: {DIRECTORY_URL}")
    hierarchy = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        try:
            page.goto(DIRECTORY_URL, wait_until="domcontentloaded")
            time.sleep(5)
            
            # Find category links
            hrefs = page.eval_on_selector_all("a", "elements => elements.map(el => el.href)")
            categories = []
            for href in hrefs:
                parts = href.rstrip("/").split("/")
                if len(parts) == 5 and parts[3] == "directory":
                    categories.append(href)
            
            # For each category, find subcategories
            for cat in sorted(categories):
                cat_slug = cat.rstrip("/").split("/")[-1]
                print(f"  --> Crawling category category: {cat_slug}")
                page.goto(cat, wait_until="domcontentloaded")
                time.sleep(4)
                
                cat_hrefs = page.eval_on_selector_all("a", "elements => elements.map(el => el.href)")
                subcategories = set()
                for sub_href in cat_hrefs:
                    parts = sub_href.rstrip("/").split("/")
                    if len(parts) == 6 and parts[3] == "directory":
                        subcategories.add(sub_href)
                hierarchy[cat_slug] = sorted(list(subcategories))
                time.sleep(1)
        except Exception as e:
            print(f"Error during category mapping: {e}")
        finally:
            browser.close()
            
    # Save mapping file
    os.makedirs(os.path.dirname(MAP_FILE), exist_ok=True)
    with open(MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=4)
    print(f"Saved category map to: {MAP_FILE}")
    return hierarchy

def load_category_map():
    if os.path.exists(MAP_FILE):
        try:
            with open(MAP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return discover_categories_and_subcategories()

def get_detail_links(subcategory_url):
    print(f"Loading subcategory page: {subcategory_url}")
    links = set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        try:
            page.goto(subcategory_url, wait_until="domcontentloaded")
            
            # Infinite scroll handler to load all business cards
            last_height = page.evaluate("document.body.scrollHeight")
            no_change_count = 0
            while True:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(2)
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    no_change_count += 1
                    if no_change_count >= 3:
                        break
                else:
                    no_change_count = 0
                    last_height = new_height
            
            time.sleep(2)
            
            # Extract links
            hrefs = page.eval_on_selector_all("a", "elements => elements.map(el => el.href)")
            for href in hrefs:
                if subcategory_url.rstrip("/") in href:
                    if href.rstrip("/") != subcategory_url.rstrip("/"):
                        links.add(href)
        except Exception as e:
            print(f"Error loading list: {e}")
        finally:
            browser.close()
    print(f"Found {len(links)} company listing URLs.")
    return sorted(list(links))

def extract_emails(text):
    if not text:
        return set()
    emails = re.findall(EMAIL_REGEX, text)
    filtered = set()
    for email in emails:
        email_lower = email.lower()
        if not any(email_lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", "example.com"]):
            filtered.add(email)
    return filtered

def extract_phones(text):
    if not text:
        return set()
    # Match standard Bahrain numbers like +973 1773 7000 or 17737000 or 39940516
    # 8 digits starting with 1, 3, 6, 7, 8, 9
    pattern = r"(?:\+973|00973)?\s*\b[136789]\d{3}\s*\d{4}\b"
    raw_matches = re.findall(pattern, text)
    phones = set()
    for match in raw_matches:
        clean = re.sub(r"\s+", "", match)
        if clean.startswith("00973"):
            clean = "+" + clean[2:]
        elif clean.startswith("973") and len(clean) > 8:
            clean = "+" + clean
        elif len(clean) == 8:
            clean = "+973" + clean
        phones.add(clean)
    return phones

def crawl_website_for_contact_info(url):
    email = ""
    phones = set()
    if not url or not url.startswith("http"):
        return email, list(phones)
    
    print(f"  --> Checking website for contacts: {url}")
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            
            # mailto check
            mailto_links = soup.select("a[href^='mailto:']")
            for link in mailto_links:
                e = link.get("href").replace("mailto:", "").split("?")[0].strip()
                if e:
                    email = e
                    break
            
            # tel links check
            tel_links = soup.select("a[href^='tel:']")
            for link in tel_links:
                t = link.get("href").replace("tel:", "").strip()
                if t:
                    phones.add(t)
            
            # body search
            body_text = soup.get_text()
            if not email:
                emails = extract_emails(body_text)
                if emails:
                    email = list(emails)[0]
                    
            phones.update(extract_phones(body_text))
                
            # contact subpages
            contact_links = soup.find_all("a", href=re.compile(r"contact|about|info", re.IGNORECASE))
            for link in contact_links:
                contact_url = urljoin(url, link.get("href"))
                print(f"    --> Checking subpage: {contact_url}")
                try:
                    sub_resp = requests.get(contact_url, headers=REQUEST_HEADERS, timeout=10)
                    if sub_resp.status_code == 200:
                        sub_soup = BeautifulSoup(sub_resp.text, "html.parser")
                        
                        sub_mailtos = sub_soup.select("a[href^='mailto:']")
                        for sub_link in sub_mailtos:
                            e = sub_link.get("href").replace("mailto:", "").split("?")[0].strip()
                            if e and not email:
                                email = e
                                
                        sub_tels = sub_soup.select("a[href^='tel:']")
                        for sub_link in sub_tels:
                            t = sub_link.get("href").replace("tel:", "").strip()
                            if t:
                                phones.add(t)
                                
                        sub_body = sub_soup.get_text()
                        if not email:
                            sub_emails = extract_emails(sub_body)
                            if sub_emails:
                                email = list(sub_emails)[0]
                                
                        phones.update(extract_phones(sub_body))
                except Exception:
                    pass
    except Exception as e:
        print(f"    Error loading website {url}: {e}")
        
    return email, list(phones)

def search_google_for_email(company_name):
    with google_lock:
        query = f'"{company_name}" Bahrain email'
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}"
        print(f"  --> Searching Google for: {query}")
        
        try:
            response = requests.get(search_url, headers=REQUEST_HEADERS, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                emails = extract_emails(soup.get_text())
                if emails:
                    email_list = list(emails)
                    print(f"    --> Found in Google snippets: {email_list}")
                    time.sleep(2)
                    return email_list[0]
        except Exception as e:
            print(f"    Error searching Google: {e}")
            
        time.sleep(2)
        return ""

def scrape_detail_page(url, idx, total, category_slug, subcategory_slug):
    print(f"[{idx}/{total}] Scraping details: {url}")
    data = {
        "Category": category_slug,
        "Subcategory": subcategory_slug,
        "Name": "",
        "WhatsApp": "",
        "Phone": "",
        "Instagram": "",
        "Location": "",
        "Website": "",
        "Email": "",
        "Website_Phone": "",
        "Source_URL": url
    }
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        page = context.new_page()
        try:
            page.goto(url, wait_until="networkidle")
            time.sleep(2)
            
            # Name
            h1 = page.query_selector("h1")
            if h1:
                data["Name"] = h1.inner_text().strip()
                
            # Phone
            phone_el = page.query_selector("a[href^='tel:']")
            if phone_el:
                href = phone_el.get_attribute("href")
                data["Phone"] = href.replace("tel:", "").strip()
                
            # WhatsApp
            wa_el = page.query_selector("a[href*='wa.me'], a[href*='whatsapp.com']")
            if wa_el:
                data["WhatsApp"] = wa_el.get_attribute("href").strip()
            else:
                wa_btn = page.query_selector("text='Contact via WhatsApp'")
                if wa_btn:
                    parent = wa_btn.query_selector("xpath=..")
                    if parent and parent.get_attribute("href"):
                        data["WhatsApp"] = parent.get_attribute("href").strip()
            
            # Instagram
            insta_el = page.query_selector("a[href*='instagram.com']")
            if insta_el:
                href = insta_el.get_attribute("href").strip()
                if "findbahrain" not in href.lower():
                    data["Instagram"] = href
                
            # Address
            address_block = page.query_selector("main div div div.space-y-4 span")
            if address_block:
                data["Location"] = address_block.inner_text().strip()
            else:
                general_address = page.query_selector("text=/Address|Neighborhood|City/i")
                if general_address:
                    data["Location"] = general_address.inner_text().strip()
                    
            # Website
            all_links = page.query_selector_all("a")
            for link in all_links:
                href = link.get_attribute("href")
                if href:
                    href_lower = href.lower()
                    if href_lower.startswith("http") and not any(x in href_lower for x in ["instagram.com", "facebook.com", "wa.me", "whatsapp.com", "google.com/maps", "mailto:", "findbahrain"]):
                        data["Website"] = href.strip()
                        break
                        
        except Exception as e:
            print(f"Error scraping detail {url}: {e}")
        finally:
            browser.close()
            
    # Enrich Email & Website Phones
    if data["Website"]:
        email, website_phones = crawl_website_for_contact_info(data["Website"])
        data["Email"] = email
        
        # Filter out numbers that are identical to the primary phone number
        unique_phones = []
        main_phone_clean = re.sub(r"\s+", "", data["Phone"])
        for p_num in website_phones:
            p_clean = re.sub(r"\s+", "", p_num)
            if p_clean != main_phone_clean:
                unique_phones.append(p_num)
        data["Website_Phone"] = ", ".join(unique_phones)
        
    if not data["Email"] and data["Name"]:
        data["Email"] = search_google_for_email(data["Name"])
        
    return data

def get_existing_listing_urls():
    existing = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Source_URL"):
                        existing.add(row["Source_URL"].strip())
        except Exception:
            pass
    return existing

def process_subcategory(subcategory_url, category_slug, subcategory_slug, writer, existing_urls, f, max_workers=5):
    print(f"\n--- Starting Subcategory: {category_slug} -> {subcategory_slug} ---")
    detail_links = get_detail_links(subcategory_url)
    
    # Filter out already scraped listing URLs
    new_links = [link for link in detail_links if link not in existing_urls]
    skipped_count = len(detail_links) - len(new_links)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} listings already present in businesses.csv (Resume Checkpoint)")
        
    total = len(new_links)
    if total == 0:
        return
        
    def worker(link_idx_tuple):
        link, idx = link_idx_tuple
        row = scrape_detail_page(link, idx, total, category_slug, subcategory_slug)
        with csv_lock:
            writer.writerow(row)
            f.flush()  # Flush python buffer to OS
            try:
                os.fsync(f.fileno())  # Force OS to write to disk immediately (required for macOS APFS)
            except Exception:
                pass
            print(f"Saved: {row}\n")
            
    # Parallel execution using a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(worker, [(link, idx) for idx, link in enumerate(new_links, 1)])

def main():
    target_url = sys.argv[1] if len(sys.argv) > 1 else ""
    
    # Load categories map
    cat_map = load_category_map()
    
    # Resolve target jobs
    jobs = [] # list of (subcategory_url, category_slug, subcategory_slug)
    
    if target_url:
        parts = target_url.rstrip("/").split("/")
        if len(parts) == 6 and parts[3] == "directory": # Specific subcategory
            cat_slug = parts[4]
            sub_slug = parts[5]
            jobs.append((target_url, cat_slug, sub_slug))
        elif len(parts) == 5 and parts[3] == "directory": # Parent category
            cat_slug = parts[4]
            if cat_slug in cat_map:
                for sub_url in cat_map[cat_slug]:
                    sub_slug = sub_url.rstrip("/").split("/")[-1]
                    jobs.append((sub_url, cat_slug, sub_slug))
            else:
                print(f"Category slug '{cat_slug}' not found in map.")
                return
        else:
            print("Invalid category/subcategory URL format. URL must contain /directory/...")
            return
    else:
        # Default: crawl the entire directory
        print("No target URL specified. Crawling all categories in the directory...")
        for cat_slug, sub_urls in cat_map.items():
            for sub_url in sub_urls:
                sub_slug = sub_url.rstrip("/").split("/")[-1]
                jobs.append((sub_url, cat_slug, sub_slug))

    if not jobs:
        print("No scraping tasks found.")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    fieldnames = ["Category", "Subcategory", "Name", "WhatsApp", "Phone", "Instagram", "Location", "Website", "Email", "Website_Phone", "Source_URL"]
    
    # Check if header needs to be written
    file_exists = os.path.exists(OUTPUT_FILE)
    
    existing_urls = get_existing_listing_urls()
    print(f"Checkpoint Resume: {len(existing_urls)} listings already recorded.")
    
    # Open CSV in append mode
    with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists or os.path.getsize(OUTPUT_FILE) == 0:
            writer.writeheader()
            
        for job_idx, (sub_url, cat_slug, sub_slug) in enumerate(jobs, 1):
            print(f"\n==================================================")
            print(f"Processing category sublist [{job_idx}/{len(jobs)}]")
            print(f"==================================================")
            process_subcategory(sub_url, cat_slug, sub_slug, writer, existing_urls, f)
            
    print(f"\nAll tasks complete! Unified dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
