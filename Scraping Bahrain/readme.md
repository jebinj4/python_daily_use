# FindBahrain.com Scraper & Email Enricher

This is a single-script scraper that crawls FindBahrain.com, extracts company details, filters out FindBahrain self-links, and crawls company websites/Google search to enrich email addresses automatically.

## Setup

Install the required python packages and Playwright browser:

```bash
pip install playwright beautifulsoup4 requests --break-system-packages
python3 -m playwright install chromium
```

## Running the Scraper

You can run the script with different levels of targets:

### 1. Scrape a Specific Subcategory (Fastest)
Pass the direct URL of a subcategory listing page:
```bash
python "Scraping Bahrain/scraper.py" https://findbahrain.com/directory/business-b2b/construction-contractors
```

### 2. Scrape an Entire Category
Pass the direct URL of a category list page (e.g. B2B, Food, Hotels). The script will automatically discover and crawl all subcategories under it:
```bash
python "Scraping Bahrain/scraper.py" https://findbahrain.com/directory/business-b2b
```

### 3. Scrape the Entire Website (Will take several hours)
Run the script without any parameters:
```bash
python "Scraping Bahrain/scraper.py"
```

---

## Features

- **Resume Checkpoint (No Duplicates)**: If the scraper gets stopped (by pressing `Ctrl+C`), running the command again will automatically inspect `businesses.csv` and skip any company listing URLs that have already been scraped, continuing exactly where you left off.
- **Unified CRM Output**: The output is appended directly to `Scraping Bahrain/businesses.csv` and includes `Category` and `Subcategory` columns for easy mapping during CRM importing.
