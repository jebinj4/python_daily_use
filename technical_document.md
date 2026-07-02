# Technical Document: FindBahrain Unified Scraper & Enricher

This document outlines the architecture, data models, and logic of the consolidated multi-category scraper script for findbahrain.com.

## Architecture

FindBahrain is a React application that queries database tables directly from a Supabase backend client (`https://omscmbpnvcmwzkijsmnm.supabase.co`).
We use **Playwright** to run a headless Chromium browser instance to load lists and details pages, allowing React to run and render the DOM fully before parsing.

## Consolidate Features

1. **Unified Flow**: Crawling, link cleaning, and email enrichment are handled end-to-end in a single command.
2. **Infinite Scroll Handler**: Listing pages automatically auto-scroll to trigger dynamic loading of all available business cards.
3. **CLI Arguments**: Scraping target is determined via dynamic arguments:
   - Specific subcategory URL: Scrapes just that subcategory list.
   - Category URL: Crawls all subcategories under that category.
   - None: Crawls the entire directory (all 10 categories).
4. **Resume Checkpoint (No Duplicates)**: If the script is stopped and run again, it automatically inspects `businesses.csv` and skips scraping any profile URL that has already been recorded, picking up exactly where it left off.

## Data Model (businesses.csv)

The output CSV contains the following columns for easy CRM mapping:
- **`Category`**: Parent category slug (e.g. `business-b2b`)
- **`Subcategory`**: Subcategory slug (e.g. `construction-contractors`)
- **`Name`**: Company name
- **`WhatsApp`**: Direct WhatsApp chat URL
- **`Phone`**: Phone number
- **`Instagram`**: Verified Instagram URL (excludes FindBahrain self-links)
- **`Location`**: Physical address / location
- **`Website`**: Verified company website URL (excludes FindBahrain self-links)
- **`Email`**: Enriched email address (scraped from company site or Google snippets)
- **`Website_Phone`**: Additional phone numbers discovered on the company website (comma-separated, filters out duplicates matching the main phone)
- **`Source_URL`**: Listing details URL (used as the unique key for resume checkpoints)

