# Error & Fix Log

This file records any errors encountered during the development and execution of the scraping script, along with their solutions.

## Log Entries

### 2026-07-01
- **Error**: `externally-managed-environment` when running `pip install playwright` on macOS.
- **Cause**: Homebrew-installed Python on macOS blocks global package installations to prevent breaking system tools.
- **Solution**: 
  - Option 1 (Recommended): Set up a Python Virtual Environment (`venv`).
  - Option 2: Run pip with `--break-system-packages` and run playwright installation via the Python module executor (`python3 -m playwright install chromium`).

