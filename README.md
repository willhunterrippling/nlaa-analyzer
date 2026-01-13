# NLAA False Positive Analyzer

A tool for identifying **false positives** in Salesforce leads flagged as "No Longer At Account" (NLAA). The analyzer enriches LinkedIn profiles via the MixRank API and uses a hybrid matching engine (programmatic + LLM) to detect leads who are actually still at their company.

---

> ⚠️ **IMPORTANT: Always use the async version with concurrency 20**
>
> ```bash
> python nlaa_analyzer_async.py --concurrency 20 [other options]
> ```
>
> Do NOT use `nlaa_analyzer.py` (sequential version) — it's 15x slower.

---

## Overview

When leads are marked as NLAA in Salesforce, some are incorrectly flagged—the person is actually still at the company. This tool:

1. Takes SFDC leads flagged as NLAA with LinkedIn URLs
2. Enriches each profile via MixRank API to get current employment data
3. Matches the SFDC account name to LinkedIn experience entries
4. Identifies **false positives** (people still at their account)

---

## Project Structure

```
NLAA analysis/
├── nlaa_analyzer_async.py  # Async/parallel version (RECOMMENDED)
├── nlaa_analyzer.py        # Sequential version (slower)
├── enrichment_async.py     # Async MixRank API client
├── enrichment.py           # Sequential MixRank API client
├── matching.py             # Hybrid company name matching engine
├── checkpoint.py           # Crash recovery & checkpointing system
├── url_normalizer.py       # LinkedIn URL normalization utilities
├── inputs/                 # Input CSV files directory
│   └── 01c1b4ef-070a-972b-5342-83029a156523.csv
├── checkpoints/            # Checkpoint state files (auto-created)
│   ├── checkpoint.json
│   └── enrichment_cache.json
└── output/                 # Results output directory (auto-created)
    └── run_TIMESTAMP_test_N/  # Each run gets its own subfolder
```

---

## Requirements

### Dependencies

Install required packages:

```bash
pip install requests python-dotenv openai aiohttp
```

Or use the `requirements.txt`:

```
requests>=2.28.0
python-dotenv>=1.0.0
openai>=1.0.0
aiohttp>=3.9.0
```

### API Keys

Create a `.env` file in the project root:

```bash
MIXRANK_API_KEY=your_mixrank_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional: only needed for LLM matching
```

Or export as environment variables:

```bash
export MIXRANK_API_KEY="your_key"
export OPENAI_API_KEY="your_key"
```

---

## Input Data

### Downloading the Dataset

The default input file (`01c1b4ef-070a-972b-5342-83029a156523.csv`) is too large for GitHub (221 MB). Download it from Google Drive:

📥 **[Download Input CSV from Google Drive](https://drive.google.com/file/d/1RC_VaW_ecgr8CatuVjOU8ZB-R2uDW1Cl/view?usp=sharing)**

After downloading, place the file in the `inputs/` directory:

```bash
# Move the downloaded file to the inputs directory
mv ~/Downloads/01c1b4ef-070a-972b-5342-83029a156523.csv "NLAA analysis/inputs/"
```

Your directory structure should look like:

```
NLAA analysis/
└── inputs/
    └── 01c1b4ef-070a-972b-5342-83029a156523.csv
```

### Input Data Format

The input CSV must contain these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `ID` | Unique record identifier | Yes |
| `ACCOUNT_NAME` | SFDC account name to match against | Yes |
| `CONTACT_LINKED_IN_URL_C` | LinkedIn profile URL | Yes |

Additional columns are preserved in the output.

---

## Quick Start

```bash
# 1. Navigate to the project directory
cd "NLAA analysis"

# 2. Install dependencies
pip install requests python-dotenv openai aiohttp

# 3. Set up API keys (create .env file or export)
export MIXRANK_API_KEY="your_mixrank_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# 4. Run a test first (500 records with async/parallel)
python nlaa_analyzer_async.py --test --test-size 500 --concurrency 20

# 5. Run the full analysis (~13 hours)
python nlaa_analyzer_async.py --concurrency 20
```

---

## Usage

### Step-by-Step: Running the Full Analysis

#### Step 1: Set Up Environment

```bash
# Navigate to the NLAA analysis directory
cd "/Users/will/Documents/Mixrank/NLAA analysis"

# Install required Python packages
pip install requests python-dotenv openai
```

#### Step 2: Configure API Keys

Create a `.env` file in the `NLAA analysis` directory:

```bash
# Create .env file
cat > .env << 'EOF'
MIXRANK_API_KEY=your_mixrank_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
EOF
```

Or export directly in your shell:

```bash
export MIXRANK_API_KEY="your_mixrank_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

#### Step 3: Test with a Small Sample

Always test first before running the full dataset:

```bash
# Test with 50 random records, verbose output
python nlaa_analyzer.py --test --verbose

# Or test with a larger sample
python nlaa_analyzer.py --test --test-size 200 --verbose
```

#### Step 4: Run the Full Analysis

```bash
# Run on the default input file (01c1b4ef-070a-972b-5342-83029a156523.csv)
python nlaa_analyzer.py

# Or specify a custom input file
python nlaa_analyzer.py --input your_data.csv

# Recommended: Run with checkpointing for large datasets
python nlaa_analyzer.py \
  --input 01c1b4ef-070a-972b-5342-83029a156523.csv \
  --checkpoint-interval 100 \
  --rate-limit 2.0
```

#### Step 5: Check Results

Results are saved in the `output/` directory:

```bash
# List output files
ls -la output/

# View the analysis report
cat output/analysis_report_*.md

# Open false positives CSV
open output/false_positives_*.csv
```

---

### Resume Interrupted Run

If processing is interrupted (Ctrl+C, crash, or connection issues), resume from the last checkpoint:

```bash
# Resume from where you left off
python nlaa_analyzer.py --resume

# Resume with a specific input file
python nlaa_analyzer.py --resume --input your_data.csv
```

### Re-run Matching Only

Skip API enrichment and re-run matching on cached data (useful for tuning matching logic):

```bash
python nlaa_analyzer.py --match-only
```

### Clear Checkpoint and Start Fresh

```bash
# Clear existing checkpoint and start over
python nlaa_analyzer.py --clear-checkpoint --input your_data.csv
```

---

## Running the Full 1M+ Record Dataset

The default input file `01c1b4ef-070a-972b-5342-83029a156523.csv` contains ~1,075,000 records. Here's how to process it:

### Performance Estimates (Full 1M Run)

| Metric | Value |
|--------|-------|
| **Concurrency** | 20 parallel requests |
| **Throughput** | ~23 records/sec |
| **Total time** | **~13 hours** |
| **Expected false positives** | ~280,000 (28%) |
| **LLM escalation rate** | ~5% |
| **LLM cost** | ~$12 |

### Recommended Command for Full Run (ASYNC)

```bash
cd "/Users/will/Documents/Mixrank/NLAA analysis"

# Full run with async/parallel processing (RECOMMENDED)
python nlaa_analyzer_async.py --concurrency 20 --checkpoint-interval 100

# Estimated time: ~13 hours at 23 req/sec
# Progress is saved every 100 records, so you can interrupt and resume

# If interrupted, resume with:
python nlaa_analyzer_async.py --resume --concurrency 20
```

### Running in Background (for long runs)

```bash
# Run in background with nohup
nohup python nlaa_analyzer_async.py --concurrency 20 > nlaa_output.log 2>&1 &

# Check progress
tail -f nlaa_output.log

# Or use screen/tmux for better session management
screen -S nlaa
python nlaa_analyzer_async.py --concurrency 20
# Detach: Ctrl+A, D
# Reattach: screen -r nlaa
```

### Adjusting Concurrency

```bash
# Lower concurrency (safer, slower)
python nlaa_analyzer_async.py --concurrency 10

# Higher concurrency (faster, may hit rate limits)
python nlaa_analyzer_async.py --concurrency 30
```

### Programmatic-Only Mode (No LLM costs)

```bash
# Skip LLM fallback to save OpenAI API costs
# Note: May miss some ambiguous matches
python nlaa_analyzer_async.py --concurrency 20 --no-llm
```

---

### Command Line Options (Async Version)

| Option | Default | Description |
|--------|---------|-------------|
| `--input, -i` | `01c1b4ef-...csv` | Input CSV file path |
| `--output, -o` | `output` | Output directory |
| `--test, -t` | - | Enable test mode (process limited records) |
| `--test-size` | `500` | Number of records in test mode |
| `--concurrency` | `20` | Number of parallel API requests |
| `--verbose, -v` | - | Show detailed output for each match |
| `--resume, -r` | - | Resume from last checkpoint |
| `--match-only` | - | Skip enrichment, use cached data |
| `--checkpoint-interval` | `100` | Save checkpoint every N records |
| `--no-llm` | - | Disable LLM fallback (programmatic only) |
| `--llm-model` | `gpt-4o-mini` | LLM model for fallback matching |
| `--clear-checkpoint` | - | Clear existing checkpoint and start fresh |

---

## How It Works

### 1. Data Preprocessing

- **URL Normalization**: Fixes malformed LinkedIn URLs
  - Double URLs: `linkedin.com/https://linkedin.com/in/...`
  - Pub URLs: `/pub/name/a/b/c` → `/in/name-cba`
  - Missing protocol: `//www.linkedin.com/...`
  - Missing domain: `in/john-doe`

- **Validation**: Skips records with missing account names or invalid URLs

### 2. Profile Enrichment

The MixRank API enriches LinkedIn profiles to retrieve:
- Current employer(s)
- Job title(s)
- Employment history

**Rate Limiting**: Default 2 requests/second with exponential backoff on errors.

### 3. Company Name Matching

A **hybrid matching engine** determines if the SFDC account matches any current LinkedIn employer:

#### Programmatic Matching (~80% of cases)
1. **Normalize names**: lowercase, strip legal suffixes (Inc, LLC, Corp, etc.)
2. **Exact match**: After normalization
3. **Substring match**: One name contains the other
4. **Token overlap**: Jaccard similarity on word tokens

#### LLM Fallback (for ambiguous cases)
When programmatic matching returns LOW confidence:
- GPT-4o-mini evaluates if names represent the same company
- Handles abbreviations, regional divisions, alternative names

#### Confidence Levels

| Level | Description | Action |
|-------|-------------|--------|
| `HIGH` | Exact or near-exact match | Auto-approve as false positive |
| `MEDIUM` | Substring match with good overlap | Auto-approve as false positive |
| `LOW` | Weak signal, needs review | Escalate to LLM |
| `NO_MATCH` | No matching company found | Not a false positive |

### 4. Checkpointing

Progress is saved automatically every N records (default: 100):
- **checkpoint.json**: Processed record IDs, statistics, timestamps
- **enrichment_cache.json**: Cached API responses for re-matching

This allows:
- Crash recovery with `--resume`
- Re-running matching without re-calling APIs with `--match-only`

---

## Output Files

Each run creates its own subfolder in `output/`:

```
output/
├── run_20260113_113304_test_500/
│   ├── false_positives.csv
│   ├── ambiguous_cases.csv
│   ├── all_results.csv
│   ├── skipped_records.csv
│   └── analysis_report_*.md
└── run_20260113_113606_test_5000/
    └── ...
```

| File | Description |
|------|-------------|
| `false_positives.csv` | Leads incorrectly flagged as NLAA |
| `ambiguous_cases.csv` | Low-confidence cases needing review |
| `all_results.csv` | Complete results (test mode only) |
| `skipped_records.csv` | Records that couldn't be processed |
| `analysis_report_*.md` | Summary report with statistics & error breakdown |

### Output Columns

The false positives CSV includes:
- All original input columns
- `matched`: Boolean - was a match found?
- `match_confidence`: high/medium/low
- `match_reason`: Explanation of the match
- `matched_company`: LinkedIn company that matched
- `matched_title`: Job title at matched company
- `all_current_companies`: All current employers from LinkedIn

---

## Examples

### Example 1: Basic Test Run

```bash
python nlaa_analyzer.py --test --verbose

# Output:
# ======================================================================
#   NLAA FALSE POSITIVE ANALYZER
# ======================================================================
# 📂 Loading data from: 01c1b4ef-...csv
#    Total records: 1,075,392
# 🧪 TEST MODE: Sampling 50 random records
# ...
#    FALSE POSITIVES FOUND: 12
#    False positive rate: 24.0%
```

### Example 2: Full Run with Custom Rate Limit

```bash
python nlaa_analyzer.py \
  --input sfdc_nlaa_leads.csv \
  --rate-limit 5.0 \
  --checkpoint-interval 500
```

### Example 3: Programmatic Matching Only (No LLM)

```bash
python nlaa_analyzer.py --test --no-llm
```

### Example 4: Re-run Matching with Different Model

```bash
python nlaa_analyzer.py --match-only --llm-model gpt-4o
```

---

## Module Details

### `enrichment.py`

MixRank API client with:
- `RateLimiter`: Token bucket rate limiting
- `BackoffStrategy`: Exponential backoff (1s → 60s, 3 retries)
- `MixRankEnricher`: Main API client class
- `ProgressTracker`: Rich progress bar with ETA

### `matching.py`

Company name matching with:
- `normalize_company_name()`: Strips suffixes, normalizes case
- `calculate_token_overlap()`: Jaccard + coverage scoring
- `programmatic_match()`: Rule-based matching
- `LLMMatcher`: GPT-4o-mini fallback
- `HybridMatcher`: Combined approach

### `checkpoint.py`

Crash recovery with:
- `CheckpointState`: Serializable state object
- `CheckpointManager`: Save/load/resume operations

### `url_normalizer.py`

LinkedIn URL fixes:
- `normalize_linkedin_url()`: Main normalization function
- `extract_linkedin_slug()`: Get profile slug from URL
- `is_valid_linkedin_url()`: Validation check

---

## Troubleshooting

### "MixRank API key required"

Set the `MIXRANK_API_KEY` environment variable or add to `.env` file.

### "Rate limited - max retries exceeded"

The API rate limit was hit. Try:
- Lower the `--rate-limit` value (e.g., `--rate-limit 1.0`)
- Wait a few minutes before retrying

### "No checkpoint found"

When using `--resume`, ensure a previous run saved checkpoints. Start a fresh run without `--resume`.

### High number of skipped records

Check the `skipped_records_TIMESTAMP.csv` for details. Common causes:
- Missing LinkedIn URLs
- Invalid URL formats (not fixable by normalizer)
- Missing account names

---

## Performance

### Async Version (Recommended) - 20 concurrent requests

| Records | Time | Throughput |
|---------|------|------------|
| 500 | ~20 seconds | 23/sec |
| 5,000 | ~3.5 minutes | 23/sec |
| 50,000 | ~35 minutes | 23/sec |
| 1,000,000 | **~13 hours** | 23/sec |

### Sequential Version - 2 req/sec

| Records | Time | Throughput |
|---------|------|------------|
| 50 | ~30 seconds | 1.5/sec |
| 1,000 | ~10 minutes | 1.5/sec |
| 10,000 | ~1.5 hours | 1.5/sec |
| 1,000,000 | **~7 days** | 1.5/sec |

**Use the async version (`nlaa_analyzer_async.py`) for 15x faster processing.**

Checkpointing allows interruption/resumption for both versions.
