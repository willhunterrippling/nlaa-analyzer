#!/usr/bin/env python3
"""
Emergency Eject Script for NLAA Analysis

This script generates a report from partial checkpoint data when you need
to abort a running analysis but still want results from what's been processed.

Features:
- Reads current checkpoint state and enrichment cache
- Re-runs matching on all cached enrichment data
- Generates a partial report showing progress
- Preserves checkpoint files for later resumption

Usage:
    python emergency_eject.py                    # Generate report with LLM matching
    python emergency_eject.py --no-llm           # Programmatic matching only (faster)
    python emergency_eject.py --output-dir out   # Custom output directory
    python emergency_eject.py --concurrency 10   # Limit concurrent LLM calls
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from checkpoint import CheckpointManager, CheckpointState
import re
from matching import AsyncHybridMatcher, HybridMatcher, MatchConfidence, programmatic_match

load_dotenv()


def fast_load_checkpoint_metadata(checkpoint_file: Path) -> dict:
    """
    Fast metadata extraction from checkpoint.json without parsing the huge processed_ids array.
    Uses regex to extract just the scalar values we need.
    """
    if not checkpoint_file.exists():
        return {}
    
    with open(checkpoint_file, 'r') as f:
        content = f.read()
    
    # Extract scalar values using regex (much faster than parsing 332K IDs)
    metadata = {}
    
    patterns = {
        'last_processed_index': r'"last_processed_index":\s*(\d+)',
        'total_records': r'"total_records":\s*(\d+)',
        'success_count': r'"success_count":\s*(\d+)',
        'error_count': r'"error_count":\s*(\d+)',
        'skipped_count': r'"skipped_count":\s*(\d+)',
        'started_at': r'"started_at":\s*"([^"]*)"',
        'last_checkpoint_at': r'"last_checkpoint_at":\s*"([^"]*)"',
        'input_file': r'"input_file":\s*"([^"]*)"',
        'checkpoint_version': r'"checkpoint_version":\s*"([^"]*)"',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            value = match.group(1)
            # Convert numeric values
            if key in ('last_processed_index', 'total_records', 'success_count', 'error_count', 'skipped_count'):
                value = int(value)
            metadata[key] = value
    
    # Count processed_ids by counting the array elements (fast string operation)
    # Find the processed_ids array and count commas + 1
    ids_match = re.search(r'"processed_ids":\s*\[', content)
    if ids_match:
        # Count quotes in the array (each ID is quoted)
        start = ids_match.end()
        # Find the closing bracket
        bracket_count = 1
        pos = start
        while bracket_count > 0 and pos < len(content):
            if content[pos] == '[':
                bracket_count += 1
            elif content[pos] == ']':
                bracket_count -= 1
            pos += 1
        
        array_content = content[start:pos-1]
        # Count IDs by counting quotes/2 (each ID is "xxx")
        if array_content.strip():
            metadata['processed_ids_count'] = array_content.count('"') // 2
        else:
            metadata['processed_ids_count'] = 0
    else:
        metadata['processed_ids_count'] = 0
    
    return metadata


# Default paths
DEFAULT_INPUT = "inputs/01c1b4ef-070a-972b-5342-83029a156523.csv"
DEFAULT_OUTPUT_DIR = "output"
CHECKPOINT_DIR = "checkpoints"


def load_csv_data(filepath: str) -> dict[str, dict]:
    """Load CSV file into dict keyed by ID."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return {row.get('ID', ''): row for row in reader}


def load_enrichment_cache(cache_file: Path, expected_count: int = 0) -> list[tuple[str, dict]]:
    """
    Load enrichment cache from JSONL file with progress display.
    Returns list of (record_id, data) tuples.
    """
    if not cache_file.exists():
        return []
    
    results = []
    start_time = time.time()
    
    with open(cache_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                results.append((entry["id"], entry["data"]))
                
                # Show progress every 10000 records
                if line_num % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = line_num / elapsed if elapsed > 0 else 0
                    if expected_count > 0:
                        pct = line_num / expected_count * 100
                        eta = (expected_count - line_num) / rate if rate > 0 else 0
                        print(f"\r   Loading cache: {pct:5.1f}% ({line_num:,}/{expected_count:,}) - {rate:.0f}/s, ETA: {eta:.0f}s", end="", flush=True)
                    else:
                        print(f"\r   Loading cache: {line_num:,} records - {rate:.0f}/s", end="", flush=True)
                    
            except (json.JSONDecodeError, KeyError) as e:
                print(f"\n  Warning: Skipping malformed cache line {line_num}: {e}")
                continue
    
    # Final progress update
    elapsed = time.time() - start_time
    print(f"\r   Loading cache: 100.0% ({len(results):,} records) - {elapsed:.1f}s total          ")
    
    return results


def count_cache_entries(cache_file: Path) -> int:
    """Count total entries in cache file using fast line counting."""
    if not cache_file.exists():
        return 0
    
    # Use wc -l style counting (much faster for large files)
    count = 0
    with open(cache_file, 'rb') as f:
        # Read in large chunks
        buf_size = 1024 * 1024  # 1MB chunks
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            count += buf.count(b'\n')
    return count


def format_experience_for_output(exp: Optional[dict]) -> dict:
    """Format experience object for CSV output."""
    if not exp:
        return {}
    return {
        'matched_company': exp.get('company', ''),
        'matched_title': exp.get('title', ''),
        'matched_start_date': exp.get('start_date', ''),
        'matched_end_date': exp.get('end_date', ''),
        'matched_is_current': exp.get('is_current', False),
    }


def save_csv(records: list[dict], filepath: Path):
    """Save records to CSV file."""
    if not records:
        return
    
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    
    priority_cols = [
        'ID', 'ACCOUNT_ID', 'ACCOUNT_NAME', 'CONTACT_NAME', 'EMAIL',
        'matched', 'match_confidence', 'match_reason',
        'matched_company', 'matched_title',
    ]
    
    fieldnames = [c for c in priority_cols if c in all_keys]
    fieldnames.extend(sorted(k for k in all_keys if k not in priority_cols))
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def generate_emergency_report(
    checkpoint_state: CheckpointState,
    processed_ids_count: int,
    results: list[dict],
    false_positives: list[dict],
    ambiguous_cases: list[dict],
    llm_errors: list[dict],
    match_stats: dict,
    elapsed_time: float,
    output_path: Path,
    timestamp: str,
    total_input_records: int,
) -> Path:
    """Generate an emergency/partial report from checkpoint data."""
    
    total_processed = len(results)
    total_enriched = checkpoint_state.success_count
    enrichment_errors = checkpoint_state.error_count
    
    fp_rate = (len(false_positives) / total_processed * 100) if total_processed > 0 else 0
    
    high_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'high')
    medium_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'medium')
    low_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'low')
    
    # Calculate progress
    progress_pct = (processed_ids_count / total_input_records * 100) if total_input_records > 0 else 0
    
    sample_fps = false_positives[:10] if false_positives else []
    
    report = f"""# NLAA False Positive Analysis - EMERGENCY EJECT REPORT

**⚠️ PARTIAL RESULTS - Analysis was interrupted**

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Original Start Time:** {checkpoint_state.started_at}  
**Last Checkpoint:** {checkpoint_state.last_checkpoint_at}

---

## Progress Status

| Metric | Value |
|--------|-------|
| **Original Dataset Size** | {total_input_records:,} records |
| **Records Processed** | {processed_ids_count:,} ({progress_pct:.1f}%) |
| **Enrichment Successes** | {total_enriched:,} |
| **Enrichment Errors** | {enrichment_errors:,} |
| **Checkpoint Preserved** | ✅ Yes - can resume later |

---

## Partial Results Summary

| Metric | Value |
|--------|-------|
| **Records Matched** | {total_processed:,} |
| **False Positives Found** | {len(false_positives):,} |
| **False Positive Rate** | **{fp_rate:.1f}%** |
| **Ambiguous Cases** | {len(ambiguous_cases):,} |
| **LLM Errors** | {len(llm_errors):,} |
| **Matching Time** | {elapsed_time:.1f} seconds |

### Key Finding (Partial)

Based on **{total_processed:,} records** ({progress_pct:.1f}% of total), **{len(false_positives):,} leads ({fp_rate:.1f}%)** were incorrectly flagged as "No Longer At Account" and are actually still at their company based on current LinkedIn data.

---

## Matching Breakdown

### By Method

| Method | Matches |
|--------|---------|
| Programmatic | {match_stats.get('programmatic_matches', 0):,} |
| LLM Fallback | {match_stats.get('llm_matches', 0):,} |
| No Match | {match_stats.get('no_matches', 0):,} |

**LLM Calls Made:** {match_stats.get('llm_calls', 0):,}

### By Confidence Level

| Confidence | Count | Percentage of FPs |
|------------|-------|-------------------|
| High | {high_conf:,} | {high_conf/len(false_positives)*100 if false_positives else 0:.1f}% |
| Medium | {medium_conf:,} | {medium_conf/len(false_positives)*100 if false_positives else 0:.1f}% |
| Low | {low_conf:,} | {low_conf/len(false_positives)*100 if false_positives else 0:.1f}% |
"""

    # Add LLM errors section if there were any
    if llm_errors:
        report += f"""
---

## ⚠️ LLM Errors

**{len(llm_errors):,} records had LLM matching failures.**

| Account Name | Current Companies | Error |
|--------------|-------------------|-------|
"""
        for err in llm_errors[:5]:
            account = err.get('ACCOUNT_NAME', 'N/A')[:35]
            companies = err.get('all_current_companies', 'N/A')[:35]
            reason = err.get('match_reason', 'N/A')
            if 'LLM error:' in reason:
                reason = reason.replace('LLM error: ', '')[:60]
            report += f"| {account} | {companies} | {reason}... |\n"
        
        if len(llm_errors) > 5:
            report += f"\n*Showing 5 of {len(llm_errors):,} LLM errors.*\n"

    report += """
---

## Sample False Positives

| Account Name | Matched LinkedIn Company | Confidence | Reason |
|--------------|-------------------------|------------|--------|
"""
    
    for fp in sample_fps:
        account = fp.get('ACCOUNT_NAME', 'N/A')[:40]
        matched = fp.get('matched_company', 'N/A')[:40]
        conf = fp.get('match_confidence', 'N/A')
        reason = fp.get('match_reason', 'N/A')[:50]
        report += f"| {account} | {matched} | {conf} | {reason}... |\n"
    
    if len(false_positives) > 10:
        report += f"\n*Showing 10 of {len(false_positives):,} false positives. See CSV for full list.*\n"
    
    report += f"""
---

## How to Resume

The checkpoint has been preserved. To continue processing:

```bash
python nlaa_analyzer_async.py --resume
```

To start fresh (⚠️ will lose progress):

```bash
python nlaa_analyzer_async.py --clear-checkpoint
```

---

## Output Files

- `false_positives_partial.csv` - {len(false_positives):,} confirmed false positives (so far)
- `ambiguous_cases_partial.csv` - {len(ambiguous_cases):,} cases needing review
- `all_results_partial.csv` - All {total_processed:,} matched results

---

*Report generated by Emergency Eject Script*  
*Checkpoint files preserved in `{CHECKPOINT_DIR}/`*
"""
    
    report_path = output_path / f"emergency_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


async def run_matching_async(
    enrichment_data: list[tuple[str, dict]],
    records_by_id: dict[str, dict],
    use_llm: bool,
    llm_model: str,
    concurrency: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """
    Run matching on enrichment data asynchronously.
    
    Returns: (results, false_positives, ambiguous_cases, llm_errors, match_stats)
    """
    matcher = AsyncHybridMatcher(use_llm=use_llm, model=llm_model)
    
    results = []
    false_positives = []
    ambiguous_cases = []
    llm_errors = []
    
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    
    # Progress tracking
    total = len(enrichment_data)
    processed_count = 0
    start_time = time.time()
    
    def print_progress():
        """Print progress bar with rate and ETA."""
        nonlocal processed_count
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        pct = processed_count / total * 100 if total > 0 else 0
        
        bar_width = 40
        filled = int(bar_width * processed_count / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        eta = (total - processed_count) / rate if rate > 0 else 0
        eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        llm_calls = matcher.get_stats().get('llm_calls', 0)
        
        print(
            f"\r   {pct:5.1f}% |{bar}| {processed_count:,}/{total:,} "
            f"[{elapsed_str}<{eta_str}] FP:{len(false_positives):,} "
            f"LLM:{llm_calls} ({rate:.1f}/s)",
            end="", flush=True
        )
    
    async def process_one(record_id: str, enriched_data: dict):
        nonlocal processed_count
        
        record = records_by_id.get(record_id, {})
        if not record:
            async with lock:
                processed_count += 1
            return None
        
        account_name = record.get('ACCOUNT_NAME', '')
        experiences = enriched_data.get('experience', [])
        
        async with semaphore:
            match_result = await matcher.match(account_name, experiences)
        
        result_record = {
            **record,
            'enrichment_success': True,
            'experience_count': len(experiences),
            'current_experience_count': len([e for e in experiences if e.get('is_current') or e.get('end_date') is None]),
            'matched': match_result.matched,
            'match_confidence': match_result.confidence.value,
            'match_reason': match_result.match_reason,
            'llm_error': match_result.llm_error,
            'all_current_companies': ', '.join(c for c in match_result.all_current_companies if c),
            **format_experience_for_output(match_result.matching_experience),
        }
        
        async with lock:
            results.append(result_record)
            
            if match_result.llm_error:
                llm_errors.append(result_record)
            elif match_result.matched:
                false_positives.append(result_record)
            elif match_result.confidence == MatchConfidence.LOW:
                ambiguous_cases.append(result_record)
            
            processed_count += 1
        
        return result_record
    
    print(f"\n   Processing {total:,} cached enrichments with {concurrency} concurrent workers...")
    print()
    
    # Create all tasks
    tasks = [process_one(rid, data) for rid, data in enrichment_data]
    
    # Process with progress updates
    last_update = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        
        # Update progress every 50 records or at the end
        if processed_count - last_update >= 50 or processed_count == total:
            print_progress()
            last_update = processed_count
    
    print()  # Newline after progress bar
    
    return results, false_positives, ambiguous_cases, llm_errors, matcher.get_stats()


def run_matching_sync(
    enrichment_data: list[tuple[str, dict]],
    records_by_id: dict[str, dict],
    use_llm: bool,
    llm_model: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """
    Run matching on enrichment data synchronously (for --no-llm mode).
    
    Returns: (results, false_positives, ambiguous_cases, llm_errors, match_stats)
    """
    matcher = HybridMatcher(use_llm=use_llm, model=llm_model)
    
    results = []
    false_positives = []
    ambiguous_cases = []
    llm_errors = []
    
    total = len(enrichment_data)
    start_time = time.time()
    
    print(f"\n   Processing {total:,} cached enrichments (programmatic only)...")
    print()
    
    for i, (record_id, enriched_data) in enumerate(enrichment_data):
        record = records_by_id.get(record_id, {})
        if not record:
            continue
        
        account_name = record.get('ACCOUNT_NAME', '')
        experiences = enriched_data.get('experience', [])
        
        match_result = matcher.match(account_name, experiences)
        
        result_record = {
            **record,
            'enrichment_success': True,
            'experience_count': len(experiences),
            'current_experience_count': len([e for e in experiences if e.get('is_current') or e.get('end_date') is None]),
            'matched': match_result.matched,
            'match_confidence': match_result.confidence.value,
            'match_reason': match_result.match_reason,
            'llm_error': match_result.llm_error,
            'all_current_companies': ', '.join(c for c in match_result.all_current_companies if c),
            **format_experience_for_output(match_result.matching_experience),
        }
        
        results.append(result_record)
        
        if match_result.llm_error:
            llm_errors.append(result_record)
        elif match_result.matched:
            false_positives.append(result_record)
        elif match_result.confidence == MatchConfidence.LOW:
            ambiguous_cases.append(result_record)
        
        # Update progress bar every 500 records or at the end
        if (i + 1) % 500 == 0 or i == total - 1:
            processed = i + 1
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            pct = processed / total * 100 if total > 0 else 0
            
            bar_width = 40
            filled = int(bar_width * processed / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_width - filled)
            
            eta = (total - processed) / rate if rate > 0 else 0
            eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
            elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
            
            print(
                f"\r   {pct:5.1f}% |{bar}| {processed:,}/{total:,} "
                f"[{elapsed_str}<{eta_str}] FP:{len(false_positives):,} ({rate:.1f}/s)",
                end="", flush=True
            )
    
    print()  # Newline
    
    return results, false_positives, ambiguous_cases, llm_errors, matcher.get_stats()


def main():
    parser = argparse.ArgumentParser(
        description="Emergency Eject - Generate report from partial checkpoint data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report with LLM matching for ambiguous cases
  python emergency_eject.py

  # Fast mode - programmatic matching only
  python emergency_eject.py --no-llm

  # Custom output directory
  python emergency_eject.py --output-dir my_output

  # Show checkpoint status only
  python emergency_eject.py --status
        """
    )
    
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT,
                        help='Original input CSV file')
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR,
                        help='Output directory for report')
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_DIR,
                        help='Checkpoint directory')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM matching (faster, programmatic only)')
    parser.add_argument('--llm-model', default='gpt-4o-mini',
                        help='LLM model for matching')
    parser.add_argument('--concurrency', type=int, default=20,
                        help='Concurrent LLM calls')
    parser.add_argument('--status', action='store_true',
                        help='Show checkpoint status and exit')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70, flush=True)
    print("  EMERGENCY EJECT - Partial Report Generator", flush=True)
    print("=" * 70, flush=True)
    
    # Check checkpoint file exists and show size
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    cache_file = checkpoint_dir / "enrichment_cache.jsonl"
    
    if not checkpoint_file.exists():
        print("\n✗ No checkpoint found!")
        print(f"  Looked in: {args.checkpoint_dir}/")
        print("\n  There's no partial data to generate a report from.")
        sys.exit(1)
    
    cp_size = checkpoint_file.stat().st_size / (1024 * 1024)  # MB
    print(f"\n⏳ Loading checkpoint metadata ({cp_size:.1f} MB)...", end="", flush=True)
    
    # Use fast metadata extraction instead of full JSON parse
    checkpoint_meta = fast_load_checkpoint_metadata(checkpoint_file)
    
    if not checkpoint_meta:
        print(" failed!")
        print("\n✗ Could not parse checkpoint file!")
        sys.exit(1)
    
    # Create a lightweight state object from metadata
    state = CheckpointState()
    state.input_file = checkpoint_meta.get('input_file', '')
    state.started_at = checkpoint_meta.get('started_at', '')
    state.last_checkpoint_at = checkpoint_meta.get('last_checkpoint_at', '')
    state.success_count = checkpoint_meta.get('success_count', 0)
    state.error_count = checkpoint_meta.get('error_count', 0)
    state.skipped_count = checkpoint_meta.get('skipped_count', 0)
    state.total_records = checkpoint_meta.get('total_records', 0)
    state.last_processed_index = checkpoint_meta.get('last_processed_index', 0)
    # We use the count instead of the actual set (much faster)
    processed_ids_count = checkpoint_meta.get('processed_ids_count', 0)
    state.processed_ids = set()  # Empty - we just use the count
    
    print(f" done!", flush=True)
    print(f"\n✓ Checkpoint loaded (fast mode)", flush=True)
    print(f"  Input file: {state.input_file}", flush=True)
    print(f"  Started: {state.started_at}", flush=True)
    print(f"  Last checkpoint: {state.last_checkpoint_at}", flush=True)
    print(f"\n  Progress:", flush=True)
    print(f"    Processed IDs: {processed_ids_count:,}", flush=True)
    print(f"    Success count: {state.success_count:,}", flush=True)
    print(f"    Error count: {state.error_count:,}", flush=True)
    print(f"    Skipped count: {state.skipped_count:,}", flush=True)
    
    # Estimate cache entries from success_count (fast) or count if needed
    # success_count should equal cache entries since we cache on success
    cache_count = state.success_count
    if cache_count == 0 and cache_file.exists():
        # Fallback to counting if success_count is 0 but file exists
        print(f"    Counting cache entries...", end="", flush=True)
        cache_count = count_cache_entries(cache_file)
        print(f"\r    Cache entries: {cache_count:,}          ", flush=True)
    else:
        print(f"    Cache entries: ~{cache_count:,} (estimated from success_count)", flush=True)
    
    if args.status:
        print("\n  (Status check only - use without --status to generate report)")
        sys.exit(0)
    
    # Load original input data
    input_file = args.input if os.path.exists(args.input) else state.input_file
    if not os.path.exists(input_file):
        print(f"\n✗ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"\n📂 Loading original input: {input_file}")
    records_by_id = load_csv_data(input_file)
    total_input_records = len(records_by_id)
    print(f"   Total records in input: {total_input_records:,}")
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"emergency_eject_{timestamp}"
    output_path = Path(args.output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output folder: {output_path}")
    
    # Load enrichment cache
    print(f"\n📥 Loading enrichment cache...")
    
    enrichment_data = load_enrichment_cache(cache_file, expected_count=cache_count)
    
    start_time = time.time()
    
    # Run matching
    print(f"\n🎯 Running matching ({'programmatic only' if args.no_llm else 'with LLM fallback'})...")
    
    if args.no_llm:
        results, false_positives, ambiguous_cases, llm_errors, match_stats = run_matching_sync(
            enrichment_data=enrichment_data,
            records_by_id=records_by_id,
            use_llm=False,
            llm_model=args.llm_model,
        )
    else:
        results, false_positives, ambiguous_cases, llm_errors, match_stats = asyncio.run(
            run_matching_async(
                enrichment_data=enrichment_data,
                records_by_id=records_by_id,
                use_llm=True,
                llm_model=args.llm_model,
                concurrency=args.concurrency,
            )
        )
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("  PARTIAL RESULTS")
    print("=" * 70)
    
    total_matched = len(results)
    progress_pct = processed_ids_count / total_input_records * 100 if total_input_records > 0 else 0
    fp_rate = len(false_positives) / total_matched * 100 if total_matched > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"   Progress: {processed_ids_count:,}/{total_input_records:,} ({progress_pct:.1f}%)")
    print(f"   Records matched: {total_matched:,}")
    print(f"\n   FALSE POSITIVES FOUND: {len(false_positives):,}")
    print(f"   False positive rate: {fp_rate:.1f}%")
    print(f"   Ambiguous cases: {len(ambiguous_cases):,}")
    
    if llm_errors:
        print(f"\n   ⚠️  LLM ERRORS: {len(llm_errors):,}")
    
    print(f"\n   Matching breakdown:")
    print(f"     Programmatic matches: {match_stats['programmatic_matches']}")
    print(f"     LLM matches: {match_stats['llm_matches']}")
    print(f"     No matches: {match_stats['no_matches']}")
    
    # Save output files
    if false_positives:
        fp_file = output_path / "false_positives_partial.csv"
        save_csv(false_positives, fp_file)
        print(f"\n💾 Saved false positives: {fp_file}")
    
    if ambiguous_cases:
        amb_file = output_path / "ambiguous_cases_partial.csv"
        save_csv(ambiguous_cases, amb_file)
        print(f"💾 Saved ambiguous cases: {amb_file}")
    
    if results:
        all_file = output_path / "all_results_partial.csv"
        save_csv(results, all_file)
        print(f"💾 Saved all results: {all_file}")
    
    # Generate report
    report_path = generate_emergency_report(
        checkpoint_state=state,
        processed_ids_count=processed_ids_count,
        results=results,
        false_positives=false_positives,
        ambiguous_cases=ambiguous_cases,
        llm_errors=llm_errors,
        match_stats=match_stats,
        elapsed_time=elapsed_time,
        output_path=output_path,
        timestamp=timestamp,
        total_input_records=total_input_records,
    )
    print(f"📊 Saved emergency report: {report_path}")
    
    print(f"\n{'=' * 70}")
    print("  CHECKPOINT PRESERVED")
    print("=" * 70)
    print(f"\n   ✅ Checkpoint files are intact in: {args.checkpoint_dir}/")
    print(f"   To resume the full analysis later:")
    print(f"      python nlaa_analyzer_async.py --resume")
    print(f"\n{'=' * 70}")
    
    return {
        'total_matched': total_matched,
        'false_positives': len(false_positives),
        'ambiguous': len(ambiguous_cases),
        'progress_pct': progress_pct,
    }


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted!")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

