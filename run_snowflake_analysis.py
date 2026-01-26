#!/usr/bin/env python3
"""
Snowflake-Based NLAA Analysis Script

Processes remaining records using Snowflake enrichment (instead of MixRank API)
and automatically combines with emergency stop results at the end.

Usage:
    python run_snowflake_analysis.py
    python run_snowflake_analysis.py --input inputs/remaining.csv
    python run_snowflake_analysis.py --test --test-size 1000
    python run_snowflake_analysis.py --no-combine  # Skip auto-combination
"""

import argparse
import asyncio
import csv
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from url_normalizer import normalize_linkedin_url
from snowflake_enrichment import SnowflakeEnricher, HybridEnricher
from matching import AsyncHybridMatcher, HybridMatcher, MatchConfidence
from combine_results import combine_results

load_dotenv()


# Default paths
DEFAULT_INPUT = "inputs/remaining.csv"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_EMERGENCY_RESULTS = "output/emergency_eject_20260113_190010/all_results_partial.csv"
CHECKPOINT_DIR = "checkpoints"


def load_csv_data(filepath: str) -> list[dict]:
    """Load CSV file into list of dicts."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def sample_records(records: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n random records from the dataset."""
    random.seed(seed)
    return random.sample(records, min(n, len(records)))


def preprocess_record(record: dict) -> tuple[dict, Optional[str]]:
    """Preprocess a single record."""
    account_name = record.get('ACCOUNT_NAME', '').strip()
    linkedin_url = record.get('CONTACT_LINKED_IN_URL_C', '').strip()
    
    if not account_name:
        return record, "Missing ACCOUNT_NAME"
    
    if not linkedin_url or linkedin_url in ('\\N', 'NULL', 'null'):
        return record, "Missing LinkedIn URL"
    
    url_result = normalize_linkedin_url(linkedin_url)
    
    if not url_result.success:
        return record, f"Invalid LinkedIn URL: {url_result.error}"
    
    record['_normalized_url'] = url_result.normalized
    record['_url_issue_type'] = url_result.issue_type
    
    return record, None


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


def generate_report(
    results: list[dict],
    false_positives: list[dict],
    ambiguous_cases: list[dict],
    llm_errors: list[dict],
    skipped_records: list[dict],
    match_stats: dict,
    enricher_stats: dict,
    elapsed_time: float,
    output_path: Path,
    timestamp: str,
    test_mode: bool = False
) -> Path:
    """Generate a markdown report of the analysis."""
    
    total_processed = len(results)
    enrichment_success = sum(1 for r in results if r.get('enrichment_success'))
    enrichment_failed = total_processed - enrichment_success
    fp_rate = (len(false_positives) / total_processed * 100) if total_processed > 0 else 0
    
    high_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'high')
    medium_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'medium')
    low_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'low')
    
    report = f"""# NLAA False Positive Analysis Report (Snowflake)

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Mode:** {"Test Mode" if test_mode else "Full Run"} | **Data Source:** Snowflake

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Records Processed** | {total_processed:,} |
| **False Positives Found** | {len(false_positives):,} |
| **False Positive Rate** | **{fp_rate:.1f}%** |
| **Ambiguous Cases** | {len(ambiguous_cases):,} |
| **LLM Errors** | {len(llm_errors):,} |
| **Skipped Records** | {len(skipped_records):,} |
| **Processing Time** | {elapsed_time/60:.1f} minutes |

### Key Finding

**{len(false_positives):,} leads ({fp_rate:.1f}%)** were incorrectly flagged as "No Longer At Account" and are actually still at their company based on current LinkedIn data.

---

## Enrichment Results (Snowflake)

| Status | Count | Percentage |
|--------|-------|------------|
| Success | {enrichment_success:,} | {enrichment_success/total_processed*100 if total_processed > 0 else 0:.1f}% |
| Failed | {enrichment_failed:,} | {enrichment_failed/total_processed*100 if total_processed > 0 else 0:.1f}% |

### Snowflake Statistics

| Metric | Value |
|--------|-------|
| URLs Queried | {enricher_stats.get('total_urls_queried', 0):,} |
| Found in Snowflake | {enricher_stats.get('found', 0):,} |
| Not Found | {enricher_stats.get('not_found', 0):,} |
| No Experience Data | {enricher_stats.get('no_experience_data', 0):,} |

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

**{len(llm_errors):,} records failed LLM matching.**

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
    
    sample_fps = false_positives[:10] if false_positives else []
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

## Skipped Records Summary

**Total Skipped:** {len(skipped_records):,}

Common skip reasons:
"""
    
    skip_reasons = {}
    for r in skipped_records:
        reason = r.get('_skip_reason', 'Unknown')
        if 'Missing LinkedIn' in reason:
            reason = 'Missing LinkedIn URL'
        elif 'Invalid LinkedIn' in reason:
            reason = 'Invalid LinkedIn URL format'
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
    
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1])[:5]:
        report += f"- {reason}: {count:,}\n"
    
    report += f"""
---

## Output Files

- `false_positives.csv` - All {len(false_positives):,} confirmed false positives
- `ambiguous_cases.csv` - {len(ambiguous_cases):,} cases needing review
- `all_results.csv` - Complete results
- `skipped_records.csv` - {len(skipped_records):,} records that couldn't be processed

---

*Report generated by Snowflake NLAA Analyzer*
"""
    
    report_path = output_path / f"analysis_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


async def run_matching_async(
    records_with_enrichment: list[tuple[dict, dict]],
    matcher: AsyncHybridMatcher,
    concurrency: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Run async matching on enriched records.
    
    Args:
        records_with_enrichment: List of (record, enrichment_data) tuples
        matcher: Async matcher instance
        concurrency: Max concurrent LLM calls
        
    Returns:
        (results, false_positives, ambiguous_cases, llm_errors)
    """
    results = []
    false_positives = []
    ambiguous_cases = []
    llm_errors = []
    
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    
    total = len(records_with_enrichment)
    processed_count = 0
    start_time = time.time()
    
    async def process_one(record: dict, enrichment_data: Optional[dict]):
        nonlocal processed_count
        
        account_name = record.get('ACCOUNT_NAME', '')
        
        if enrichment_data and enrichment_data.get('experience'):
            experiences = enrichment_data.get('experience', [])
            
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
        else:
            # Enrichment failed
            result_record = {
                **record,
                'enrichment_success': False,
                'enrichment_error': enrichment_data.get('error', 'No data') if enrichment_data else 'No enrichment data',
            }
            
            async with lock:
                results.append(result_record)
                processed_count += 1
        
        return result_record
    
    def print_progress():
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        pct = processed_count / total * 100 if total > 0 else 0
        
        bar_width = 40
        filled = int(bar_width * processed_count / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        eta = (total - processed_count) / rate if rate > 0 else 0
        eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        print(
            f"\r   {pct:5.1f}% |{bar}| {processed_count:,}/{total:,} "
            f"[{elapsed_str}<{eta_str}] FP:{len(false_positives):,} ({rate:.1f}/s)",
            end="", flush=True
        )
    
    print(f"\n   Matching {total:,} records with {concurrency} concurrent workers...")
    print()
    
    # Create all tasks
    tasks = [process_one(record, enrichment) for record, enrichment in records_with_enrichment]
    
    # Process with progress updates
    last_update = 0
    for coro in asyncio.as_completed(tasks):
        await coro
        
        if processed_count - last_update >= 100 or processed_count == total:
            print_progress()
            last_update = processed_count
    
    print()  # Newline after progress bar
    
    return results, false_positives, ambiguous_cases, llm_errors


def run_matching_sync(
    records_with_enrichment: list[tuple[dict, dict]],
    use_llm: bool,
    llm_model: str,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Run synchronous matching (for --no-llm mode)."""
    matcher = HybridMatcher(use_llm=use_llm, model=llm_model)
    
    results = []
    false_positives = []
    ambiguous_cases = []
    llm_errors = []
    
    total = len(records_with_enrichment)
    start_time = time.time()
    
    print(f"\n   Matching {total:,} records (programmatic only)...")
    print()
    
    for i, (record, enrichment_data) in enumerate(records_with_enrichment):
        account_name = record.get('ACCOUNT_NAME', '')
        
        if enrichment_data and enrichment_data.get('experience'):
            experiences = enrichment_data.get('experience', [])
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
        else:
            result_record = {
                **record,
                'enrichment_success': False,
                'enrichment_error': enrichment_data.get('error', 'No data') if enrichment_data else 'No enrichment data',
            }
            results.append(result_record)
        
        # Progress update
        if (i + 1) % 1000 == 0 or i == total - 1:
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
    
    print()
    
    return results, false_positives, ambiguous_cases, llm_errors, matcher.get_stats()


async def run_analysis(
    input_file: str,
    output_dir: str,
    emergency_results: str,
    test_mode: bool = False,
    test_size: int = 1000,
    batch_size: int = 10000,
    concurrency: int = 20,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
    auto_combine: bool = True,
    enable_mixrank_fallback: bool = False,
):
    """
    Main analysis pipeline using Snowflake enrichment.
    """
    
    print("\n" + "=" * 70)
    print("  NLAA FALSE POSITIVE ANALYZER (SNOWFLAKE)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_snowflake_{timestamp}_{'test' if test_mode else 'full'}_{test_size if test_mode else 'all'}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output folder: {output_path}")
    
    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    all_records = load_csv_data(input_file)
    print(f"   Total records: {len(all_records):,}")
    
    if test_mode:
        print(f"\n🧪 TEST MODE: Sampling {test_size} random records")
        records = sample_records(all_records, test_size)
    else:
        records = all_records
    
    # Preprocess records
    print(f"\n🔧 Preprocessing records...")
    valid_records = []
    skipped_records = []
    
    for record in records:
        processed, error = preprocess_record(record)
        
        if error:
            skipped_records.append({**record, '_skip_reason': error})
        else:
            valid_records.append(processed)
    
    print(f"   Valid records: {len(valid_records):,}")
    print(f"   Skipped: {len(skipped_records):,}")
    
    # Initialize Snowflake enricher
    print(f"\n❄️  Initializing Snowflake enricher (batch_size: {batch_size:,})...")
    
    if enable_mixrank_fallback:
        enricher = HybridEnricher(
            snowflake_batch_size=batch_size,
            enable_mixrank_fallback=True,
        )
    else:
        enricher = SnowflakeEnricher(batch_size=batch_size)
    
    # Test connection
    success, msg = enricher.test_connection()
    if not success:
        print(f"   ✗ Connection failed: {msg}")
        sys.exit(1)
    print(f"   ✓ {msg}")
    
    # Build URL to ID mapping for batch query
    print(f"\n📡 Enriching {len(valid_records):,} records from Snowflake...")
    
    url_to_record = {}
    for record in valid_records:
        url = record.get('_normalized_url', '')
        record_id = record.get('ID', '')
        if url and record_id:
            url_to_record[url] = record
    
    # Query Snowflake in batches
    enrichment_start = time.time()
    url_to_id = {url: record.get('ID', '') for url, record in url_to_record.items()}
    
    # Process in batches for progress display
    all_urls = list(url_to_id.keys())
    all_enrichment_results = {}
    
    total_batches = (len(all_urls) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, len(all_urls))
        batch_urls = all_urls[batch_start:batch_end]
        
        batch_url_to_id = {url: url_to_id[url] for url in batch_urls}
        
        print(f"\r   Batch {batch_num + 1}/{total_batches}: {batch_start:,}-{batch_end:,} of {len(all_urls):,}...", end="", flush=True)
        
        batch_results = enricher.enrich_batch(batch_url_to_id)
        all_enrichment_results.update(batch_results)
    
    enrichment_elapsed = time.time() - enrichment_start
    print(f"\r   ✓ Enrichment complete: {len(all_enrichment_results):,} URLs processed in {enrichment_elapsed:.1f}s")
    
    # Print enrichment stats
    enricher_stats = enricher.get_stats()
    found = enricher_stats.get('found', 0)
    not_found = enricher_stats.get('not_found', 0)
    no_exp = enricher_stats.get('no_experience_data', 0)
    
    print(f"   Found in Snowflake: {found:,}")
    print(f"   Not found: {not_found:,}")
    print(f"   No experience data: {no_exp:,}")
    
    # Prepare records with enrichment data for matching
    records_with_enrichment = []
    for url, record in url_to_record.items():
        enrichment_result = all_enrichment_results.get(url)
        if enrichment_result and enrichment_result.success:
            records_with_enrichment.append((record, enrichment_result.data))
        else:
            error = enrichment_result.error if enrichment_result else 'URL not queried'
            records_with_enrichment.append((record, {'error': error}))
    
    # Run matching
    print(f"\n🎯 Running matching ({'with LLM' if use_llm else 'programmatic only'})...")
    
    if use_llm:
        matcher = AsyncHybridMatcher(use_llm=True, model=llm_model)
        results, false_positives, ambiguous_cases, llm_errors = await run_matching_async(
            records_with_enrichment=records_with_enrichment,
            matcher=matcher,
            concurrency=concurrency,
        )
        match_stats = matcher.get_stats()
    else:
        results, false_positives, ambiguous_cases, llm_errors, match_stats = run_matching_sync(
            records_with_enrichment=records_with_enrichment,
            use_llm=False,
            llm_model=llm_model,
        )
    
    elapsed_time = time.time() - start_time
    
    # Print results summary
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print("=" * 70)
    
    total_processed = len(results)
    enrichment_success = sum(1 for r in results if r.get('enrichment_success'))
    enrichment_failed = total_processed - enrichment_success
    
    print(f"\n📊 Summary:")
    print(f"   Total processed: {total_processed:,}")
    if total_processed > 0:
        print(f"   Enrichment success: {enrichment_success:,} ({enrichment_success/total_processed*100:.1f}%)")
    print(f"   Enrichment failed: {enrichment_failed:,}")
    print(f"\n   FALSE POSITIVES FOUND: {len(false_positives):,}")
    if total_processed > 0:
        fp_rate = len(false_positives) / total_processed * 100
        print(f"   False positive rate: {fp_rate:.1f}%")
    print(f"   Ambiguous cases: {len(ambiguous_cases):,}")
    
    if llm_errors:
        print(f"\n   ⚠️  LLM ERRORS: {len(llm_errors):,}")
    
    print(f"\n   Matching breakdown:")
    print(f"     Programmatic matches: {match_stats.get('programmatic_matches', 0)}")
    print(f"     LLM matches: {match_stats.get('llm_matches', 0)}")
    print(f"     No matches: {match_stats.get('no_matches', 0)}")
    
    print(f"\n   Performance:")
    print(f"     Total time: {elapsed_time/60:.1f} minutes")
    if elapsed_time > 0:
        print(f"     Throughput: {total_processed/elapsed_time:.1f} records/sec")
    
    # Save output files
    if false_positives:
        fp_file = output_path / "false_positives.csv"
        save_csv(false_positives, fp_file)
        print(f"\n💾 Saved false positives: {fp_file}")
    
    if ambiguous_cases:
        amb_file = output_path / "ambiguous_cases.csv"
        save_csv(ambiguous_cases, amb_file)
        print(f"💾 Saved ambiguous cases: {amb_file}")
    
    if results:
        all_file = output_path / "all_results.csv"
        save_csv(results, all_file)
        print(f"💾 Saved all results: {all_file}")
    
    if skipped_records:
        skip_file = output_path / "skipped_records.csv"
        save_csv(skipped_records, skip_file)
        print(f"💾 Saved skipped records: {skip_file}")
    
    # Generate report
    report_path = generate_report(
        results=results,
        false_positives=false_positives,
        ambiguous_cases=ambiguous_cases,
        llm_errors=llm_errors,
        skipped_records=skipped_records,
        match_stats=match_stats,
        enricher_stats=enricher_stats,
        elapsed_time=elapsed_time,
        output_path=output_path,
        timestamp=timestamp,
        test_mode=test_mode,
    )
    print(f"📊 Saved analysis report: {report_path}")
    
    # Close enricher connection
    enricher.close()
    
    print(f"\n{'=' * 70}")
    print("  SNOWFLAKE ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Auto-combine with emergency results
    new_results_path = str(output_path / "all_results.csv")
    
    if auto_combine and os.path.exists(emergency_results) and os.path.exists(new_results_path):
        print(f"\n🔗 Auto-combining with emergency stop results...")
        print(f"   Emergency results: {emergency_results}")
        print(f"   New results: {new_results_path}")
        
        try:
            combine_results(
                emergency_results_path=emergency_results,
                new_results_path=new_results_path,
                output_dir=output_dir,
            )
        except Exception as e:
            print(f"\n   ⚠️  Auto-combine failed: {e}")
            print(f"   You can run manually: python combine_results.py -e '{emergency_results}' -n '{new_results_path}'")
    elif auto_combine:
        if not os.path.exists(emergency_results):
            print(f"\n⚠️  Auto-combine skipped: Emergency results not found at {emergency_results}")
        if not os.path.exists(new_results_path):
            print(f"\n⚠️  Auto-combine skipped: New results not generated")
    else:
        print(f"\n📝 To combine with emergency results later:")
        print(f"   python combine_results.py -e '{emergency_results}' -n '{new_results_path}'")
    
    return {
        'total_processed': total_processed,
        'false_positives': len(false_positives),
        'ambiguous': len(ambiguous_cases),
        'skipped': len(skipped_records),
        'output_path': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="NLAA False Positive Analyzer using Snowflake enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run on remaining records
  python run_snowflake_analysis.py

  # Test with 1000 records
  python run_snowflake_analysis.py --test --test-size 1000

  # Full run without auto-combining
  python run_snowflake_analysis.py --no-combine

  # Use MixRank API fallback for missing Snowflake data
  python run_snowflake_analysis.py --mixrank-fallback
        """
    )
    
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT,
                        help=f'Input CSV file (default: {DEFAULT_INPUT})')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--emergency-results', '-e', default=DEFAULT_EMERGENCY_RESULTS,
                        help=f'Path to emergency eject results for combining (default: {DEFAULT_EMERGENCY_RESULTS})')
    parser.add_argument('--test', '-t', action='store_true',
                        help='Test mode with sampled records')
    parser.add_argument('--test-size', type=int, default=1000,
                        help='Number of records to sample in test mode (default: 1000)')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Snowflake batch query size (default: 10000)')
    parser.add_argument('--concurrency', type=int, default=20,
                        help='Concurrent LLM calls (default: 20)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM matching (programmatic only)')
    parser.add_argument('--llm-model', default='gpt-4o-mini',
                        help='LLM model for matching (default: gpt-4o-mini)')
    parser.add_argument('--no-combine', action='store_true',
                        help='Skip auto-combining with emergency results')
    parser.add_argument('--mixrank-fallback', action='store_true',
                        help='Enable MixRank API fallback for URLs not in Snowflake')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    try:
        results = asyncio.run(run_analysis(
            input_file=args.input,
            output_dir=args.output,
            emergency_results=args.emergency_results,
            test_mode=args.test,
            test_size=args.test_size,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            use_llm=not args.no_llm,
            llm_model=args.llm_model,
            auto_combine=not args.no_combine,
            enable_mixrank_fallback=args.mixrank_fallback,
        ))
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted!")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

