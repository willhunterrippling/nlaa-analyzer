#!/usr/bin/env python3
"""
NLAA (No Longer At Account) False Positive Analyzer - Async Version

High-throughput processing using async/parallel API calls.
Achieves 10-20+ req/sec vs ~1.5 req/sec sequential.

Usage:
    # Test mode (500 random records)
    python nlaa_analyzer_async.py --test --test-size 500

    # Full run with 20 concurrent requests
    python nlaa_analyzer_async.py --concurrency 20

    # Resume from checkpoint
    python nlaa_analyzer_async.py --resume
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from url_normalizer import normalize_linkedin_url
from checkpoint import AsyncCheckpointManager, CheckpointManager
from enrichment_async import AsyncMixRankEnricher, AsyncProgressTracker
from matching import AsyncHybridMatcher, MatchConfidence

load_dotenv()


# Default paths
DEFAULT_INPUT = "inputs/01c1b4ef-070a-972b-5342-83029a156523.csv"
DEFAULT_OUTPUT_DIR = "output"
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
    
    sample_fps = false_positives[:10] if false_positives else []
    
    report = f"""# NLAA False Positive Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Mode:** {"Test Mode" if test_mode else "Full Run"}

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

## Enrichment Results

| Status | Count | Percentage |
|--------|-------|------------|
| Success | {enrichment_success:,} | {enrichment_success/total_processed*100:.1f}% |
| Failed | {enrichment_failed:,} | {enrichment_failed/total_processed*100:.1f}% |

---

## Matching Breakdown

### By Method

| Method | Matches |
|--------|---------|
| Programmatic | {match_stats.get('programmatic_matches', 0):,} |
| LLM Fallback | {match_stats.get('llm_matches', 0):,} |
| No Match | {match_stats.get('no_matches', 0):,} |

**LLM Calls Made:** {match_stats.get('llm_calls', 0):,} ({match_stats.get('llm_calls', 0)/total_processed*100:.1f}% escalation rate)

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

**{len(llm_errors):,} records failed LLM matching** and need to be re-processed!

These records had ambiguous programmatic matches that were escalated to the LLM for decision, but the LLM API call failed.

| Account Name | Current Companies | Error |
|--------------|-------------------|-------|
"""
        for err in llm_errors[:5]:
            account = err.get('ACCOUNT_NAME', 'N/A')[:35]
            companies = err.get('all_current_companies', 'N/A')[:35]
            reason = err.get('match_reason', 'N/A')
            # Extract just the error message
            if 'LLM error:' in reason:
                reason = reason.replace('LLM error: ', '')[:60]
            report += f"| {account} | {companies} | {reason}... |\n"
        
        if len(llm_errors) > 5:
            report += f"\n*Showing 5 of {len(llm_errors):,} LLM errors.*\n"
        
        report += """
**Action Required:** Fix the LLM configuration issue and re-run these records.
"""

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
    
    # Issues & Errors section
    total_requests = enricher_stats.get('total_requests', 0)
    successful = enricher_stats.get('successful', 0)
    failed = enricher_stats.get('failed', 0)
    retries = enricher_stats.get('retries', 0)
    
    report += f"""
---

## Issues & Errors

### API Performance

| Metric | Count |
|--------|-------|
| Total API Requests | {total_requests:,} |
| Successful | {successful:,} |
| Failed | {failed:,} |
| Retries (rate limits/timeouts) | {retries:,} |

"""
    
    if retries > 0:
        report += f"""**⚠️ {retries} retries occurred** - This may indicate rate limiting. Consider reducing concurrency if this number is high.

"""
    
    if failed > 0:
        # Categorize errors from results
        error_types = {}
        for r in results:
            if not r.get('enrichment_success'):
                error = r.get('enrichment_error', 'Unknown error')
                # Simplify error messages
                if '429' in str(error):
                    error_type = 'Rate Limited (429)'
                elif '404' in str(error):
                    error_type = 'Profile Not Found (404)'
                elif '500' in str(error) or '502' in str(error) or '503' in str(error):
                    error_type = 'Server Error (5xx)'
                elif 'timeout' in str(error).lower():
                    error_type = 'Timeout'
                else:
                    error_type = 'Other'
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        report += "### Error Breakdown\n\n"
        report += "| Error Type | Count |\n|------------|-------|\n"
        for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
            report += f"| {error_type} | {count:,} |\n"
        report += "\n"
    else:
        report += "**✓ No errors encountered during enrichment.**\n"
    
    report += f"""
---

## Output Files

- `false_positives.csv` - All {len(false_positives):,} confirmed false positives
- `ambiguous_cases.csv` - {len(ambiguous_cases):,} cases needing review
- `all_results.csv` - Complete results{"" if not test_mode else " (test mode only)"}
- `skipped_records.csv` - {len(skipped_records):,} records that couldn't be processed

---

## Recommendations

1. **Review false positives** - These {len(false_positives):,} leads should have their NLAA flag removed
2. **Investigate ambiguous cases** - {len(ambiguous_cases):,} records need manual review
"""
    
    if llm_errors:
        report += f"""3. **⚠️ Fix LLM errors** - {len(llm_errors):,} records failed LLM matching and need re-processing
"""
    
    report += f"""{"4" if llm_errors else "3"}. **Fix data quality issues** - {len(skipped_records):,} records have invalid/missing LinkedIn URLs

---

*Report generated by NLAA False Positive Analyzer (Async)*
"""
    
    report_path = output_path / f"analysis_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


async def run_analysis_async(
    input_file: str,
    output_dir: str,
    test_mode: bool = False,
    test_size: int = 500,
    verbose: bool = False,
    resume: bool = False,
    match_only: bool = False,
    checkpoint_interval: int = 100,
    concurrency: int = 20,
    use_llm: bool = True,
    llm_model: str = "gpt-5-mini"
):
    """
    Main async analysis pipeline.
    """
    
    print("\n" + "=" * 70)
    print("  NLAA FALSE POSITIVE ANALYZER (ASYNC)")
    print("=" * 70)
    
    start_time = time.time()
    
    # Setup output directory with timestamp subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}_{'test' if test_mode else 'full'}_{test_size if test_mode else 'all'}"
    output_path = Path(output_dir) / run_name
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output folder: {output_path}")
    
    # Initialize async checkpoint manager
    checkpoint_mgr = AsyncCheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_interval=checkpoint_interval
    )
    
    if resume:
        if await checkpoint_mgr.load_checkpoint():
            print(f"\n✓ Resuming from checkpoint")
            progress = checkpoint_mgr.get_progress()
            print(f"  Previously processed: {progress['processed']}/{progress['total']}")
        else:
            print("\n⚠ No checkpoint found, starting fresh")
            resume = False
    
    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    all_records = load_csv_data(input_file)
    print(f"   Total records: {len(all_records):,}")
    
    if test_mode:
        print(f"\n🧪 TEST MODE: Sampling {test_size} random records")
        records = sample_records(all_records, test_size)
    else:
        records = all_records
    
    if not resume:
        checkpoint_mgr.initialize(input_file, len(records))
    
    # Preprocess records
    print(f"\n🔧 Preprocessing records...")
    valid_records = []
    skipped_records = []
    url_issues = {"double_url": 0, "wrong_path_structure": 0, "missing_protocol": 0, "missing_domain_prefix": 0}
    
    for record in records:
        processed, error = preprocess_record(record)
        
        if error:
            skipped_records.append({**record, '_skip_reason': error})
        else:
            valid_records.append(processed)
            if processed.get('_url_issue_type'):
                url_issues[processed['_url_issue_type']] += 1
    
    print(f"   Valid records: {len(valid_records):,}")
    print(f"   Skipped: {len(skipped_records):,}")
    
    if any(url_issues.values()):
        print(f"\n   URL fixes applied:")
        for issue, count in url_issues.items():
            if count > 0:
                print(f"     - {issue}: {count}")
    
    # Filter out already processed records
    if resume:
        records_to_process = [
            r for r in valid_records
            if not checkpoint_mgr.is_processed(r.get('ID', ''))
        ]
        print(f"\n   Records remaining after checkpoint: {len(records_to_process):,}")
    else:
        records_to_process = valid_records
    
    # Initialize enricher
    if not match_only:
        print(f"\n🔗 Initializing async MixRank enricher (concurrency: {concurrency})")
        try:
            enricher = AsyncMixRankEnricher(concurrency=concurrency)
        except ValueError as e:
            print(f"   ✗ Error: {e}")
            sys.exit(1)
    
    # Initialize async matcher
    print(f"\n🎯 Initializing async matcher (LLM: {'enabled' if use_llm else 'disabled'})")
    matcher = AsyncHybridMatcher(use_llm=use_llm, model=llm_model)
    
    # Process records
    print(f"\n{'=' * 70}")
    print("  PROCESSING (ASYNC)")
    print("=" * 70)
    
    results = []
    false_positives = []
    ambiguous_cases = []
    llm_errors = []
    enrichment_cache = {}
    
    # Prepare batch for enrichment
    batch = [
        (r.get('ID', f'row_{i}'), r.get('_normalized_url', ''))
        for i, r in enumerate(records_to_process)
    ]
    
    # Create record lookup by ID
    record_by_id = {r.get('ID', f'row_{i}'): r for i, r in enumerate(records_to_process)}
    
    # Progress tracking
    total_to_process = len(batch)
    processed_count = 0
    success_count = 0
    error_count = 0
    
    print(f"\n   Processing {total_to_process:,} records with {concurrency} concurrent requests...")
    
    # Semaphore to limit concurrent matching operations (especially LLM calls)
    match_semaphore = asyncio.Semaphore(concurrency)
    
    # Lock for thread-safe updates to shared state
    state_lock = asyncio.Lock()
    
    async def process_single_result(result):
        """Process a single enrichment result with async matching."""
        nonlocal processed_count, success_count, error_count
        
        record_id = result.record_id
        record = record_by_id.get(record_id)
        
        if not record:
            return None
        
        account_name = record.get('ACCOUNT_NAME', '')
        
        if result.success:
            enrichment_cache[record_id] = result.data
            experiences = result.data.get('experience', [])
            
            # Match account name to experiences (async, non-blocking for LLM calls)
            async with match_semaphore:
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
            
            # Thread-safe updates to shared collections
            async with state_lock:
                results.append(result_record)
                
                if match_result.llm_error:
                    llm_errors.append(result_record)
                elif match_result.matched:
                    false_positives.append(result_record)
                elif match_result.confidence == MatchConfidence.LOW:
                    ambiguous_cases.append(result_record)
                
                success_count += 1
                processed_count += 1
            
            # Checkpoint (outside lock to avoid blocking, async I/O)
            await checkpoint_mgr.record_processed(record_id, success=True, enriched_data=result.data)
        else:
            result_record = {
                **record,
                'enrichment_success': False,
                'enrichment_error': result.error,
            }
            
            async with state_lock:
                results.append(result_record)
                error_count += 1
                processed_count += 1
            
            await checkpoint_mgr.record_processed(record_id, success=False)
        
        return result_record
    
    def print_progress():
        """Print progress bar."""
        elapsed = time.time() - start_time
        rate = processed_count / elapsed if elapsed > 0 else 0
        pct = processed_count / total_to_process * 100 if total_to_process > 0 else 0
        
        bar_width = 40
        filled = int(bar_width * processed_count / total_to_process) if total_to_process > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        eta = (total_to_process - processed_count) / rate if rate > 0 else 0
        eta_str = f"{int(eta//60):02d}:{int(eta%60):02d}"
        elapsed_str = f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"
        
        print(
            f"\r   {pct:5.1f}% |{bar}| {processed_count}/{total_to_process} "
            f"[{elapsed_str}<{eta_str}] ✓{success_count} ✗{error_count} "
            f"FP:{len(false_positives)} ({rate:.1f}/s)",
            end="", flush=True
        )
    
    # Run async enrichment with parallel result processing
    if not match_only and batch:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Create all enrichment tasks
            enrichment_tasks = [
                enricher.enrich_profile(session, record_id, url)
                for record_id, url in batch
            ]
            
            # Process results as they complete, spawning matching tasks in parallel
            pending_match_tasks = set()
            last_progress_update = 0
            
            for coro in asyncio.as_completed(enrichment_tasks):
                enrichment_result = await coro
                
                # Create a matching task (runs in parallel with other matching)
                match_task = asyncio.create_task(process_single_result(enrichment_result))
                pending_match_tasks.add(match_task)
                match_task.add_done_callback(pending_match_tasks.discard)
                
                # Update progress periodically
                current_count = processed_count
                if current_count - last_progress_update >= 10 or current_count == total_to_process:
                    print_progress()
                    last_progress_update = current_count
            
            # Wait for all remaining matching tasks to complete
            if pending_match_tasks:
                await asyncio.gather(*pending_match_tasks, return_exceptions=True)
            
            # Final progress update
            print_progress()
    
    print()  # Newline after progress bar
    
    # Force final checkpoint save
    await checkpoint_mgr.save_checkpoint(force=True)
    
    elapsed_time = time.time() - start_time
    
    # Output results
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
        print(f"\n   ⚠️  LLM ERRORS: {len(llm_errors):,} (these need re-processing!)")
    
    match_stats = matcher.get_stats()
    print(f"\n   Matching breakdown:")
    print(f"     Programmatic matches: {match_stats['programmatic_matches']}")
    print(f"     LLM matches: {match_stats['llm_matches']}")
    print(f"     LLM calls made: {match_stats['llm_calls']}")
    print(f"     LLM errors: {len(llm_errors)}")
    print(f"     No matches: {match_stats['no_matches']}")
    
    print(f"\n   Performance:")
    print(f"     Total time: {elapsed_time/60:.1f} minutes")
    if elapsed_time > 0:
        print(f"     Throughput: {total_processed/elapsed_time:.1f} records/sec")
    
    # Save output files (no timestamp in filenames - it's in the folder name)
    if false_positives:
        fp_file = output_path / "false_positives.csv"
        save_csv(false_positives, fp_file)
        print(f"\n💾 Saved false positives: {fp_file}")
    
    if ambiguous_cases:
        amb_file = output_path / "ambiguous_cases.csv"
        save_csv(ambiguous_cases, amb_file)
        print(f"💾 Saved ambiguous cases: {amb_file}")
    
    if test_mode and results:
        all_file = output_path / "all_results.csv"
        save_csv(results, all_file)
        print(f"💾 Saved all results: {all_file}")
    
    if skipped_records:
        skip_file = output_path / "skipped_records.csv"
        save_csv(skipped_records, skip_file)
        print(f"💾 Saved skipped records: {skip_file}")
    
    # Generate markdown report
    enricher_stats = enricher.get_stats() if not match_only else {"total_requests": 0, "successful": 0, "failed": 0, "retries": 0}
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
        test_mode=test_mode
    )
    print(f"📊 Saved analysis report: {report_path}")
    
    print(f"\n{'=' * 70}")
    print("  COMPLETE")
    print("=" * 70)
    
    return {
        'total_processed': total_processed,
        'false_positives': len(false_positives),
        'ambiguous': len(ambiguous_cases),
        'skipped': len(skipped_records),
    }


def main():
    parser = argparse.ArgumentParser(
        description="NLAA False Positive Analyzer (Async/Parallel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 500 random records
  python nlaa_analyzer_async.py --test --test-size 500

  # Full run with 20 concurrent requests
  python nlaa_analyzer_async.py --concurrency 20

  # Resume interrupted run
  python nlaa_analyzer_async.py --resume
        """
    )
    
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT)
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--test-size', type=int, default=500)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--resume', '-r', action='store_true')
    parser.add_argument('--match-only', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--no-llm', action='store_true')
    parser.add_argument('--llm-model', default='gpt-4o-mini')
    parser.add_argument('--clear-checkpoint', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if args.clear_checkpoint:
        checkpoint_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
        checkpoint_mgr.clear_checkpoint()
        print("✓ Checkpoint cleared")
    
    try:
        results = asyncio.run(run_analysis_async(
            input_file=args.input,
            output_dir=args.output,
            test_mode=args.test,
            test_size=args.test_size,
            verbose=args.verbose,
            resume=args.resume,
            match_only=args.match_only,
            checkpoint_interval=args.checkpoint_interval,
            concurrency=args.concurrency,
            use_llm=not args.no_llm,
            llm_model=args.llm_model
        ))
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted! Progress has been saved to checkpoint.")
        print("  Run with --resume to continue.")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

