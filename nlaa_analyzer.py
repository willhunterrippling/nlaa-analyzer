#!/usr/bin/env python3
"""
NLAA (No Longer At Account) False Positive Analyzer

Processes SFDC leads flagged as NLAA, enriches via MixRank API,
and identifies false positives (people still at their account).

Usage:
    # Test mode (50 random records, verbose output)
    python nlaa_analyzer.py --test --verbose

    # Full run with checkpointing
    python nlaa_analyzer.py --input data.csv --checkpoint-interval 100

    # Resume from checkpoint
    python nlaa_analyzer.py --resume

    # Re-run matching only (skip enrichment, use cache)
    python nlaa_analyzer.py --match-only
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import re

from url_normalizer import normalize_linkedin_url, NormalizationResult
from checkpoint import CheckpointManager
from enrichment import MixRankEnricher, ProgressTracker
from matching import HybridMatcher, AsyncHybridMatcher, MatchConfidence, MatchResult
from snowflake_enrichment import SnowflakeEnricher, HybridEnricher, AsyncHybridEnricher


def load_cached_ids_fast(cache_file: str, show_progress: bool = True) -> set[str]:
    """Fast scan of cache file to get all record IDs (without full JSON parse)."""
    ids = set()
    if not os.path.exists(cache_file):
        return ids
    
    # Get file size for progress
    file_size = os.path.getsize(cache_file)
    bytes_read = 0
    last_pct = -1
    
    with open(cache_file, 'r') as f:
        for line in f:
            bytes_read += len(line.encode('utf-8'))
            # Fast extraction without full JSON parse
            match = re.search(r'"id":\s*"([^"]+)"', line)
            if match:
                ids.add(match.group(1))
            
            # Progress update every 5%
            if show_progress and file_size > 0:
                pct = int(bytes_read / file_size * 100)
                if pct >= last_pct + 5:
                    print(f"\r   Scanning cache... {pct}% ({len(ids):,} records)", end="", flush=True)
                    last_pct = pct
    
    if show_progress:
        print()  # Newline after progress
    return ids

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
    """
    Preprocess a single record.
    
    Returns:
        (processed_record, error_message or None)
    """
    # Check required fields
    account_name = record.get('ACCOUNT_NAME', '').strip()
    linkedin_url = record.get('CONTACT_LINKED_IN_URL_C', '').strip()
    record_id = record.get('ID', '')
    
    if not account_name:
        return record, "Missing ACCOUNT_NAME"
    
    if not linkedin_url or linkedin_url in ('\\N', 'NULL', 'null'):
        return record, "Missing LinkedIn URL"
    
    # Normalize LinkedIn URL
    url_result = normalize_linkedin_url(linkedin_url)
    
    if not url_result.success:
        return record, f"Invalid LinkedIn URL: {url_result.error}"
    
    # Update record with normalized URL
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


def run_analysis(
    input_file: str,
    output_dir: str,
    test_mode: bool = False,
    test_size: int = 50,
    verbose: bool = False,
    resume: bool = False,
    match_only: bool = False,
    checkpoint_interval: int = 100,
    rate_limit: float = 2.0,
    use_llm: bool = True,
    llm_model: str = "gpt-5-mini",
    use_snowflake: bool = False,
    snowflake_batch_size: int = 10000,
    mixrank_fallback: bool = True,
):
    """
    Main analysis pipeline.
    
    Steps:
    1. Load and preprocess data
    2. Enrich via MixRank API or Snowflake (with checkpointing)
    3. Match account names to experiences
    4. Output results
    """
    
    print("\n" + "=" * 70)
    print("  NLAA FALSE POSITIVE ANALYZER")
    print("=" * 70)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_interval=checkpoint_interval
    )
    
    # Load existing checkpoint if resuming
    if resume:
        if checkpoint_mgr.load_checkpoint():
            print(f"\n✓ Resuming from checkpoint")
            progress = checkpoint_mgr.get_progress()
            print(f"  Previously processed: {progress['processed']}/{progress['total']}")
            print(f"  Success: {progress['success']}, Errors: {progress['errors']}")
        else:
            print("\n⚠ No checkpoint found, starting fresh")
            resume = False
    
    # Load cached IDs from cache file (source of truth for what's already enriched)
    cache_file = Path(CHECKPOINT_DIR) / "enrichment_cache.jsonl"
    cached_record_ids = set()
    if cache_file.exists():
        cache_size_mb = cache_file.stat().st_size / (1024 * 1024)
        print(f"\n📂 Loading enrichment cache ({cache_size_mb:.0f} MB)...")
        import time as _time
        _start = _time.time()
        cached_record_ids = load_cached_ids_fast(str(cache_file))
        print(f"   ✓ Scanned {len(cached_record_ids):,} record IDs ({_time.time()-_start:.1f}s)")
        
        # Ensure cache is loaded for data lookup (even if not formally resuming)
        if cached_record_ids and not resume:
            _start2 = _time.time()
            checkpoint_mgr.enrichment_cache = checkpoint_mgr._load_jsonl_cache()
            print(f"   ✓ Loaded full cache data ({_time.time()-_start2:.1f}s)")
    
    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    all_records = load_csv_data(input_file)
    print(f"   Total records: {len(all_records):,}")
    
    # Sample for test mode
    if test_mode:
        print(f"\n🧪 TEST MODE: Sampling {test_size} random records")
        records = sample_records(all_records, test_size)
    else:
        records = all_records
    
    # Initialize checkpoint for new run
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
            if verbose:
                print(f"   ⊘ Skipping {record.get('ID', '?')}: {error}")
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
    
    # Initialize enricher (lazy for Snowflake - only connect if needed)
    enricher = None
    snowflake_initialized = False
    is_async_enricher = False
    
    def get_hybrid_enricher():
        nonlocal enricher, snowflake_initialized, is_async_enricher
        if snowflake_initialized:
            return enricher
        
        fallback_msg = "with async MixRank fallback (50 concurrent)" if mixrank_fallback else "Snowflake only"
        print(f"\n🔗 Initializing async hybrid enricher ({fallback_msg})")
        print(f"   Snowflake batch size: {snowflake_batch_size}")
        
        try:
            # Use async enricher for concurrent MixRank calls
            enricher = AsyncHybridEnricher(
                snowflake_batch_size=snowflake_batch_size,
                mixrank_max_concurrent=50,  # MixRank allows up to 100
                enable_mixrank_fallback=mixrank_fallback,
            )
            is_async_enricher = True
            success, msg = enricher.test_connection()
            if success:
                print(f"   ✓ {msg}")
            else:
                print(f"   ✗ {msg}")
                enricher = None
        except Exception as e:
            print(f"   ✗ Error: {e}")
            enricher = None
        snowflake_initialized = True
        return enricher
    
    if not match_only:
        if use_snowflake:
            # Check if we need Snowflake at all (skip if all records cached)
            uncached_count = sum(1 for r in valid_records if r.get('ID', '') not in cached_record_ids)
            if uncached_count > 0:
                print(f"\n🔗 {uncached_count:,} records need enrichment")
                enricher = get_hybrid_enricher()
                if not enricher:
                    print("   ✗ Failed to connect - will use cache only")
            else:
                print(f"\n✓ All {len(valid_records):,} records found in cache - enrichment not needed")
        else:
            print(f"\n🔗 Initializing MixRank enricher (rate: {rate_limit} req/sec)")
            try:
                enricher = MixRankEnricher(requests_per_second=rate_limit)
            except ValueError as e:
                print(f"   ✗ Error: {e}")
                sys.exit(1)
    
    # Initialize matcher (use async for LLM calls)
    print(f"\n🎯 Initializing matcher (LLM: {'enabled' if use_llm else 'disabled'})")
    matcher = AsyncHybridMatcher(use_llm=use_llm, model=llm_model) if use_llm else HybridMatcher(use_llm=False)
    
    # Process records
    print(f"\n{'=' * 70}")
    print("  PROCESSING")
    print("=" * 70)
    
    results = []
    false_positives = []
    ambiguous_cases = []
    
    # ============================================================
    # PHASE 1: Batch enrichment for uncached records
    # ============================================================
    # Collect all URLs that need enrichment (not in cache)
    urls_to_enrich = {}
    records_from_cache = {}
    
    for i, record in enumerate(valid_records):
        record_id = record.get('ID', f'row_{i}')
        linkedin_url = record.get('_normalized_url', '')
        
        if record_id in cached_record_ids:
            # Will use cache
            records_from_cache[record_id] = True
        elif not match_only and use_snowflake and linkedin_url:
            # Need enrichment
            urls_to_enrich[linkedin_url] = record_id
    
    # Batch enrich all uncached URLs
    enrichment_results = {}
    if urls_to_enrich and enricher:
        print(f"\n📊 Batch enriching {len(urls_to_enrich):,} uncached URLs...")
        
        if is_async_enricher:
            # Run async enricher
            import asyncio
            enrichment_results = asyncio.run(enricher.enrich_batch(urls_to_enrich))
        else:
            # Sync enricher
            enrichment_results = enricher.enrich_batch(urls_to_enrich)
        
        print(f"   ✓ Enrichment complete")
    
    # ============================================================
    # PHASE 2: Gather enrichment data for all records
    # ============================================================
    print("\n📦 Gathering enrichment data...")
    records_to_match = []  # List of (record, enriched_data)
    enrichment_failures = []  # Track failed enrichments
    
    for i, record in enumerate(valid_records):
        record_id = record.get('ID', f'row_{i}')
        linkedin_url = record.get('_normalized_url', '')
        
        # Get enrichment data from various sources
        enriched_data = None
        enrichment_error = None
        
        # Check cache first
        if record_id in cached_record_ids:
            enriched_data = checkpoint_mgr.get_cached_enrichment(record_id)
        elif match_only:
            enriched_data = checkpoint_mgr.get_cached_enrichment(record_id)
        elif use_snowflake:
            # Get from pre-fetched results
            enrich_result = enrichment_results.get(linkedin_url)
            if enrich_result and enrich_result.success:
                enriched_data = enrich_result.data
            elif enrich_result:
                enrichment_error = enrich_result.error
        else:
            # Direct MixRank API call (sync) - only for non-batched fallback
            enrich_result = enricher.enrich_profile(linkedin_url)
            if enrich_result.success:
                enriched_data = enrich_result.data
            else:
                enrichment_error = enrich_result.error
        
        if enriched_data:
            records_to_match.append((record, enriched_data))
        elif enrichment_error:
            # Track failures
            enrichment_failures.append({
                **record,
                'enrichment_success': False,
                'enrichment_error': enrichment_error,
            })
    
    print(f"   ✓ {len(records_to_match)} records ready for matching")
    if enrichment_failures:
        print(f"   ⚠ {len(enrichment_failures)} records failed enrichment")
    
    # ============================================================
    # PHASE 3: Batch matching with async LLM calls
    # ============================================================
    import asyncio
    
    async def batch_match_all():
        """Run all matching in a single async batch for efficiency."""
        match_tasks = []
        
        for record, enriched_data in records_to_match:
            account_name = record.get('ACCOUNT_NAME', '')
            experiences = enriched_data.get('experience', [])
            
            if use_llm:
                # Create async task for each match
                match_tasks.append(matcher.match(account_name, experiences))
            else:
                # Sync matching doesn't need async
                match_tasks.append(asyncio.coroutine(lambda r=matcher.match(account_name, experiences): r)())
        
        if use_llm:
            # Run all LLM calls concurrently
            return await asyncio.gather(*match_tasks)
        else:
            # For sync matching, just run sequentially
            return [matcher.match(r.get('ACCOUNT_NAME', ''), e.get('experience', [])) 
                    for r, e in records_to_match]
    
    print("\n🎯 Running batch matching...")
    match_start = time.time()
    
    if use_llm:
        match_results = asyncio.run(batch_match_all())
    else:
        match_results = [matcher.match(r.get('ACCOUNT_NAME', ''), e.get('experience', [])) 
                        for r, e in records_to_match]
    
    match_elapsed = time.time() - match_start
    print(f"   ✓ Matching complete in {match_elapsed:.1f}s ({len(match_results)/max(match_elapsed, 0.1):.1f} records/sec)")
    
    # ============================================================
    # PHASE 4: Process match results
    # ============================================================
    tracker = ProgressTracker(
        total=len(records_to_match),
        description="Processing",
        update_interval=1 if test_mode else 10,
        verbose=verbose
    )
    
    for i, ((record, enriched_data), match_result) in enumerate(zip(records_to_match, match_results)):
        record_id = record.get('ID', f'row_{i}')
        account_name = record.get('ACCOUNT_NAME', '')
        experiences = enriched_data.get('experience', [])
        
        # Build result record
        result_record = {
            **record,
            'enrichment_success': True,
            'experience_count': len(experiences),
            'current_experience_count': len([e for e in experiences if e.get('is_current') or e.get('end_date') is None]),
            'matched': match_result.matched,
            'match_confidence': match_result.confidence.value,
            'match_reason': match_result.match_reason,
            'all_current_companies': ', '.join(c for c in match_result.all_current_companies if c),
            **format_experience_for_output(match_result.matching_experience),
        }
        
        results.append(result_record)
        
        # Categorize
        if match_result.matched:
            false_positives.append(result_record)
            
            if verbose:
                print(f"\n   ✓ FALSE POSITIVE FOUND:")
                print(f"     ID: {record_id}")
                print(f"     Account: {account_name}")
                print(f"     Matched: {match_result.matching_experience.get('company', 'N/A')}")
                print(f"     Confidence: {match_result.confidence.value}")
                print(f"     Reason: {match_result.match_reason}")
        
        elif match_result.confidence == MatchConfidence.LOW:
            ambiguous_cases.append(result_record)
        
        # Record checkpoint (only for new records not already in cache)
        if record_id not in cached_record_ids:
            checkpoint_mgr.record_processed(
                record_id=record_id,
                success=True,
                enriched_data=enriched_data
            )
        
        tracker.update(
            success=True,
            message=f"{'✓ MATCH' if match_result.matched else '✗ No match'}: {account_name[:30]}"
        )
    
    # Force final checkpoint save
    checkpoint_mgr.save_checkpoint(force=True)
    tracker.finish()
    
    # Add enrichment failures to results
    results.extend(enrichment_failures)
    
    # Export URLs not found in Snowflake (if using hybrid enricher)
    snowflake_missing_count = 0
    if use_snowflake and enricher and hasattr(enricher, 'export_snowflake_missing'):
        enricher.print_stats()
        snowflake_missing_file = output_path / f"snowflake_missing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        snowflake_missing_count = enricher.export_snowflake_missing(str(snowflake_missing_file))
        if snowflake_missing_count > 0:
            print(f"\n💾 Exported {snowflake_missing_count:,} URLs not found in Snowflake: {snowflake_missing_file}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - tracker.start_time
    
    # Output results
    print(f"\n{'=' * 70}")
    print("  RESULTS")
    print("=" * 70)
    
    # Summary statistics
    total_processed = len(results)
    enrichment_success = sum(1 for r in results if r.get('enrichment_success'))
    enrichment_failed = total_processed - enrichment_success
    
    print(f"\n📊 Summary:")
    print(f"   Total processed: {total_processed:,}")
    print(f"   Enrichment success: {enrichment_success:,} ({enrichment_success/total_processed*100:.1f}%)" if total_processed > 0 else "")
    print(f"   Enrichment failed: {enrichment_failed:,}")
    print(f"\n   FALSE POSITIVES FOUND: {len(false_positives):,}")
    if total_processed > 0:
        fp_rate = len(false_positives) / total_processed * 100
        print(f"   False positive rate: {fp_rate:.1f}%")
    print(f"   Ambiguous cases: {len(ambiguous_cases):,}")
    
    # Matching stats
    match_stats = matcher.get_stats()
    print(f"\n   Matching breakdown:")
    print(f"     Programmatic matches: {match_stats['programmatic_matches']}")
    print(f"     LLM matches: {match_stats['llm_matches']}")
    print(f"     LLM calls made: {match_stats['llm_calls']}")
    print(f"     No matches: {match_stats['no_matches']}")
    
    # Save output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # False positives CSV
    if false_positives:
        fp_file = output_path / f"false_positives_{timestamp}.csv"
        save_csv(false_positives, fp_file)
        print(f"\n💾 Saved false positives: {fp_file}")
    
    # Ambiguous cases CSV
    if ambiguous_cases:
        amb_file = output_path / f"ambiguous_cases_{timestamp}.csv"
        save_csv(ambiguous_cases, amb_file)
        print(f"💾 Saved ambiguous cases: {amb_file}")
    
    # Full results (for debugging in test mode)
    if test_mode and results:
        all_file = output_path / f"all_results_{timestamp}.csv"
        save_csv(results, all_file)
        print(f"💾 Saved all results: {all_file}")
    
    # Skipped records
    if skipped_records:
        skip_file = output_path / f"skipped_records_{timestamp}.csv"
        save_csv(skipped_records, skip_file)
        print(f"💾 Saved skipped records: {skip_file}")
    
    # Generate markdown report
    report_path = generate_report(
        results=results,
        false_positives=false_positives,
        ambiguous_cases=ambiguous_cases,
        skipped_records=skipped_records,
        match_stats=match_stats,
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
        'snowflake_missing': snowflake_missing_count,
    }


def save_csv(records: list[dict], filepath: Path):
    """Save records to CSV file."""
    if not records:
        return
    
    # Get all unique keys
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    
    # Prioritize certain columns first
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
    skipped_records: list[dict],
    match_stats: dict,
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
    
    # Calculate confidence breakdown
    high_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'high')
    medium_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'medium')
    low_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'low')
    
    # Get sample false positives for the report
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
| High | {high_conf:,} | {high_conf/len(false_positives)*100:.1f}% |
| Medium | {medium_conf:,} | {medium_conf/len(false_positives)*100:.1f}% |
| Low | {low_conf:,} | {low_conf/len(false_positives)*100:.1f}% |

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
    
    # Count skip reasons
    skip_reasons = {}
    for r in skipped_records:
        reason = r.get('_skip_reason', 'Unknown')
        # Simplify reason
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

- `false_positives_{timestamp}.csv` - All {len(false_positives):,} confirmed false positives
- `ambiguous_cases_{timestamp}.csv` - {len(ambiguous_cases):,} cases needing review
- `all_results_{timestamp}.csv` - Complete results{"" if not test_mode else " (test mode only)"}
- `skipped_records_{timestamp}.csv` - {len(skipped_records):,} records that couldn't be processed

---

## Recommendations

1. **Review false positives** - These {len(false_positives):,} leads should have their NLAA flag removed
2. **Investigate ambiguous cases** - {len(ambiguous_cases):,} records need manual review
3. **Fix data quality issues** - {len(skipped_records):,} records have invalid/missing LinkedIn URLs

---

*Report generated by NLAA False Positive Analyzer*
"""
    
    report_path = output_path / f"analysis_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="NLAA False Positive Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 50 random records (verbose)
  python nlaa_analyzer.py --test --verbose

  # Full run
  python nlaa_analyzer.py --input data.csv

  # Resume interrupted run
  python nlaa_analyzer.py --resume

  # Re-run matching without API calls
  python nlaa_analyzer.py --match-only
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default=DEFAULT_INPUT,
        help=f'Input CSV file (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--test', '-t',
        action='store_true',
        help='Test mode: process only 50 random records'
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=50,
        help='Number of records for test mode (default: 50)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output (show each match decision)'
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--match-only',
        action='store_true',
        help='Skip enrichment, only run matching on cached data'
    )
    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=100,
        help='Save checkpoint every N records (default: 100)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=2.0,
        help='API requests per second (default: 2.0)'
    )
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM fallback (programmatic matching only)'
    )
    parser.add_argument(
        '--llm-model',
        default='gpt-5-mini',
        help='LLM model for fallback matching (default: gpt-5-mini)'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Clear existing checkpoint and start fresh'
    )
    parser.add_argument(
        '--snowflake',
        action='store_true',
        help='Use Snowflake for enrichment (with MixRank API fallback for missing data)'
    )
    parser.add_argument(
        '--snowflake-batch-size',
        type=int,
        default=10000,
        help='Batch size for Snowflake queries (default: 10000)'
    )
    parser.add_argument(
        '--no-mixrank-fallback',
        action='store_true',
        help='Disable MixRank API fallback when using Snowflake (Snowflake only)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint_mgr = CheckpointManager(checkpoint_dir=CHECKPOINT_DIR)
        checkpoint_mgr.clear_checkpoint()
        print("✓ Checkpoint cleared")
    
    # Run analysis
    try:
        results = run_analysis(
            input_file=args.input,
            output_dir=args.output,
            test_mode=args.test,
            test_size=args.test_size,
            verbose=args.verbose,
            resume=args.resume,
            match_only=args.match_only,
            checkpoint_interval=args.checkpoint_interval,
            rate_limit=args.rate_limit,
            use_llm=not args.no_llm,
            llm_model=args.llm_model,
            use_snowflake=args.snowflake,
            snowflake_batch_size=args.snowflake_batch_size,
            mixrank_fallback=not args.no_mixrank_fallback,
        )
        
        # Exit code based on results
        if results['false_positives'] > 0:
            sys.exit(0)  # Success - found false positives
        else:
            sys.exit(0)  # Success - no false positives (still valid result)
            
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

