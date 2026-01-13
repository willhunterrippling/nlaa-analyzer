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

from url_normalizer import normalize_linkedin_url, NormalizationResult
from checkpoint import CheckpointManager
from enrichment import MixRankEnricher, ProgressTracker
from matching import HybridMatcher, MatchConfidence, MatchResult

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
    llm_model: str = "gpt-5-mini"
):
    """
    Main analysis pipeline.
    
    Steps:
    1. Load and preprocess data
    2. Enrich via MixRank (with checkpointing)
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
    
    # Initialize enricher
    if not match_only:
        print(f"\n🔗 Initializing MixRank enricher (rate: {rate_limit} req/sec)")
        try:
            enricher = MixRankEnricher(requests_per_second=rate_limit)
        except ValueError as e:
            print(f"   ✗ Error: {e}")
            sys.exit(1)
    
    # Initialize matcher
    print(f"\n🎯 Initializing matcher (LLM: {'enabled' if use_llm else 'disabled'})")
    matcher = HybridMatcher(use_llm=use_llm, model=llm_model)
    
    # Process records
    print(f"\n{'=' * 70}")
    print("  PROCESSING")
    print("=" * 70)
    
    results = []
    false_positives = []
    ambiguous_cases = []
    
    # Progress tracker
    tracker = ProgressTracker(
        total=len(valid_records),
        description="Enriching & Matching",
        update_interval=1 if test_mode else 10,
        verbose=verbose
    )
    
    for i, record in enumerate(valid_records):
        record_id = record.get('ID', f'row_{i}')
        account_name = record.get('ACCOUNT_NAME', '')
        linkedin_url = record.get('_normalized_url', '')
        
        # Skip if already processed (resume mode)
        if checkpoint_mgr.is_processed(record_id):
            cached = checkpoint_mgr.get_cached_enrichment(record_id)
            if cached:
                # Re-run matching on cached data
                experiences = cached.get('experience', [])
                match_result = matcher.match(account_name, experiences)
                
                if match_result.matched:
                    false_positives.append({
                        **record,
                        **format_experience_for_output(match_result.matching_experience),
                        'match_confidence': match_result.confidence.value,
                        'match_reason': match_result.match_reason,
                    })
            
            tracker.update(success=True, skipped=True, message=f"[CACHED] {record_id}")
            continue
        
        # Enrich profile
        enriched_data = None
        enrichment_error = None
        
        if match_only:
            # Try to get from cache
            enriched_data = checkpoint_mgr.get_cached_enrichment(record_id)
            if not enriched_data:
                tracker.update(success=False, message=f"No cached data for {record_id}")
                continue
        else:
            # Call MixRank API
            enrich_result = enricher.enrich_profile(linkedin_url)
            
            if enrich_result.success:
                enriched_data = enrich_result.data
            else:
                enrichment_error = enrich_result.error
        
        # Process result
        if enriched_data:
            experiences = enriched_data.get('experience', [])
            
            # Match account name to experiences
            match_result = matcher.match(account_name, experiences)
            
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
            
            # Record checkpoint
            checkpoint_mgr.record_processed(
                record_id=record_id,
                success=True,
                enriched_data=enriched_data
            )
            
            tracker.update(
                success=True,
                message=f"{'✓ MATCH' if match_result.matched else '✗ No match'}: {account_name[:30]}"
            )
        
        else:
            # Enrichment failed
            result_record = {
                **record,
                'enrichment_success': False,
                'enrichment_error': enrichment_error,
            }
            results.append(result_record)
            
            checkpoint_mgr.record_processed(
                record_id=record_id,
                success=False
            )
            
            tracker.update(
                success=False,
                message=f"Enrichment failed: {enrichment_error[:50] if enrichment_error else 'Unknown'}"
            )
    
    # Force final checkpoint save
    checkpoint_mgr.save_checkpoint(force=True)
    tracker.finish()
    
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
            llm_model=args.llm_model
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

