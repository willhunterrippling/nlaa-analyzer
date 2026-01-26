#!/usr/bin/env python3
"""
Snowflake Enrichment Test & Validation Script

Tests the Snowflake enrichment module before production use:
1. --test-connection: Verify Snowflake connectivity
2. --validate-cache: Compare Snowflake data to cached MixRank data
3. --test-new: Test on records NOT in cache (failure rate check)
4. --full-test: End-to-end test with matching pipeline

Usage:
    # Test connection
    python test_snowflake_enrichment.py --test-connection
    
    # Validate against cache (100 records from cache)
    python test_snowflake_enrichment.py --validate-cache --test-size 100
    
    # Test new records (100 records NOT in cache)
    python test_snowflake_enrichment.py --test-new --test-size 100
    
    # Full end-to-end test
    python test_snowflake_enrichment.py --full-test --test-size 100
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Import our modules
from snowflake_enrichment import SnowflakeEnricher, EnrichmentResult
from url_normalizer import normalize_linkedin_url, extract_linkedin_slug
from matching import HybridMatcher, MatchConfidence


# Paths
INPUT_FILE = "inputs/01c1b4ef-070a-972b-5342-83029a156523.csv"  # Main input file
CACHE_FILE = "checkpoints/enrichment_cache.jsonl"
CHECKPOINT_FILE = "checkpoints/checkpoint.json"


def load_input_records(filepath: str, limit: Optional[int] = None) -> list[dict]:
    """Load records from input CSV file."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            records.append(row)
    return records


def load_cached_ids_fast(cache_file: str) -> set[str]:
    """Fast scan of cache file to get all record IDs."""
    ids = set()
    if not os.path.exists(cache_file):
        return ids
    
    with open(cache_file, 'r') as f:
        for line in f:
            # Fast extraction without full JSON parse
            match = re.search(r'"id":\s*"([^"]+)"', line)
            if match:
                ids.add(match.group(1))
    return ids


def load_cached_records(cache_file: str, record_ids: set[str]) -> dict[str, dict]:
    """Load specific cached records by ID."""
    cache = {}
    if not os.path.exists(cache_file):
        return cache
    
    with open(cache_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                rid = entry.get('id')
                if rid in record_ids:
                    cache[rid] = entry.get('data', {})
            except json.JSONDecodeError:
                continue
    return cache


def test_connection():
    """Test Snowflake connection."""
    print("\n" + "=" * 60)
    print("  TEST: Snowflake Connection")
    print("=" * 60)
    
    try:
        enricher = SnowflakeEnricher()
        success, message = enricher.test_connection()
        
        if success:
            print(f"\n✓ {message}")
            enricher.close()
            return True
        else:
            print(f"\n✗ {message}")
            return False
            
    except Exception as e:
        print(f"\n✗ Connection error: {e}")
        return False


def validate_against_cache(test_size: int = 100):
    """
    Validate Snowflake data against cached MixRank data.
    
    Tests TWO things:
    1. DATA COMPLETENESS: Does Snowflake have experience data where MixRank did?
    2. FALSE POSITIVE AGREEMENT: Do both sources agree on false positive classification?
    
    Exports:
    - output/missed_fps_mixrank_only.csv: FPs MixRank caught but Snowflake missed
    - output/snowflake_missing_data.csv: Records where Snowflake has no data but MixRank did
    """
    print("\n" + "=" * 60)
    print("  TEST: Validate Snowflake vs Cached MixRank Data")
    print("=" * 60)
    
    # Load cached record IDs
    print(f"\n📂 Loading cached record IDs from {CACHE_FILE}...")
    start = time.time()
    cached_ids = load_cached_ids_fast(CACHE_FILE)
    print(f"   Found {len(cached_ids):,} cached records ({time.time()-start:.1f}s)")
    
    if not cached_ids:
        print("\n✗ No cached records found!")
        return False
    
    # Load input records
    print(f"\n📂 Loading input records from {INPUT_FILE}...")
    all_records = load_input_records(INPUT_FILE)
    print(f"   Loaded {len(all_records):,} records")
    
    # Find records that are both in input AND in cache
    import random
    all_cached_records = [
        r for r in all_records 
        if r.get('ID') in cached_ids and r.get('ACCOUNT_NAME')
    ]
    print(f"   Found {len(all_cached_records):,} records in cache with account names")
    
    # Load ALL cached data to filter for records where MixRank has experience data
    print(f"\n📂 Loading cached enrichment data to find records with MixRank data...")
    all_cached_ids = {r['ID'] for r in all_cached_records}
    all_cached_data = load_cached_records(CACHE_FILE, all_cached_ids)
    
    # Filter to only records where MixRank actually has experience data
    records_with_mr_data = [
        r for r in all_cached_records
        if all_cached_data.get(r['ID'], {}).get('experience', [])
    ]
    print(f"   Found {len(records_with_mr_data):,} records where MixRank has experience data")
    
    # Random sample for testing
    records_to_test = random.sample(records_with_mr_data, min(test_size, len(records_with_mr_data)))
    print(f"\n   Selected {len(records_to_test)} records for validation (MixRank has data for all)")
    
    if not records_to_test:
        print("\n✗ No matching records found!")
        return False
    
    # Get cached data for test records
    record_ids = {r['ID'] for r in records_to_test}
    cached_data = {rid: all_cached_data[rid] for rid in record_ids if rid in all_cached_data}
    
    # Query Snowflake for the same URLs
    print(f"\n🔗 Querying Snowflake...")
    enricher = SnowflakeEnricher()
    
    # Build URL -> ID mapping
    url_to_id = {}
    id_to_url = {}
    id_to_record = {}
    for record in records_to_test:
        url = record.get('CONTACT_LINKED_IN_URL_C', '')
        if url:
            norm_result = normalize_linkedin_url(url)
            if norm_result.success:
                url_to_id[norm_result.normalized] = record['ID']
                id_to_url[record['ID']] = norm_result.normalized
                id_to_record[record['ID']] = record
    
    start = time.time()
    sf_results = enricher.enrich_batch(url_to_id)
    print(f"   Query completed ({time.time()-start:.1f}s)")
    
    # Initialize matcher for false positive detection
    print(f"\n🎯 Running matching on both data sources...")
    matcher = HybridMatcher(use_llm=False)
    
    # ============================================================
    # METRIC 1: DATA COMPLETENESS (Snowflake has data where MixRank did?)
    # ============================================================
    completeness_stats = {
        'both_have_data': 0,
        'mixrank_only': 0,  # Snowflake missing where MixRank had data
    }
    
    # ============================================================
    # METRIC 2: FALSE POSITIVE AGREEMENT
    # ============================================================
    fp_agreement_stats = {
        'agree_fp': 0,       # Both say FALSE POSITIVE
        'agree_not_fp': 0,   # Both say NOT false positive
        'disagree_mr_fp': 0, # MixRank says FP, Snowflake says not
        'disagree_sf_fp': 0, # Snowflake says FP, MixRank says not
    }
    
    # For exports
    missed_fps = []  # MixRank FP, Snowflake missed
    snowflake_missing = []  # Snowflake has no data
    
    for record in records_to_test:
        rid = record['ID']
        url = id_to_url.get(rid)
        account_name = record.get('ACCOUNT_NAME', '')
        linkedin_url = record.get('CONTACT_LINKED_IN_URL_C', '')
        
        if not url:
            continue
        
        # Get MixRank data (we know it has data - we filtered for this)
        mr_data = cached_data.get(rid, {})
        mr_experiences = mr_data.get('experience', [])
        
        # Get Snowflake data
        sf_result = sf_results.get(url)
        sf_experiences = []
        if sf_result and sf_result.success and sf_result.data:
            sf_experiences = sf_result.data.get('experience', [])
        
        # ---- COMPLETENESS CHECK ----
        sf_has_data = len(sf_experiences) > 0
        
        if sf_has_data:
            completeness_stats['both_have_data'] += 1
        else:
            completeness_stats['mixrank_only'] += 1
            # Get MixRank current companies for export
            mr_current = [e.get('company') for e in mr_experiences if e.get('is_current') or not e.get('end_date')]
            snowflake_missing.append({
                'record_id': rid,
                'account_name': account_name,
                'linkedin_url': linkedin_url,
                'mixrank_experience_count': len(mr_experiences),
                'mixrank_current_companies': '; '.join([c for c in mr_current if c][:5]),
            })
            continue  # Can't compare FP if Snowflake has no data
        
        # ---- FALSE POSITIVE AGREEMENT ----
        # Run matching on MixRank data
        mr_match = matcher.match(account_name, mr_experiences)
        mr_is_fp = mr_match.matched
        
        # Run matching on Snowflake data
        sf_match = matcher.match(account_name, sf_experiences)
        sf_is_fp = sf_match.matched
        
        if mr_is_fp and sf_is_fp:
            fp_agreement_stats['agree_fp'] += 1
        elif not mr_is_fp and not sf_is_fp:
            fp_agreement_stats['agree_not_fp'] += 1
        elif mr_is_fp and not sf_is_fp:
            fp_agreement_stats['disagree_mr_fp'] += 1
            # MixRank found FP but Snowflake didn't - THIS IS THE CONCERN
            sf_current = [e.get('company') for e in sf_experiences if e.get('is_current') or not e.get('end_date')]
            mr_current = [e.get('company') for e in mr_experiences if e.get('is_current') or not e.get('end_date')]
            missed_fps.append({
                'record_id': rid,
                'account_name': account_name,
                'linkedin_url': linkedin_url,
                'mixrank_matched_company': mr_match.matching_experience.get('company') if mr_match.matching_experience else '',
                'mixrank_match_confidence': mr_match.confidence.value if mr_match.confidence else '',
                'snowflake_current_companies': '; '.join([c for c in sf_current if c][:5]),
                'mixrank_current_companies': '; '.join([c for c in mr_current if c][:5]),
                'snowflake_experience_count': len(sf_experiences),
                'mixrank_experience_count': len(mr_experiences),
            })
        else:  # sf_is_fp and not mr_is_fp
            fp_agreement_stats['disagree_sf_fp'] += 1
    
    enricher.close()
    
    # ============================================================
    # EXPORT CSVs
    # ============================================================
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Export missed FPs
    missed_fps_file = output_dir / "missed_fps_mixrank_only.csv"
    if missed_fps:
        with open(missed_fps_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=missed_fps[0].keys())
            writer.writeheader()
            writer.writerows(missed_fps)
        print(f"\n💾 Exported {len(missed_fps)} missed FPs to: {missed_fps_file}")
    
    # Export Snowflake missing data
    sf_missing_file = output_dir / "snowflake_missing_data.csv"
    if snowflake_missing:
        with open(sf_missing_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=snowflake_missing[0].keys())
            writer.writeheader()
            writer.writerows(snowflake_missing)
        print(f"💾 Exported {len(snowflake_missing)} missing data records to: {sf_missing_file}")
    
    # ============================================================
    # REPORT
    # ============================================================
    total_records = completeness_stats['both_have_data'] + completeness_stats['mixrank_only']
    total_comparable = completeness_stats['both_have_data']
    total_agreements = fp_agreement_stats['agree_fp'] + fp_agreement_stats['agree_not_fp']
    
    sf_missing_rate = completeness_stats['mixrank_only'] / total_records * 100 if total_records > 0 else 0
    agreement_rate = total_agreements / total_comparable * 100 if total_comparable > 0 else 0
    missed_fp_rate = fp_agreement_stats['disagree_mr_fp'] / total_comparable * 100 if total_comparable > 0 else 0
    
    print(f"\n" + "=" * 60)
    print("  1. DATA COMPLETENESS (MixRank has data for all tested)")
    print("=" * 60)
    print(f"\n  Total records tested: {total_records}")
    print(f"  ✓ Snowflake also has data: {completeness_stats['both_have_data']} ({100-sf_missing_rate:.1f}%)")
    print(f"  ✗ Snowflake MISSING data: {completeness_stats['mixrank_only']} ({sf_missing_rate:.1f}%)")
    
    print(f"\n" + "=" * 60)
    print("  2. FALSE POSITIVE AGREEMENT (when Snowflake has data)")
    print("=" * 60)
    print(f"\n  Comparable records: {total_comparable}")
    print(f"  ✓ Both agree FP: {fp_agreement_stats['agree_fp']}")
    print(f"  ✓ Both agree NOT FP: {fp_agreement_stats['agree_not_fp']}")
    print(f"  ✗ MixRank FP, Snowflake MISSED: {fp_agreement_stats['disagree_mr_fp']} ({missed_fp_rate:.1f}%)")
    print(f"  ⚠ Snowflake FP, MixRank missed: {fp_agreement_stats['disagree_sf_fp']}")
    
    print(f"\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Snowflake data coverage: {100-sf_missing_rate:.1f}%")
    print(f"  FP Agreement rate: {agreement_rate:.1f}%")
    print(f"  ⚠ MISSED FP RATE: {missed_fp_rate:.1f}% ({fp_agreement_stats['disagree_mr_fp']} records)")
    
    print(f"\n  EXPORTS:")
    print(f"    - {missed_fps_file}: {len(missed_fps)} missed FPs for review")
    print(f"    - {sf_missing_file}: {len(snowflake_missing)} records with missing Snowflake data")
    
    # Success criteria
    completeness_ok = sf_missing_rate <= 20
    missed_fp_ok = missed_fp_rate <= 5
    
    print(f"\n  {'✓' if completeness_ok else '✗'} Data coverage: {'>=' if completeness_ok else '<'} 80%")
    print(f"  {'✓' if missed_fp_ok else '✗'} Missed FP rate: {'<=' if missed_fp_ok else '>'} 5%")
    
    success = completeness_ok and missed_fp_ok
    print(f"\n{'✓ PASSED' if success else '✗ NEEDS INVESTIGATION'}")
    
    return success


def test_new_records(test_size: int = 100):
    """
    Test Snowflake enrichment on records NOT in cache.
    
    Validates failure rate matches expected API behavior.
    """
    print("\n" + "=" * 60)
    print("  TEST: New Records (Not in Cache)")
    print("=" * 60)
    
    # Load cached IDs
    print(f"\n📂 Loading cached record IDs...")
    cached_ids = load_cached_ids_fast(CACHE_FILE)
    print(f"   Found {len(cached_ids):,} cached records")
    
    # Load input records NOT in cache
    print(f"\n📂 Loading input records NOT in cache...")
    all_records = load_input_records(INPUT_FILE)
    
    new_records = [
        r for r in all_records
        if r.get('ID') not in cached_ids
        and r.get('CONTACT_LINKED_IN_URL_C')
        and 'linkedin' in r.get('CONTACT_LINKED_IN_URL_C', '').lower()
    ][:test_size]
    
    print(f"   Selected {len(new_records)} new records for testing")
    
    if not new_records:
        print("\n✗ No new records found!")
        return False
    
    # Build URL -> ID mapping
    url_to_id = {}
    for record in new_records:
        url = record.get('CONTACT_LINKED_IN_URL_C', '')
        norm_result = normalize_linkedin_url(url)
        if norm_result.success:
            url_to_id[norm_result.normalized] = record['ID']
    
    # Query Snowflake
    print(f"\n🔗 Querying Snowflake for {len(url_to_id)} URLs...")
    enricher = SnowflakeEnricher()
    
    start = time.time()
    results = enricher.enrich_batch(url_to_id)
    elapsed = time.time() - start
    
    # Analyze results
    success_count = sum(1 for r in results.values() if r.success)
    failure_count = len(results) - success_count
    
    # Categorize failures
    not_found = sum(1 for r in results.values() if not r.success and 'not found' in (r.error or '').lower())
    no_data = sum(1 for r in results.values() if not r.success and 'no experience' in (r.error or '').lower())
    other_errors = failure_count - not_found - no_data
    
    success_rate = success_count / len(results) * 100 if results else 0
    
    enricher.close()
    
    print(f"\n" + "=" * 60)
    print("  NEW RECORDS RESULTS")
    print("=" * 60)
    print(f"\n  Total queried: {len(results)}")
    print(f"  Query time: {elapsed:.1f}s ({len(results)/elapsed:.1f} records/sec)")
    print(f"\n  ✓ Success (has experience data): {success_count} ({success_rate:.1f}%)")
    print(f"  ✗ Failures: {failure_count} ({100-success_rate:.1f}%)")
    print(f"    - Not found in Snowflake: {not_found}")
    print(f"    - No experience data: {no_data}")
    print(f"    - Other errors: {other_errors}")
    
    print(f"\n  Enricher stats: {enricher.get_stats()}")
    
    # Expected: similar to API failure rate (typically 10-30%)
    # We consider it a pass if we get >50% success
    success = success_rate >= 50
    print(f"\n{'✓ PASSED' if success else '✗ FAILED'}: Success rate {'≥' if success else '<'} 50%")
    
    return success


def full_test(test_size: int = 100, use_llm: bool = False):
    """
    Full end-to-end test with matching pipeline.
    """
    print("\n" + "=" * 60)
    print("  TEST: Full Pipeline (Enrichment + Matching)")
    print("=" * 60)
    
    # Load new records (not in cache)
    cached_ids = load_cached_ids_fast(CACHE_FILE)
    all_records = load_input_records(INPUT_FILE)
    
    test_records = [
        r for r in all_records
        if r.get('ID') not in cached_ids
        and r.get('CONTACT_LINKED_IN_URL_C')
        and r.get('ACCOUNT_NAME')
    ][:test_size]
    
    print(f"\n📂 Selected {len(test_records)} records for full test")
    
    if not test_records:
        print("\n✗ No suitable test records found!")
        return False
    
    # Enrich via Snowflake
    print(f"\n🔗 Enriching via Snowflake...")
    enricher = SnowflakeEnricher()
    
    url_to_id = {}
    id_to_record = {}
    for record in test_records:
        url = record.get('CONTACT_LINKED_IN_URL_C', '')
        norm_result = normalize_linkedin_url(url)
        if norm_result.success:
            url_to_id[norm_result.normalized] = record['ID']
            id_to_record[record['ID']] = record
    
    enrichment_results = enricher.enrich_batch(url_to_id)
    enricher.close()
    
    enriched_count = sum(1 for r in enrichment_results.values() if r.success)
    print(f"   Enriched: {enriched_count}/{len(enrichment_results)}")
    
    # Run matching
    print(f"\n🎯 Running matching ({'with LLM' if use_llm else 'programmatic only'})...")
    matcher = HybridMatcher(use_llm=use_llm)
    
    results = []
    false_positives = []
    
    for url, enrich_result in enrichment_results.items():
        record_id = url_to_id.get(url)
        record = id_to_record.get(record_id, {})
        account_name = record.get('ACCOUNT_NAME', '')
        
        if not enrich_result.success:
            continue
        
        experiences = enrich_result.data.get('experience', [])
        match_result = matcher.match(account_name, experiences)
        
        result = {
            'id': record_id,
            'account_name': account_name,
            'matched': match_result.matched,
            'confidence': match_result.confidence.value,
            'matched_company': match_result.matching_experience.get('company') if match_result.matching_experience else None,
        }
        results.append(result)
        
        if match_result.matched:
            false_positives.append(result)
    
    # Report
    match_stats = matcher.get_stats()
    fp_rate = len(false_positives) / len(results) * 100 if results else 0
    
    print(f"\n" + "=" * 60)
    print("  FULL PIPELINE RESULTS")
    print("=" * 60)
    print(f"\n  Records with enrichment data: {len(results)}")
    print(f"  FALSE POSITIVES FOUND: {len(false_positives)} ({fp_rate:.1f}%)")
    print(f"\n  Matching stats:")
    print(f"    Programmatic matches: {match_stats.get('programmatic_matches', 0)}")
    print(f"    LLM matches: {match_stats.get('llm_matches', 0)}")
    print(f"    No matches: {match_stats.get('no_matches', 0)}")
    
    if false_positives:
        print(f"\n  Sample false positives:")
        for fp in false_positives[:5]:
            print(f"    - {fp['account_name'][:40]} -> {fp['matched_company']}")
    
    print(f"\n✓ Full pipeline test completed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test Snowflake enrichment before production use",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--test-connection', action='store_true',
                        help='Test Snowflake connection only')
    parser.add_argument('--validate-cache', action='store_true',
                        help='Validate Snowflake data against cached MixRank data')
    parser.add_argument('--test-new', action='store_true',
                        help='Test on records NOT in cache')
    parser.add_argument('--full-test', action='store_true',
                        help='Full end-to-end test with matching')
    parser.add_argument('--test-size', type=int, default=100,
                        help='Number of records to test (default: 100)')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for matching in full test')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests')
    
    args = parser.parse_args()
    
    # Default to connection test if no args
    if not any([args.test_connection, args.validate_cache, args.test_new, args.full_test, args.all]):
        args.test_connection = True
    
    print("\n" + "=" * 60)
    print("  SNOWFLAKE ENRICHMENT TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    if args.test_connection or args.all:
        results['connection'] = test_connection()
    
    if args.validate_cache or args.all:
        results['validate_cache'] = validate_against_cache(args.test_size)
    
    if args.test_new or args.all:
        results['test_new'] = test_new_records(args.test_size)
    
    if args.full_test or args.all:
        results['full_test'] = full_test(args.test_size, args.use_llm)
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
