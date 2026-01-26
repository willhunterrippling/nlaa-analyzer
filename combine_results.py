#!/usr/bin/env python3
"""
Combine Results Script

Stitches together results from multiple analysis runs into a single combined output.
Specifically designed to combine:
1. Emergency eject partial results (~330k records from original run)
2. New Snowflake-enriched results (remaining records)

Usage:
    python combine_results.py --emergency-results <path> --new-results <path> [--output-dir <path>]

Example:
    python combine_results.py \
        --emergency-results output/emergency_eject_20260113_190010/all_results_partial.csv \
        --new-results output/run_snowflake_20260127/all_results.csv \
        --output-dir output/combined_final
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_EMERGENCY_RESULTS = "output/emergency_eject_20260113_190010/all_results_partial.csv"
DEFAULT_OUTPUT_DIR = "output"


def load_csv(filepath: str) -> tuple[list[dict], list[str]]:
    """Load CSV file and return (records, fieldnames)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = list(reader)
        return records, reader.fieldnames or []


def count_csv_rows(filepath: str) -> int:
    """Fast line count (excluding header)."""
    count = 0
    with open(filepath, 'rb') as f:
        # Read in chunks for speed
        buf_size = 1024 * 1024
        while True:
            buf = f.read(buf_size)
            if not buf:
                break
            count += buf.count(b'\n')
    return max(0, count - 1)  # Subtract header line


def save_csv(records: list[dict], filepath: Path, fieldnames: Optional[list[str]] = None):
    """Save records to CSV file."""
    if not records:
        return
    
    if fieldnames is None:
        # Gather all unique keys
        all_keys = set()
        for r in records:
            all_keys.update(r.keys())
        
        # Prioritize common columns
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


def generate_combined_report(
    emergency_count: int,
    new_count: int,
    combined_records: list[dict],
    false_positives: list[dict],
    duplicates_removed: int,
    emergency_source: str,
    new_source: str,
    output_path: Path,
    timestamp: str,
) -> Path:
    """Generate a combined analysis report."""
    
    total = len(combined_records)
    fp_rate = (len(false_positives) / total * 100) if total > 0 else 0
    
    # Count by confidence
    high_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'high')
    medium_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'medium')
    low_conf = sum(1 for r in false_positives if r.get('match_confidence') == 'low')
    
    # Count enrichment success
    enrichment_success = sum(1 for r in combined_records if str(r.get('enrichment_success', '')).lower() == 'true')
    enrichment_failed = total - enrichment_success
    
    # Count ambiguous cases (matched=False, confidence=low)
    ambiguous_cases = [
        r for r in combined_records 
        if str(r.get('matched', '')).lower() == 'false' 
        and r.get('match_confidence') == 'low'
    ]
    
    # Count LLM errors
    llm_errors = [r for r in combined_records if str(r.get('llm_error', '')).lower() == 'true']
    
    report = f"""# NLAA False Positive Analysis - COMBINED FINAL REPORT

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| Emergency Eject | {emergency_count:,} | `{emergency_source}` |
| New Snowflake Run | {new_count:,} | `{new_source}` |
| **Combined Total** | **{total:,}** | After deduplication |

{f"⚠️ **{duplicates_removed:,} duplicate records** were found and removed based on ID." if duplicates_removed > 0 else "✅ No duplicate records found."}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Records Analyzed** | {total:,} |
| **False Positives Found** | {len(false_positives):,} |
| **False Positive Rate** | **{fp_rate:.1f}%** |
| **Ambiguous Cases** | {len(ambiguous_cases):,} |
| **LLM Errors** | {len(llm_errors):,} |

### Key Finding

**{len(false_positives):,} leads ({fp_rate:.1f}%)** were incorrectly flagged as "No Longer At Account" and are actually still at their company based on current LinkedIn data.

---

## Enrichment Results

| Status | Count | Percentage |
|--------|-------|------------|
| Success | {enrichment_success:,} | {enrichment_success/total*100:.1f}% |
| Failed | {enrichment_failed:,} | {enrichment_failed/total*100:.1f}% |

---

## False Positives by Confidence Level

| Confidence | Count | Percentage of FPs |
|------------|-------|-------------------|
| High | {high_conf:,} | {high_conf/len(false_positives)*100 if false_positives else 0:.1f}% |
| Medium | {medium_conf:,} | {medium_conf/len(false_positives)*100 if false_positives else 0:.1f}% |
| Low | {low_conf:,} | {low_conf/len(false_positives)*100 if false_positives else 0:.1f}% |

---

## Sample False Positives

| Account Name | Matched LinkedIn Company | Confidence | Reason |
|--------------|-------------------------|------------|--------|
"""
    
    sample_fps = false_positives[:10]
    for fp in sample_fps:
        account = str(fp.get('ACCOUNT_NAME', 'N/A'))[:40]
        matched = str(fp.get('matched_company', 'N/A'))[:40]
        conf = fp.get('match_confidence', 'N/A')
        reason = str(fp.get('match_reason', 'N/A'))[:50]
        report += f"| {account} | {matched} | {conf} | {reason}... |\n"
    
    if len(false_positives) > 10:
        report += f"\n*Showing 10 of {len(false_positives):,} false positives. See CSV for full list.*\n"
    
    report += f"""
---

## Output Files

- `combined_all_results.csv` - All {total:,} combined results
- `combined_false_positives.csv` - All {len(false_positives):,} confirmed false positives
- `combined_ambiguous.csv` - {len(ambiguous_cases):,} cases needing review
- `combined_report.md` - This report

---

## Recommendations

1. **Review false positives** - These {len(false_positives):,} leads should have their NLAA flag removed
2. **Investigate ambiguous cases** - {len(ambiguous_cases):,} records need manual review
"""
    
    if llm_errors:
        report += f"""3. **Fix LLM errors** - {len(llm_errors):,} records had LLM matching failures
"""
    
    report += f"""
---

*Combined report generated by combine_results.py*
"""
    
    report_path = output_path / f"combined_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_path


def combine_results(
    emergency_results_path: str,
    new_results_path: str,
    output_dir: str,
):
    """
    Combine results from emergency eject and new Snowflake run.
    """
    
    print("\n" + "=" * 70)
    print("  COMBINE RESULTS - Stitching Analysis Outputs")
    print("=" * 70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(output_dir) / f"combined_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validate inputs
    if not os.path.exists(emergency_results_path):
        print(f"\n✗ Emergency results not found: {emergency_results_path}")
        sys.exit(1)
    
    if not os.path.exists(new_results_path):
        print(f"\n✗ New results not found: {new_results_path}")
        sys.exit(1)
    
    # Fast count first
    print(f"\n📊 Counting records...")
    emergency_count = count_csv_rows(emergency_results_path)
    new_count = count_csv_rows(new_results_path)
    print(f"   Emergency eject: {emergency_count:,} records")
    print(f"   New Snowflake:   {new_count:,} records")
    print(f"   Expected total:  {emergency_count + new_count:,} records")
    
    # Load emergency results
    print(f"\n📂 Loading emergency results: {emergency_results_path}")
    emergency_records, emergency_fields = load_csv(emergency_results_path)
    print(f"   Loaded: {len(emergency_records):,} records")
    
    # Load new results
    print(f"\n📂 Loading new results: {new_results_path}")
    new_records, new_fields = load_csv(new_results_path)
    print(f"   Loaded: {len(new_records):,} records")
    
    # Combine and deduplicate by ID
    print(f"\n🔗 Combining results...")
    combined_by_id = {}
    duplicates = 0
    
    # Add emergency records first (they were processed first chronologically)
    for record in emergency_records:
        record_id = record.get('ID', '')
        if record_id:
            combined_by_id[record_id] = record
    
    # Add new records (newer data takes precedence if ID exists)
    for record in new_records:
        record_id = record.get('ID', '')
        if record_id:
            if record_id in combined_by_id:
                duplicates += 1
            combined_by_id[record_id] = record  # Newer data wins
    
    combined_records = list(combined_by_id.values())
    
    print(f"   Combined total: {len(combined_records):,} records")
    if duplicates > 0:
        print(f"   ⚠️  Duplicates found and merged: {duplicates:,}")
    
    # Identify false positives (matched=True means they're still at the company = false positive)
    false_positives = [
        r for r in combined_records 
        if str(r.get('matched', '')).lower() == 'true'
    ]
    
    fp_rate = len(false_positives) / len(combined_records) * 100 if combined_records else 0
    
    print(f"\n📊 Analysis Summary:")
    print(f"   Total records:      {len(combined_records):,}")
    print(f"   False positives:    {len(false_positives):,} ({fp_rate:.1f}%)")
    
    # Find ambiguous cases
    ambiguous_cases = [
        r for r in combined_records 
        if str(r.get('matched', '')).lower() == 'false' 
        and r.get('match_confidence') == 'low'
    ]
    print(f"   Ambiguous cases:    {len(ambiguous_cases):,}")
    
    # Merge field names (prioritize emergency fields, add any new ones)
    all_fields = list(emergency_fields) if emergency_fields else []
    for field in (new_fields or []):
        if field not in all_fields:
            all_fields.append(field)
    
    # Save combined outputs
    print(f"\n💾 Saving combined outputs to: {output_path}")
    
    # All results
    all_file = output_path / "combined_all_results.csv"
    save_csv(combined_records, all_file, all_fields)
    print(f"   ✓ All results: {all_file}")
    
    # False positives
    if false_positives:
        fp_file = output_path / "combined_false_positives.csv"
        save_csv(false_positives, fp_file, all_fields)
        print(f"   ✓ False positives: {fp_file}")
    
    # Ambiguous cases
    if ambiguous_cases:
        amb_file = output_path / "combined_ambiguous.csv"
        save_csv(ambiguous_cases, amb_file, all_fields)
        print(f"   ✓ Ambiguous cases: {amb_file}")
    
    # Generate report
    report_path = generate_combined_report(
        emergency_count=len(emergency_records),
        new_count=len(new_records),
        combined_records=combined_records,
        false_positives=false_positives,
        duplicates_removed=duplicates,
        emergency_source=emergency_results_path,
        new_source=new_results_path,
        output_path=output_path,
        timestamp=timestamp,
    )
    print(f"   ✓ Report: {report_path}")
    
    print(f"\n{'=' * 70}")
    print("  COMBINATION COMPLETE")
    print("=" * 70)
    print(f"\n   📁 Output folder: {output_path}")
    print(f"   📊 Total records: {len(combined_records):,}")
    print(f"   🎯 False positives: {len(false_positives):,} ({fp_rate:.1f}%)")
    print(f"\n{'=' * 70}")
    
    return {
        'combined_total': len(combined_records),
        'false_positives': len(false_positives),
        'ambiguous': len(ambiguous_cases),
        'duplicates_removed': duplicates,
        'output_dir': str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Combine results from multiple NLAA analysis runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine emergency eject with new Snowflake results
  python combine_results.py \\
      --emergency-results output/emergency_eject_20260113_190010/all_results_partial.csv \\
      --new-results output/run_snowflake_YYYYMMDD/all_results.csv

  # Specify custom output directory
  python combine_results.py \\
      --emergency-results <path1> \\
      --new-results <path2> \\
      --output-dir output/final_combined
        """
    )
    
    parser.add_argument(
        '--emergency-results', '-e',
        default=DEFAULT_EMERGENCY_RESULTS,
        help=f'Path to emergency eject results CSV (default: {DEFAULT_EMERGENCY_RESULTS})'
    )
    parser.add_argument(
        '--new-results', '-n',
        required=True,
        help='Path to new Snowflake run results CSV (required)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    args = parser.parse_args()
    
    try:
        results = combine_results(
            emergency_results_path=args.emergency_results,
            new_results_path=args.new_results,
            output_dir=args.output_dir,
        )
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

