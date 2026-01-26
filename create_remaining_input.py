#!/usr/bin/env python3
"""
Create a remaining input file by filtering out already-processed records.

This script takes the original input CSV and removes all records that have
already been processed (present in all_results_partial.csv).
"""

import csv
import sys
from pathlib import Path


def create_remaining_input(
    original_input: Path,
    partial_results: Path,
    output_file: Path
) -> dict:
    """
    Filter original input to exclude already-processed records.
    
    Args:
        original_input: Path to the original input CSV
        partial_results: Path to all_results_partial.csv with processed records
        output_file: Path for the output remaining records CSV
    
    Returns:
        dict with statistics about the operation
    """
    # Step 1: Load all processed IDs into a set for O(1) lookup
    print(f"Loading processed IDs from {partial_results}...")
    processed_ids = set()
    
    with open(partial_results, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            processed_ids.add(row['ID'])
    
    print(f"  Found {len(processed_ids):,} processed records")
    
    # Step 2: Stream through original input and filter
    print(f"Filtering {original_input}...")
    original_count = 0
    remaining_count = 0
    
    with open(original_input, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            original_count += 1
            if row['ID'] not in processed_ids:
                writer.writerow(row)
                remaining_count += 1
            
            # Progress indicator every 100k records
            if original_count % 100000 == 0:
                print(f"  Processed {original_count:,} records...")
    
    stats = {
        'original_count': original_count,
        'processed_count': len(processed_ids),
        'remaining_count': remaining_count,
        'filtered_count': original_count - remaining_count
    }
    
    print(f"\nResults:")
    print(f"  Original records:  {stats['original_count']:,}")
    print(f"  Already processed: {stats['filtered_count']:,}")
    print(f"  Remaining records: {stats['remaining_count']:,}")
    print(f"\nOutput written to: {output_file}")
    
    return stats


def main():
    # Default paths
    base_dir = Path(__file__).parent
    
    original_input = base_dir / "inputs" / "01c1b4ef-070a-972b-5342-83029a156523.csv"
    partial_results = base_dir / "output" / "emergency_eject_20260113_190010" / "all_results_partial.csv"
    output_file = base_dir / "inputs" / "remaining.csv"
    
    # Validate inputs exist
    if not original_input.exists():
        print(f"Error: Original input not found: {original_input}")
        sys.exit(1)
    
    if not partial_results.exists():
        print(f"Error: Partial results not found: {partial_results}")
        sys.exit(1)
    
    # Run the filtering
    stats = create_remaining_input(original_input, partial_results, output_file)
    
    # Sanity check
    expected_remaining = stats['original_count'] - stats['processed_count']
    if stats['remaining_count'] != expected_remaining:
        print(f"\nWarning: Expected {expected_remaining:,} remaining, got {stats['remaining_count']:,}")
        print("  This may indicate some processed IDs were not in the original input.")


if __name__ == "__main__":
    main()

