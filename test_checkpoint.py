#!/usr/bin/env python3
"""
Checkpoint System Test Script

Validates that the checkpoint/resume system works correctly by running
through various scenarios with isolated temp directories.

Usage:
    python test_checkpoint.py
"""

import json
import os
import sys
import shutil
import tempfile
from pathlib import Path

from checkpoint import CheckpointManager, CheckpointState


class TestRunner:
    """Simple test runner with pass/fail tracking."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def run_test(self, name: str, test_func):
        """Run a test and track result."""
        try:
            test_func()
            self.passed += 1
            self.results.append((name, True, None))
            print(f"[PASS] {name}")
        except AssertionError as e:
            self.failed += 1
            self.results.append((name, False, str(e)))
            print(f"[FAIL] {name}")
            print(f"       Reason: {e}")
        except Exception as e:
            self.failed += 1
            self.results.append((name, False, str(e)))
            print(f"[ERROR] {name}")
            print(f"        Exception: {e}")
    
    def summary(self):
        """Print summary and return exit code."""
        total = self.passed + self.failed
        print(f"\n{'=' * 50}")
        print(f"Results: {self.passed}/{total} tests passed")
        
        if self.failed > 0:
            print(f"\nFailed tests:")
            for name, passed, error in self.results:
                if not passed:
                    print(f"  - {name}: {error}")
        
        return 0 if self.failed == 0 else 1


def create_temp_checkpoint_dir():
    """Create an isolated temp directory for testing."""
    return tempfile.mkdtemp(prefix="checkpoint_test_")


def test_save_load_roundtrip():
    """Test 1: State survives being saved to disk and reloaded."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        # Create manager and initialize
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=10)
        mgr.initialize("test_input.csv", total_records=100)
        
        # Process some records
        for i in range(5):
            mgr.record_processed(
                record_id=f"rec_{i:03d}",
                success=True,
                enriched_data={"name": f"Person {i}"}
            )
        
        # Force save
        mgr.save_checkpoint(force=True)
        
        # Load in new manager
        mgr2 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=10)
        loaded = mgr2.load_checkpoint()
        
        assert loaded, "Checkpoint should load successfully"
        assert mgr2.state.total_records == 100, f"Expected 100 total records, got {mgr2.state.total_records}"
        assert mgr2.state.input_file == "test_input.csv", "Input file should match"
        assert len(mgr2.state.processed_ids) == 5, f"Expected 5 processed IDs, got {len(mgr2.state.processed_ids)}"
        assert mgr2.state.success_count == 5, f"Expected 5 successes, got {mgr2.state.success_count}"
        
    finally:
        shutil.rmtree(test_dir)


def test_processed_id_tracking():
    """Test 2: is_processed() correctly identifies processed records."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        mgr.initialize("test.csv", total_records=50)
        
        # Process specific records
        mgr.record_processed("ABC123", success=True)
        mgr.record_processed("DEF456", success=True)
        mgr.record_processed("GHI789", success=False)
        
        # Check tracking
        assert mgr.is_processed("ABC123"), "ABC123 should be marked as processed"
        assert mgr.is_processed("DEF456"), "DEF456 should be marked as processed"
        assert mgr.is_processed("GHI789"), "GHI789 should be marked as processed (even if failed)"
        assert not mgr.is_processed("XYZ999"), "XYZ999 should NOT be marked as processed"
        
        # Save and reload
        mgr.save_checkpoint(force=True)
        
        mgr2 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        mgr2.load_checkpoint()
        
        # Verify tracking persists
        assert mgr2.is_processed("ABC123"), "ABC123 should still be processed after reload"
        assert not mgr2.is_processed("XYZ999"), "XYZ999 should still NOT be processed after reload"
        
    finally:
        shutil.rmtree(test_dir)


def test_enrichment_cache():
    """Test 3: Cached enrichment data persists and can be retrieved after reload."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        mgr.initialize("test.csv", total_records=10)
        
        # Process with enrichment data
        enriched = {
            "name": "John Doe",
            "experience": [
                {"company": "Acme Corp", "title": "Engineer", "is_current": True},
                {"company": "Old Job", "title": "Junior", "is_current": False},
            ]
        }
        mgr.record_processed("REC001", success=True, enriched_data=enriched)
        mgr.record_processed("REC002", success=False)  # No enrichment for failures
        
        mgr.save_checkpoint(force=True)
        
        # Reload
        mgr2 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        mgr2.load_checkpoint()
        
        # Verify cache
        cached = mgr2.get_cached_enrichment("REC001")
        assert cached is not None, "REC001 should have cached data"
        assert cached["name"] == "John Doe", "Cached name should match"
        assert len(cached["experience"]) == 2, "Should have 2 experiences"
        assert cached["experience"][0]["company"] == "Acme Corp", "First company should be Acme Corp"
        
        assert mgr2.get_cached_enrichment("REC002") is None, "REC002 should have no cached data"
        assert mgr2.get_cached_enrichment("REC999") is None, "Non-existent record should return None"
        
    finally:
        shutil.rmtree(test_dir)


def test_checkpoint_interval():
    """Test 4: Saves only happen at configured intervals (or when forced)."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=5)
        mgr.initialize("test.csv", total_records=20)
        
        checkpoint_file = Path(test_dir) / "checkpoint.json"
        
        # Process 3 records - should NOT auto-save (interval is 5)
        for i in range(3):
            mgr.record_processed(f"rec_{i}", success=True)
        
        # Check file doesn't exist yet (no auto-save)
        # Note: The initialize doesn't create the file, only save_checkpoint does
        assert not checkpoint_file.exists(), "Checkpoint file should not exist before interval reached"
        
        # Process 2 more records - should trigger auto-save at 5
        for i in range(3, 5):
            mgr.record_processed(f"rec_{i}", success=True)
        
        assert checkpoint_file.exists(), "Checkpoint file should exist after interval reached"
        
        # Delete file to test force save
        checkpoint_file.unlink()
        
        # Process 1 more record - should NOT auto-save
        mgr.record_processed("rec_5", success=True)
        assert not checkpoint_file.exists(), "Should not save after just 1 record"
        
        # Force save
        mgr.save_checkpoint(force=True)
        assert checkpoint_file.exists(), "Checkpoint file should exist after force save"
        
    finally:
        shutil.rmtree(test_dir)


def test_statistics_accuracy():
    """Test 5: Success/error/skipped counts remain accurate across save/load."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        mgr.initialize("test.csv", total_records=15)
        
        # Process with different outcomes
        for i in range(5):
            mgr.record_processed(f"success_{i}", success=True)
        
        for i in range(3):
            mgr.record_processed(f"error_{i}", success=False)
        
        for i in range(2):
            mgr.record_processed(f"skip_{i}", success=True, skipped=True)
        
        mgr.save_checkpoint(force=True)
        
        # Verify stats before reload
        assert mgr.state.success_count == 5, f"Expected 5 successes, got {mgr.state.success_count}"
        assert mgr.state.error_count == 3, f"Expected 3 errors, got {mgr.state.error_count}"
        assert mgr.state.skipped_count == 2, f"Expected 2 skipped, got {mgr.state.skipped_count}"
        
        # Reload
        mgr2 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        mgr2.load_checkpoint()
        
        # Verify stats after reload
        assert mgr2.state.success_count == 5, f"After reload: Expected 5 successes, got {mgr2.state.success_count}"
        assert mgr2.state.error_count == 3, f"After reload: Expected 3 errors, got {mgr2.state.error_count}"
        assert mgr2.state.skipped_count == 2, f"After reload: Expected 2 skipped, got {mgr2.state.skipped_count}"
        
        # Verify get_progress returns correct data
        progress = mgr2.get_progress()
        assert progress["processed"] == 10, f"Progress should show 10 processed, got {progress['processed']}"
        assert progress["success"] == 5, f"Progress should show 5 success, got {progress['success']}"
        assert progress["errors"] == 3, f"Progress should show 3 errors, got {progress['errors']}"
        
    finally:
        shutil.rmtree(test_dir)


def test_resume_simulation():
    """Test 6: Simulate a crash mid-processing, verify new manager can resume correctly."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        # Phase 1: Process first batch and "crash"
        mgr1 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=5)
        mgr1.initialize("big_dataset.csv", total_records=100)
        
        # Process records 0-9
        for i in range(10):
            mgr1.record_processed(
                f"record_{i:03d}",
                success=True,
                enriched_data={"data": f"enriched_{i}"}
            )
        
        mgr1.save_checkpoint(force=True)
        # Simulate crash - mgr1 goes out of scope
        del mgr1
        
        # Phase 2: "Resume" with new manager
        mgr2 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=5)
        loaded = mgr2.load_checkpoint()
        
        assert loaded, "Should successfully load checkpoint"
        assert mgr2.state.input_file == "big_dataset.csv", "Input file should match"
        assert len(mgr2.state.processed_ids) == 10, "Should have 10 processed IDs"
        
        # Process more records (simulating resume)
        new_records_processed = 0
        for i in range(15):  # Try to process 0-14
            record_id = f"record_{i:03d}"
            
            if mgr2.is_processed(record_id):
                # Skip already processed
                continue
            
            # Process new record
            mgr2.record_processed(
                record_id,
                success=True,
                enriched_data={"data": f"enriched_{i}"}
            )
            new_records_processed += 1
        
        # Should have only processed 10-14 (5 new records)
        assert new_records_processed == 5, f"Should process 5 new records, processed {new_records_processed}"
        assert len(mgr2.state.processed_ids) == 15, f"Total should be 15 processed, got {len(mgr2.state.processed_ids)}"
        
        # Verify old cached data still accessible
        old_cached = mgr2.get_cached_enrichment("record_005")
        assert old_cached is not None, "Old cached data should still be accessible"
        assert old_cached["data"] == "enriched_5", "Old cached data should be correct"
        
        # Verify new cached data accessible
        new_cached = mgr2.get_cached_enrichment("record_012")
        assert new_cached is not None, "New cached data should be accessible"
        assert new_cached["data"] == "enriched_12", "New cached data should be correct"
        
    finally:
        shutil.rmtree(test_dir)


def test_clear_checkpoint():
    """Test 7: Clearing removes all state and files."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=5)
        mgr.initialize("test.csv", total_records=50)
        
        # Process and save
        for i in range(10):
            mgr.record_processed(f"rec_{i}", success=True, enriched_data={"x": i})
        mgr.save_checkpoint(force=True)
        
        checkpoint_file = Path(test_dir) / "checkpoint.json"
        cache_file = Path(test_dir) / "enrichment_cache.jsonl"  # Updated to JSONL
        
        # Verify files exist
        assert checkpoint_file.exists(), "Checkpoint file should exist before clear"
        assert cache_file.exists(), "Cache file should exist before clear"
        
        # Clear
        mgr.clear_checkpoint()
        
        # Verify files removed
        assert not checkpoint_file.exists(), "Checkpoint file should be removed after clear"
        assert not cache_file.exists(), "Cache file should be removed after clear"
        
        # Verify state reset
        assert len(mgr.state.processed_ids) == 0, "Processed IDs should be empty"
        assert mgr.state.success_count == 0, "Success count should be 0"
        assert len(mgr.enrichment_cache) == 0, "Enrichment cache should be empty"
        
        # Verify load returns False
        mgr2 = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=5)
        assert not mgr2.load_checkpoint(), "Load should return False after clear"
        
    finally:
        shutil.rmtree(test_dir)


def test_jsonl_format():
    """Test 8: Verify JSONL format is used for cache."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=5)
        mgr.initialize("test.csv", total_records=10)
        
        # Process records with enrichment data
        for i in range(3):
            mgr.record_processed(
                f"rec_{i}",
                success=True,
                enriched_data={"value": i, "name": f"Person {i}"}
            )
        
        mgr.save_checkpoint(force=True)
        
        # Verify JSONL file exists
        cache_file = Path(test_dir) / "enrichment_cache.jsonl"
        assert cache_file.exists(), "JSONL cache file should exist"
        
        # Verify format: each line should be valid JSON
        with open(cache_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 3, f"Should have 3 lines, got {len(lines)}"
        
        for i, line in enumerate(lines):
            entry = json.loads(line.strip())
            assert "id" in entry, f"Line {i} should have 'id' field"
            assert "data" in entry, f"Line {i} should have 'data' field"
            assert entry["data"]["value"] == i, f"Line {i} data value should be {i}"
        
    finally:
        shutil.rmtree(test_dir)


def test_append_only_writes():
    """Test 9: Verify append-only behavior (new records appended, not rewritten)."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=3)
        mgr.initialize("test.csv", total_records=20)
        
        cache_file = Path(test_dir) / "enrichment_cache.jsonl"
        
        # First batch: 3 records (triggers save)
        for i in range(3):
            mgr.record_processed(f"batch1_rec_{i}", success=True, enriched_data={"batch": 1, "idx": i})
        
        # Verify first batch written
        assert cache_file.exists(), "Cache file should exist after first batch"
        with open(cache_file, 'r') as f:
            lines1 = f.readlines()
        assert len(lines1) == 3, f"Should have 3 lines after first batch, got {len(lines1)}"
        
        # Second batch: 3 more records (triggers another save)
        for i in range(3):
            mgr.record_processed(f"batch2_rec_{i}", success=True, enriched_data={"batch": 2, "idx": i})
        
        # Verify second batch appended (not rewritten)
        with open(cache_file, 'r') as f:
            lines2 = f.readlines()
        assert len(lines2) == 6, f"Should have 6 lines after second batch, got {len(lines2)}"
        
        # Verify first batch records still there (not overwritten)
        first_entry = json.loads(lines2[0])
        assert first_entry["id"] == "batch1_rec_0", "First entry should still be from batch 1"
        assert first_entry["data"]["batch"] == 1, "First entry should have batch=1"
        
        # Verify second batch records appended
        fourth_entry = json.loads(lines2[3])
        assert fourth_entry["id"] == "batch2_rec_0", "Fourth entry should be from batch 2"
        assert fourth_entry["data"]["batch"] == 2, "Fourth entry should have batch=2"
        
    finally:
        shutil.rmtree(test_dir)


def test_legacy_json_migration():
    """Test 10: Verify migration from legacy JSON cache to JSONL."""
    test_dir = create_temp_checkpoint_dir()
    
    try:
        # Create legacy JSON cache file manually
        legacy_cache = {
            "legacy_rec_1": {"name": "Legacy Person 1", "value": 100},
            "legacy_rec_2": {"name": "Legacy Person 2", "value": 200},
            "legacy_rec_3": {"name": "Legacy Person 3", "value": 300},
        }
        
        legacy_cache_file = Path(test_dir) / "enrichment_cache.json"
        with open(legacy_cache_file, 'w') as f:
            json.dump(legacy_cache, f)
        
        # Create checkpoint file
        checkpoint_data = {
            "processed_ids": ["legacy_rec_1", "legacy_rec_2", "legacy_rec_3"],
            "last_processed_index": 3,
            "total_records": 10,
            "success_count": 3,
            "error_count": 0,
            "skipped_count": 0,
            "started_at": "2026-01-01T00:00:00",
            "last_checkpoint_at": "2026-01-01T00:01:00",
            "input_file": "legacy_test.csv",
            "checkpoint_version": "1.0",
        }
        checkpoint_file = Path(test_dir) / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)
        
        # Load with new manager - should trigger migration
        mgr = CheckpointManager(checkpoint_dir=test_dir, checkpoint_interval=100)
        loaded = mgr.load_checkpoint()
        
        assert loaded, "Should load checkpoint successfully"
        
        # Verify migration occurred
        jsonl_file = Path(test_dir) / "enrichment_cache.jsonl"
        backup_file = Path(test_dir) / "enrichment_cache.json.bak"
        
        assert jsonl_file.exists(), "JSONL cache file should exist after migration"
        assert backup_file.exists(), "Legacy cache should be backed up"
        assert not legacy_cache_file.exists(), "Original legacy file should be moved"
        
        # Verify all data preserved
        assert len(mgr.enrichment_cache) == 3, f"Should have 3 cache entries, got {len(mgr.enrichment_cache)}"
        assert mgr.get_cached_enrichment("legacy_rec_1")["value"] == 100, "Legacy data should be preserved"
        assert mgr.get_cached_enrichment("legacy_rec_2")["name"] == "Legacy Person 2", "Legacy data should be preserved"
        
        # Verify JSONL format is correct
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 3, f"JSONL should have 3 lines, got {len(lines)}"
        
        # Process new records after migration
        mgr.record_processed("new_rec_1", success=True, enriched_data={"name": "New Person"})
        mgr.save_checkpoint(force=True)
        
        # Verify new record appended
        with open(jsonl_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) == 4, f"JSONL should have 4 lines after new record, got {len(lines)}"
        
    finally:
        shutil.rmtree(test_dir)


def main():
    print("=" * 50)
    print("  Checkpoint System Tests")
    print("=" * 50)
    print()
    
    runner = TestRunner()
    
    runner.run_test("Test 1: Save/Load Round-Trip", test_save_load_roundtrip)
    runner.run_test("Test 2: Processed ID Tracking", test_processed_id_tracking)
    runner.run_test("Test 3: Enrichment Cache Persistence", test_enrichment_cache)
    runner.run_test("Test 4: Checkpoint Interval Logic", test_checkpoint_interval)
    runner.run_test("Test 5: Statistics Accuracy", test_statistics_accuracy)
    runner.run_test("Test 6: Resume Simulation", test_resume_simulation)
    runner.run_test("Test 7: Clear Checkpoint", test_clear_checkpoint)
    runner.run_test("Test 8: JSONL Format", test_jsonl_format)
    runner.run_test("Test 9: Append-Only Writes", test_append_only_writes)
    runner.run_test("Test 10: Legacy JSON Migration", test_legacy_json_migration)
    
    exit_code = runner.summary()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
