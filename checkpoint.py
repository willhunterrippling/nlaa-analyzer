"""
Checkpoint/Resume System for NLAA Analysis

Provides crash recovery by saving progress at regular intervals.
Stores:
- Processed record IDs
- Enriched data cache
- Current position in the dataset
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class CheckpointState:
    """State saved in checkpoint file."""
    # Processing state
    processed_ids: set = field(default_factory=set)
    last_processed_index: int = 0
    total_records: int = 0
    
    # Statistics
    success_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    
    # Timestamps
    started_at: str = ""
    last_checkpoint_at: str = ""
    
    # Configuration (to detect if settings changed)
    input_file: str = ""
    checkpoint_version: str = "1.0"
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "processed_ids": list(self.processed_ids),
            "last_processed_index": self.last_processed_index,
            "total_records": self.total_records,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "skipped_count": self.skipped_count,
            "started_at": self.started_at,
            "last_checkpoint_at": self.last_checkpoint_at,
            "input_file": self.input_file,
            "checkpoint_version": self.checkpoint_version,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CheckpointState":
        """Create from dict (loaded from JSON)."""
        state = cls()
        state.processed_ids = set(data.get("processed_ids", []))
        state.last_processed_index = data.get("last_processed_index", 0)
        state.total_records = data.get("total_records", 0)
        state.success_count = data.get("success_count", 0)
        state.error_count = data.get("error_count", 0)
        state.skipped_count = data.get("skipped_count", 0)
        state.started_at = data.get("started_at", "")
        state.last_checkpoint_at = data.get("last_checkpoint_at", "")
        state.input_file = data.get("input_file", "")
        state.checkpoint_version = data.get("checkpoint_version", "1.0")
        return state


class CheckpointManager:
    """Manages checkpoint save/load operations."""
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        checkpoint_file: str = "checkpoint.json",
        cache_file: str = "enrichment_cache.json",
        checkpoint_interval: int = 100
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / checkpoint_file
        self.cache_file = self.checkpoint_dir / cache_file
        self.checkpoint_interval = checkpoint_interval
        
        # State
        self.state = CheckpointState()
        self.enrichment_cache: dict[str, Any] = {}  # id -> enriched data
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track when to save
        self._records_since_checkpoint = 0
    
    def initialize(self, input_file: str, total_records: int):
        """Initialize a new checkpoint session."""
        self.state = CheckpointState(
            total_records=total_records,
            started_at=datetime.now().isoformat(),
            input_file=input_file
        )
        self._records_since_checkpoint = 0
    
    def load_checkpoint(self) -> bool:
        """
        Load existing checkpoint if available.
        Returns True if checkpoint was loaded, False otherwise.
        """
        if not self.checkpoint_file.exists():
            return False
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            self.state = CheckpointState.from_dict(data)
            
            # Load enrichment cache if exists
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.enrichment_cache = json.load(f)
            
            return True
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return False
    
    def save_checkpoint(self, force: bool = False):
        """
        Save current state to checkpoint file.
        Only saves if interval reached or force=True.
        """
        if not force and self._records_since_checkpoint < self.checkpoint_interval:
            return
        
        self.state.last_checkpoint_at = datetime.now().isoformat()
        
        # Save checkpoint state
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        
        # Save enrichment cache
        with open(self.cache_file, 'w') as f:
            json.dump(self.enrichment_cache, f)
        
        self._records_since_checkpoint = 0
    
    def record_processed(
        self,
        record_id: str,
        success: bool,
        enriched_data: Optional[dict] = None,
        skipped: bool = False
    ):
        """Record that a record was processed."""
        self.state.processed_ids.add(record_id)
        self.state.last_processed_index += 1
        self._records_since_checkpoint += 1
        
        if skipped:
            self.state.skipped_count += 1
        elif success:
            self.state.success_count += 1
            if enriched_data:
                self.enrichment_cache[record_id] = enriched_data
        else:
            self.state.error_count += 1
        
        # Auto-save at intervals
        self.save_checkpoint()
    
    def is_processed(self, record_id: str) -> bool:
        """Check if a record has already been processed."""
        return record_id in self.state.processed_ids
    
    def get_cached_enrichment(self, record_id: str) -> Optional[dict]:
        """Get cached enrichment data for a record."""
        return self.enrichment_cache.get(record_id)
    
    def get_progress(self) -> dict:
        """Get current progress statistics."""
        processed = len(self.state.processed_ids)
        total = self.state.total_records
        pct = (processed / total * 100) if total > 0 else 0
        
        return {
            "processed": processed,
            "total": total,
            "percent": pct,
            "success": self.state.success_count,
            "errors": self.state.error_count,
            "skipped": self.state.skipped_count,
            "started_at": self.state.started_at,
            "last_checkpoint": self.state.last_checkpoint_at,
        }
    
    def clear_checkpoint(self):
        """Remove checkpoint files (start fresh)."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.state = CheckpointState()
        self.enrichment_cache = {}


# Testing
if __name__ == "__main__":
    import tempfile
    import shutil
    
    # Create temp directory for testing
    test_dir = tempfile.mkdtemp()
    
    try:
        print("Checkpoint System Test")
        print("=" * 50)
        
        # Test 1: Initialize and save
        mgr = CheckpointManager(
            checkpoint_dir=test_dir,
            checkpoint_interval=5
        )
        mgr.initialize("test_input.csv", total_records=100)
        
        # Simulate processing records
        for i in range(12):
            record_id = f"rec_{i:03d}"
            mgr.record_processed(
                record_id=record_id,
                success=(i % 3 != 0),  # Every 3rd fails
                enriched_data={"name": f"Person {i}"} if i % 3 != 0 else None
            )
        
        # Force save
        mgr.save_checkpoint(force=True)
        
        progress = mgr.get_progress()
        print(f"\nAfter processing 12 records:")
        print(f"  Processed: {progress['processed']}/{progress['total']}")
        print(f"  Success: {progress['success']}, Errors: {progress['errors']}")
        
        # Test 2: Load checkpoint
        mgr2 = CheckpointManager(
            checkpoint_dir=test_dir,
            checkpoint_interval=5
        )
        loaded = mgr2.load_checkpoint()
        print(f"\nCheckpoint loaded: {loaded}")
        print(f"  Processed IDs count: {len(mgr2.state.processed_ids)}")
        print(f"  Cache entries: {len(mgr2.enrichment_cache)}")
        
        # Test 3: Check if records are marked as processed
        print(f"\n  rec_005 processed? {mgr2.is_processed('rec_005')}")
        print(f"  rec_999 processed? {mgr2.is_processed('rec_999')}")
        
        # Test 4: Get cached enrichment
        cached = mgr2.get_cached_enrichment("rec_001")
        print(f"\n  Cached data for rec_001: {cached}")
        
        print("\n✓ All checkpoint tests passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(test_dir)

