"""
MixRank Enrichment Module

Handles API calls to MixRank with:
- Conservative rate limiting (default 2 req/sec)
- Automatic backoff on errors
- Progress tracking with rich output
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EnrichmentResult:
    """Result of a single enrichment API call."""
    linkedin_url: str
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    http_status: Optional[int] = None


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_second: float = 2.0):
        self.rate = requests_per_second
        self.last_request_time = 0.0
        self.min_interval = 1.0 / requests_per_second
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()


class BackoffStrategy:
    """Exponential backoff for error handling."""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        max_retries: int = 3
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.max_retries = max_retries
        self.current_delay = initial_delay
        self.retry_count = 0
    
    def reset(self):
        """Reset after successful request."""
        self.current_delay = self.initial_delay
        self.retry_count = 0
    
    def should_retry(self) -> bool:
        """Check if we should retry."""
        return self.retry_count < self.max_retries
    
    def wait_and_increment(self):
        """Wait for backoff period and increment counter."""
        time.sleep(self.current_delay)
        self.current_delay = min(
            self.current_delay * self.multiplier,
            self.max_delay
        )
        self.retry_count += 1


class MixRankEnricher:
    """
    MixRank API client for LinkedIn profile enrichment.
    
    Features:
    - Rate limiting to avoid hitting API limits
    - Exponential backoff on errors
    - Detailed progress callbacks
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        requests_per_second: float = 2.0,
        timeout: int = 30
    ):
        self.api_key = api_key or os.environ.get("MIXRANK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MixRank API key required. Set MIXRANK_API_KEY env var or pass api_key parameter."
            )
        
        self.base_url = f"https://api.mixrank.com/v2/json/{self.api_key}/linkedin/profile"
        self.timeout = timeout
        self.rate_limiter = RateLimiter(requests_per_second)
        self.backoff = BackoffStrategy()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "retries": 0,
        }
    
    def enrich_profile(self, linkedin_url: str) -> EnrichmentResult:
        """
        Enrich a single LinkedIn profile.
        
        Handles rate limiting and retries automatically.
        """
        self.stats["total_requests"] += 1
        
        while True:
            # Wait for rate limit
            self.rate_limiter.wait()
            
            try:
                response = requests.get(
                    self.base_url,
                    params={"url": linkedin_url},
                    timeout=self.timeout
                )
                
                # Success
                if response.status_code == 200:
                    self.backoff.reset()
                    self.stats["successful"] += 1
                    return EnrichmentResult(
                        linkedin_url=linkedin_url,
                        success=True,
                        data=response.json(),
                        http_status=200
                    )
                
                # Rate limited - backoff and retry
                if response.status_code == 429:
                    if self.backoff.should_retry():
                        self.stats["retries"] += 1
                        self.backoff.wait_and_increment()
                        continue
                    else:
                        self.stats["failed"] += 1
                        return EnrichmentResult(
                            linkedin_url=linkedin_url,
                            success=False,
                            error="Rate limited - max retries exceeded",
                            http_status=429
                        )
                
                # Server error - backoff and retry
                if response.status_code >= 500:
                    if self.backoff.should_retry():
                        self.stats["retries"] += 1
                        self.backoff.wait_and_increment()
                        continue
                    else:
                        self.stats["failed"] += 1
                        return EnrichmentResult(
                            linkedin_url=linkedin_url,
                            success=False,
                            error=f"Server error: {response.status_code}",
                            http_status=response.status_code
                        )
                
                # Client error (4xx except 429) - don't retry
                self.backoff.reset()
                self.stats["failed"] += 1
                return EnrichmentResult(
                    linkedin_url=linkedin_url,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    http_status=response.status_code
                )
                
            except requests.exceptions.Timeout:
                if self.backoff.should_retry():
                    self.stats["retries"] += 1
                    self.backoff.wait_and_increment()
                    continue
                else:
                    self.stats["failed"] += 1
                    return EnrichmentResult(
                        linkedin_url=linkedin_url,
                        success=False,
                        error="Request timeout - max retries exceeded"
                    )
                    
            except requests.exceptions.RequestException as e:
                self.backoff.reset()
                self.stats["failed"] += 1
                return EnrichmentResult(
                    linkedin_url=linkedin_url,
                    success=False,
                    error=f"Request error: {str(e)}"
                )
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.copy()


class ProgressTracker:
    """
    Track and display progress for long-running operations.
    
    Provides:
    - Progress bar
    - ETA calculation
    - Success/error counts
    - Detailed logging
    """
    
    def __init__(
        self,
        total: int,
        description: str = "Processing",
        update_interval: int = 1,  # Update display every N records
        verbose: bool = False
    ):
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.verbose = verbose
        
        self.processed = 0
        self.success = 0
        self.errors = 0
        self.skipped = 0
        
        self.start_time = time.time()
        self.last_update_time = self.start_time
    
    def update(
        self,
        success: bool = True,
        skipped: bool = False,
        message: Optional[str] = None
    ):
        """Update progress after processing a record."""
        self.processed += 1
        
        if skipped:
            self.skipped += 1
        elif success:
            self.success += 1
        else:
            self.errors += 1
        
        # Print verbose message if enabled
        if self.verbose and message:
            print(f"  [{self.processed}/{self.total}] {message}")
        
        # Update progress display at intervals
        if self.processed % self.update_interval == 0 or self.processed == self.total:
            self._print_progress()
    
    def _print_progress(self):
        """Print progress bar and stats."""
        elapsed = time.time() - self.start_time
        pct = (self.processed / self.total) * 100 if self.total > 0 else 0
        
        # Calculate ETA
        if self.processed > 0:
            rate = self.processed / elapsed
            remaining = self.total - self.processed
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "--:--"
        
        # Build progress bar
        bar_width = 40
        filled = int(bar_width * self.processed / self.total) if self.total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Stats line
        stats = f"✓{self.success} ✗{self.errors} ⊘{self.skipped}"
        
        # Print (overwrite previous line)
        print(
            f"\r{self.description}: {pct:5.1f}% |{bar}| "
            f"{self.processed}/{self.total} [{self._format_time(elapsed)}<{eta_str}] {stats}",
            end="",
            flush=True
        )
        
        # Newline at completion
        if self.processed == self.total:
            print()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        if seconds < 0:
            return "--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def finish(self):
        """Print final summary."""
        elapsed = time.time() - self.start_time
        print()
        print("=" * 60)
        print(f"  {self.description} Complete!")
        print(f"  Total: {self.processed} | Success: {self.success} | "
              f"Errors: {self.errors} | Skipped: {self.skipped}")
        print(f"  Time: {self._format_time(elapsed)}")
        if self.processed > 0:
            rate = self.processed / elapsed
            print(f"  Rate: {rate:.1f} records/sec")
        print("=" * 60)


# Testing (without actual API calls)
if __name__ == "__main__":
    print("Enrichment Module Components Test")
    print("=" * 50)
    
    # Test RateLimiter
    print("\n1. Rate Limiter Test (2 req/sec):")
    limiter = RateLimiter(2.0)
    start = time.time()
    for i in range(5):
        limiter.wait()
        print(f"   Request {i+1} at t={time.time()-start:.2f}s")
    
    # Test ProgressTracker
    print("\n2. Progress Tracker Test:")
    tracker = ProgressTracker(total=20, description="Test", verbose=False)
    for i in range(20):
        success = i % 5 != 0
        skipped = i % 7 == 0
        tracker.update(success=success, skipped=skipped)
        time.sleep(0.05)
    tracker.finish()
    
    # Test BackoffStrategy
    print("\n3. Backoff Strategy Test:")
    backoff = BackoffStrategy(initial_delay=0.1, max_delay=1.0, multiplier=2.0, max_retries=4)
    for i in range(5):
        if backoff.should_retry():
            print(f"   Retry {i+1}: delay={backoff.current_delay:.2f}s")
            backoff.wait_and_increment()
        else:
            print(f"   Max retries reached")
            break
    
    print("\n✓ All component tests passed!")
    print("\nNote: Actual API tests require MIXRANK_API_KEY environment variable.")

