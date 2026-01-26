"""
Snowflake Enrichment Module

Queries LinkedIn profile data from Snowflake MIXRANK_LEADS_DATA table
instead of calling the MixRank API directly.

Key features:
- Batch queries (10k URLs at a time) for efficiency
- URL normalization to handle format differences
- Double-encoded JSON parsing for JOB_EXPERIENCE_ARR
- Compatible with existing EnrichmentResult interface
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import snowflake.connector
from dotenv import load_dotenv

from url_normalizer import extract_linkedin_slug

load_dotenv()


@dataclass
class EnrichmentResult:
    """Result of a single enrichment lookup (compatible with MixRank version)."""
    linkedin_url: str
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    http_status: Optional[int] = None  # Not used for Snowflake, kept for compatibility


class SnowflakeEnricher:
    """
    Query Snowflake MIXRANK_LEADS_DATA for LinkedIn profile enrichment.
    
    Replaces MixRankEnricher for cost savings while maintaining the same interface.
    """
    
    TABLE = "PROD_RIPPLING_DWH.GROWTH.MIXRANK_LEADS_DATA"
    
    def __init__(self, batch_size: int = 10000):
        """
        Initialize Snowflake connection.
        
        Requires environment variables:
        - SNOWFLAKE_ACCOUNT
        - SNOWFLAKE_USER  
        - SNOWFLAKE_PASSWORD
        - SNOWFLAKE_WAREHOUSE
        - SNOWFLAKE_DATABASE (optional, defaults to PROD_RIPPLING_DWH)
        - SNOWFLAKE_SCHEMA (optional, defaults to GROWTH)
        """
        self.batch_size = batch_size
        self._conn = None
        
        # Statistics
        self.stats = {
            "total_queries": 0,
            "total_urls_queried": 0,
            "found": 0,
            "not_found": 0,
            "no_experience_data": 0,
            "parse_errors": 0,
        }
    
    @property
    def conn(self):
        """Lazy connection initialization."""
        if self._conn is None:
            # Check for authentication method
            authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", 
                                          os.environ.get("GROWTH_SNOWFLAKE_AUTHENTICATOR"))
            
            conn_params = {
                "account": os.environ["SNOWFLAKE_ACCOUNT"],
                "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
                "database": os.environ.get("SNOWFLAKE_DATABASE", "PROD_RIPPLING_DWH"),
                "schema": os.environ.get("SNOWFLAKE_SCHEMA", "GROWTH"),
            }
            
            # Add role if specified
            if os.environ.get("SNOWFLAKE_ROLE"):
                conn_params["role"] = os.environ["SNOWFLAKE_ROLE"]
            
            # Handle different authentication methods
            if authenticator and authenticator.upper() == "EXTERNALBROWSER":
                # SSO/Browser-based authentication
                conn_params["authenticator"] = "externalbrowser"
                # User might be optional for SSO, but include if present
                if os.environ.get("SNOWFLAKE_USER"):
                    conn_params["user"] = os.environ["SNOWFLAKE_USER"]
            else:
                # Username/password authentication
                conn_params["user"] = os.environ["SNOWFLAKE_USER"]
                conn_params["password"] = os.environ["SNOWFLAKE_PASSWORD"]
            
            self._conn = snowflake.connector.connect(**conn_params)
        return self._conn
    
    def close(self):
        """Close the Snowflake connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
    
    def _normalize_url_for_snowflake(self, url: str) -> Optional[str]:
        """
        Normalize LinkedIn URL to Snowflake format.
        
        SFDC uses: https://www.linkedin.com/in/slug
        Snowflake uses: https://linkedin.com/in/slug (no www)
        
        Returns normalized URL or None if invalid.
        """
        slug = extract_linkedin_slug(url)
        if slug:
            return f"https://linkedin.com/in/{slug}"
        return None
    
    def enrich_profile(self, linkedin_url: str) -> EnrichmentResult:
        """
        Enrich a single LinkedIn profile.
        
        For single lookups, wraps enrich_batch for simplicity.
        For bulk operations, use enrich_batch directly for efficiency.
        """
        results = self.enrich_batch({linkedin_url: linkedin_url})
        return results.get(linkedin_url, EnrichmentResult(
            linkedin_url=linkedin_url,
            success=False,
            error="URL processing failed"
        ))
    
    def enrich_batch(self, url_to_id: dict[str, str]) -> dict[str, EnrichmentResult]:
        """
        Query Snowflake for a batch of LinkedIn URLs.
        
        Args:
            url_to_id: Dict mapping LinkedIn URL -> record ID
                       (record ID is used for cache keying)
        
        Returns:
            Dict mapping original LinkedIn URL -> EnrichmentResult
        """
        if not url_to_id:
            return {}
        
        # Normalize URLs to Snowflake format
        normalized_to_original: dict[str, str] = {}
        for url in url_to_id.keys():
            sf_url = self._normalize_url_for_snowflake(url)
            if sf_url:
                normalized_to_original[sf_url] = url
        
        if not normalized_to_original:
            # All URLs failed normalization
            return {
                url: EnrichmentResult(
                    linkedin_url=url,
                    success=False,
                    error="Invalid LinkedIn URL format"
                )
                for url in url_to_id.keys()
            }
        
        # Build and execute query
        normalized_urls = list(normalized_to_original.keys())
        placeholders = ','.join(['%s'] * len(normalized_urls))
        
        query = f"""
        SELECT LINKEDIN_URL, JOB_EXPERIENCE_ARR
        FROM {self.TABLE}
        WHERE LINKEDIN_URL IN ({placeholders})
        """
        
        self.stats["total_queries"] += 1
        self.stats["total_urls_queried"] += len(normalized_urls)
        
        cursor = self.conn.cursor()
        cursor.execute(query, normalized_urls)
        
        results: dict[str, EnrichmentResult] = {}
        found_sf_urls: set[str] = set()
        
        for row in cursor:
            sf_url = row[0]
            exp_arr = row[1]
            
            found_sf_urls.add(sf_url)
            original_url = normalized_to_original.get(sf_url, sf_url)
            
            # Parse experience data
            experiences = self._parse_experience_arr(exp_arr)
            
            if experiences:
                self.stats["found"] += 1
                results[original_url] = EnrichmentResult(
                    linkedin_url=original_url,
                    success=True,
                    data={'experience': experiences}
                )
            else:
                self.stats["no_experience_data"] += 1
                results[original_url] = EnrichmentResult(
                    linkedin_url=original_url,
                    success=False,
                    error="No experience data in Snowflake record"
                )
        
        cursor.close()
        
        # Mark URLs not found in Snowflake as failures
        for sf_url, original_url in normalized_to_original.items():
            if sf_url not in found_sf_urls:
                self.stats["not_found"] += 1
                results[original_url] = EnrichmentResult(
                    linkedin_url=original_url,
                    success=False,
                    error="LinkedIn URL not found in Snowflake"
                )
        
        # Mark URLs that failed normalization
        for url in url_to_id.keys():
            if url not in results:
                results[url] = EnrichmentResult(
                    linkedin_url=url,
                    success=False,
                    error="Invalid LinkedIn URL format"
                )
        
        return results
    
    def _parse_experience_arr(self, exp_arr: str) -> list[dict]:
        """
        Parse JOB_EXPERIENCE_ARR from Snowflake to matcher-compatible format.
        
        Snowflake stores this as DOUBLE-ENCODED JSON:
        - Outer: JSON array of strings
        - Inner: Each string is a JSON object
        
        Example raw value:
        '["{\"company_name\":\"Acme\",\"title\":\"Engineer\",\"is_current\":true}"]'
        
        Converts to matcher format:
        [{"company": "Acme", "title": "Engineer", "is_current": True}]
        """
        if not exp_arr:
            return []
        
        # Handle string representations of empty/null
        if isinstance(exp_arr, str):
            exp_arr = exp_arr.strip()
            if exp_arr in ('', '[]', 'null', 'None', 'NULL'):
                return []
        
        try:
            # First parse: string -> list
            outer = json.loads(exp_arr)
            
            if not outer or not isinstance(outer, list):
                return []
            
            experiences = []
            for item in outer:
                # Handle double-encoding: each item might be a string that needs parsing
                if isinstance(item, str):
                    try:
                        exp = json.loads(item)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(item, dict):
                    exp = item
                else:
                    continue
                
                # Map Snowflake field names to matcher format
                experiences.append({
                    'company': exp.get('company_name', ''),
                    'title': exp.get('title', ''),
                    'start_date': exp.get('start_date'),
                    'end_date': exp.get('end_date'),
                    'is_current': exp.get('is_current', False),
                    # Preserve additional fields that might be useful
                    'locality': exp.get('locality'),
                    'seniority': exp.get('seniority'),
                })
            
            return experiences
            
        except (json.JSONDecodeError, TypeError) as e:
            self.stats["parse_errors"] += 1
            return []
    
    def get_stats(self) -> dict:
        """Get current statistics."""
        return self.stats.copy()
    
    def test_connection(self) -> tuple[bool, str]:
        """
        Test Snowflake connection and table access.
        
        Returns:
            (success, message)
        """
        try:
            cursor = self.conn.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT CURRENT_TIMESTAMP()")
            timestamp = cursor.fetchone()[0]
            
            # Test table access
            cursor.execute(f"SELECT COUNT(*) FROM {self.TABLE} LIMIT 1")
            
            # Test a sample query
            cursor.execute(f"""
                SELECT LINKEDIN_URL, JOB_EXPERIENCE_ARR 
                FROM {self.TABLE} 
                WHERE JOB_EXPERIENCE_ARR IS NOT NULL 
                LIMIT 1
            """)
            sample = cursor.fetchone()
            
            cursor.close()
            
            if sample:
                return True, f"Connected successfully. Sample URL: {sample[0][:50]}..."
            else:
                return True, "Connected successfully. Table accessible but no sample data found."
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"


class HybridEnricher:
    """
    Hybrid enricher that tries Snowflake first, falls back to MixRank API.
    
    Strategy:
    1. Query Snowflake in batch (fast, free)
    2. For records with no Snowflake data, fall back to MixRank API (slower, costly)
    3. Track stats for both sources
    
    This gives ~85-90% cost savings while maintaining full coverage.
    """
    
    def __init__(
        self,
        snowflake_batch_size: int = 10000,
        mixrank_rate_limit: float = 2.0,
        enable_mixrank_fallback: bool = True,
    ):
        """
        Initialize hybrid enricher with both Snowflake and MixRank.
        
        Args:
            snowflake_batch_size: Batch size for Snowflake queries
            mixrank_rate_limit: Rate limit for MixRank API (requests per second)
            enable_mixrank_fallback: If False, skip MixRank and just report missing
        """
        from enrichment import MixRankEnricher
        
        self.snowflake_enricher = SnowflakeEnricher(batch_size=snowflake_batch_size)
        self.enable_mixrank_fallback = enable_mixrank_fallback
        self.mixrank_enricher = None
        
        if enable_mixrank_fallback:
            try:
                self.mixrank_enricher = MixRankEnricher(requests_per_second=mixrank_rate_limit)
            except ValueError as e:
                print(f"   ⚠ MixRank fallback disabled: {e}")
                self.enable_mixrank_fallback = False
        
        self.stats = {
            "total_urls": 0,
            "snowflake_found": 0,
            "snowflake_missing": 0,
            "mixrank_called": 0,
            "mixrank_success": 0,
            "mixrank_failed": 0,
            "total_success": 0,
            "total_failed": 0,
        }
        
        # Track URLs not found in Snowflake (had to use MixRank fallback)
        self.snowflake_missing_urls: list[dict] = []
    
    def test_connection(self) -> tuple[bool, str]:
        """Test connections to both Snowflake and MixRank."""
        sf_success, sf_msg = self.snowflake_enricher.test_connection()
        
        if not sf_success:
            return False, f"Snowflake: {sf_msg}"
        
        result_msg = f"Snowflake: {sf_msg}"
        
        if self.enable_mixrank_fallback:
            result_msg += " | MixRank fallback: enabled"
        else:
            result_msg += " | MixRank fallback: disabled"
        
        return True, result_msg
    
    def enrich_batch(self, url_to_id: dict[str, str]) -> dict[str, EnrichmentResult]:
        """
        Enrich a batch of URLs using Snowflake first, MixRank fallback for missing.
        
        Args:
            url_to_id: Mapping of LinkedIn URLs to record IDs
            
        Returns:
            Dict mapping original URLs to EnrichmentResults
        """
        if not url_to_id:
            return {}
        
        self.stats["total_urls"] += len(url_to_id)
        results = {}
        
        # Step 1: Query Snowflake for all URLs
        sf_results = self.snowflake_enricher.enrich_batch(url_to_id)
        
        # Separate successes from failures
        missing_urls = {}
        for url, result in sf_results.items():
            if result.success and result.data and result.data.get('experience'):
                # Snowflake has data
                results[url] = result
                self.stats["snowflake_found"] += 1
            else:
                # Snowflake missing - need fallback
                missing_urls[url] = url_to_id.get(url, url)
                self.stats["snowflake_missing"] += 1
        
        # Step 2: Fall back to MixRank for missing URLs
        if missing_urls and self.enable_mixrank_fallback and self.mixrank_enricher:
            print(f"   📡 Falling back to MixRank API for {len(missing_urls)} URLs...")
            
            for url in missing_urls:
                record_id = url_to_id.get(url, url)
                self.stats["mixrank_called"] += 1
                
                try:
                    mr_result = self.mixrank_enricher.enrich_profile(url)
                    
                    # Convert MixRank result to our format (they should be compatible)
                    results[url] = EnrichmentResult(
                        linkedin_url=url,
                        success=mr_result.success,
                        data=mr_result.data,
                        error=mr_result.error,
                        http_status=mr_result.http_status,
                    )
                    
                    if mr_result.success:
                        self.stats["mixrank_success"] += 1
                        # Track this URL as missing from Snowflake (but found via MixRank)
                        self.snowflake_missing_urls.append({
                            'record_id': record_id,
                            'linkedin_url': url,
                            'source': 'mixrank_api',
                            'mixrank_success': True,
                        })
                    else:
                        self.stats["mixrank_failed"] += 1
                        # Track this URL as missing from both sources
                        self.snowflake_missing_urls.append({
                            'record_id': record_id,
                            'linkedin_url': url,
                            'source': 'mixrank_api',
                            'mixrank_success': False,
                            'mixrank_error': mr_result.error,
                        })
                        
                except Exception as e:
                    results[url] = EnrichmentResult(
                        linkedin_url=url,
                        success=False,
                        error=f"MixRank error: {str(e)}",
                    )
                    self.stats["mixrank_failed"] += 1
                    # Track this URL as missing from both sources
                    self.snowflake_missing_urls.append({
                        'record_id': record_id,
                        'linkedin_url': url,
                        'source': 'mixrank_api',
                        'mixrank_success': False,
                        'mixrank_error': str(e),
                    })
        else:
            # No MixRank fallback - mark as failed
            for url in missing_urls:
                record_id = url_to_id.get(url, url)
                results[url] = EnrichmentResult(
                    linkedin_url=url,
                    success=False,
                    error="Not found in Snowflake (MixRank fallback disabled)",
                )
                # Track this URL as missing from Snowflake (no MixRank fallback)
                self.snowflake_missing_urls.append({
                    'record_id': record_id,
                    'linkedin_url': url,
                    'source': 'snowflake_only',
                    'mixrank_success': False,
                })
        
        # Update totals
        self.stats["total_success"] = self.stats["snowflake_found"] + self.stats["mixrank_success"]
        self.stats["total_failed"] = self.stats["mixrank_failed"] + (
            self.stats["snowflake_missing"] - self.stats["mixrank_called"]
        )
        
        return results
    
    def enrich_profile(self, linkedin_url: str) -> EnrichmentResult:
        """Enrich a single profile (convenience wrapper)."""
        results = self.enrich_batch({linkedin_url: linkedin_url})
        return results.get(linkedin_url, EnrichmentResult(
            linkedin_url=linkedin_url,
            success=False,
            error="Unknown error",
        ))
    
    def get_stats(self) -> dict:
        """Get combined stats from both enrichers."""
        stats = self.stats.copy()
        stats["snowflake_stats"] = self.snowflake_enricher.get_stats()
        if self.mixrank_enricher:
            stats["mixrank_stats"] = {
                "total_calls": self.stats["mixrank_called"],
                "success": self.stats["mixrank_success"],
                "failed": self.stats["mixrank_failed"],
            }
        return stats
    
    def close(self):
        """Close connections."""
        self.snowflake_enricher.close()
    
    def print_stats(self):
        """Print a summary of enrichment sources."""
        total = self.stats["total_urls"]
        if total == 0:
            return
        
        sf_found = self.stats["snowflake_found"]
        sf_missing = self.stats["snowflake_missing"]
        mr_success = self.stats["mixrank_success"]
        mr_failed = self.stats["mixrank_failed"]
        
        print(f"\n📊 Enrichment Source Summary:")
        print(f"   Total URLs: {total:,}")
        print(f"   ✓ Snowflake: {sf_found:,} ({sf_found/total*100:.1f}%)")
        if self.enable_mixrank_fallback:
            print(f"   ✓ MixRank fallback: {mr_success:,} ({mr_success/total*100:.1f}%)")
            print(f"   ✗ Failed: {mr_failed:,} ({mr_failed/total*100:.1f}%)")
        else:
            print(f"   ✗ Not in Snowflake: {sf_missing:,} ({sf_missing/total*100:.1f}%)")
    
    def get_snowflake_missing_urls(self) -> list[dict]:
        """
        Get list of URLs that were not found in Snowflake.
        
        Returns list of dicts with:
        - record_id: The original record ID
        - linkedin_url: The LinkedIn profile URL
        - source: 'mixrank_api' or 'snowflake_only'
        - mixrank_success: Whether MixRank API call succeeded (if applicable)
        - mixrank_error: Error message if MixRank failed
        """
        return self.snowflake_missing_urls.copy()
    
    def export_snowflake_missing(self, filepath: str) -> int:
        """
        Export URLs not found in Snowflake to a CSV file.
        
        Args:
            filepath: Path to output CSV file
            
        Returns:
            Number of records exported
        """
        import csv
        
        if not self.snowflake_missing_urls:
            return 0
        
        fieldnames = ['record_id', 'linkedin_url', 'source', 'mixrank_success', 'mixrank_error']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.snowflake_missing_urls)
        
        return len(self.snowflake_missing_urls)


class AsyncHybridEnricher:
    """
    Async hybrid enricher with concurrent MixRank API calls.
    
    Strategy:
    1. Query Snowflake in batch (fast, free)
    2. For missing records, call MixRank API concurrently (up to 50 at a time)
    3. Track stats from both sources
    
    With 50 concurrent MixRank requests, ~100k fallback URLs can be processed in ~30 min
    instead of 15+ hours with sequential calls.
    """
    
    def __init__(
        self,
        snowflake_batch_size: int = 10000,
        mixrank_max_concurrent: int = 50,
        enable_mixrank_fallback: bool = True,
    ):
        """
        Initialize async hybrid enricher.
        
        Args:
            snowflake_batch_size: Batch size for Snowflake queries
            mixrank_max_concurrent: Max concurrent MixRank API calls (default 50, max 100)
            enable_mixrank_fallback: If False, skip MixRank and just report missing
        """
        from enrichment import AsyncMixRankEnricher
        
        self.snowflake_enricher = SnowflakeEnricher(batch_size=snowflake_batch_size)
        self.enable_mixrank_fallback = enable_mixrank_fallback
        self.mixrank_enricher = None
        self.mixrank_max_concurrent = min(mixrank_max_concurrent, 100)  # Cap at 100
        
        if enable_mixrank_fallback:
            try:
                self.mixrank_enricher = AsyncMixRankEnricher(
                    max_concurrent=self.mixrank_max_concurrent
                )
            except ValueError as e:
                print(f"   ⚠ MixRank fallback disabled: {e}")
                self.enable_mixrank_fallback = False
        
        self.stats = {
            "total_urls": 0,
            "snowflake_found": 0,
            "snowflake_missing": 0,
            "mixrank_called": 0,
            "mixrank_success": 0,
            "mixrank_failed": 0,
            "total_success": 0,
            "total_failed": 0,
        }
        
        self.snowflake_missing_urls: list[dict] = []
    
    def test_connection(self) -> tuple[bool, str]:
        """Test Snowflake connection (MixRank tested on first use)."""
        sf_success, sf_msg = self.snowflake_enricher.test_connection()
        
        if not sf_success:
            return False, f"Snowflake: {sf_msg}"
        
        result_msg = f"Snowflake: {sf_msg}"
        
        if self.enable_mixrank_fallback:
            result_msg += f" | MixRank: async ({self.mixrank_max_concurrent} concurrent)"
        else:
            result_msg += " | MixRank fallback: disabled"
        
        return True, result_msg
    
    async def enrich_batch(self, url_to_id: dict[str, str]) -> dict[str, EnrichmentResult]:
        """
        Enrich URLs using Snowflake first, then async MixRank fallback.
        
        Args:
            url_to_id: Mapping of LinkedIn URLs to record IDs
            
        Returns:
            Dict mapping original URLs to EnrichmentResults
        """
        import asyncio
        
        if not url_to_id:
            return {}
        
        self.stats["total_urls"] += len(url_to_id)
        results = {}
        
        # Step 1: Query Snowflake for all URLs (sync - it's already batched)
        sf_results = self.snowflake_enricher.enrich_batch(url_to_id)
        
        # Separate successes from failures
        missing_urls = {}
        for url, result in sf_results.items():
            if result.success and result.data and result.data.get('experience'):
                results[url] = result
                self.stats["snowflake_found"] += 1
            else:
                missing_urls[url] = url_to_id.get(url, url)
                self.stats["snowflake_missing"] += 1
        
        # Step 2: Fall back to MixRank concurrently for missing URLs
        if missing_urls and self.enable_mixrank_fallback and self.mixrank_enricher:
            print(f"   📡 Async MixRank fallback for {len(missing_urls)} URLs ({self.mixrank_max_concurrent} concurrent)...")
            
            self.stats["mixrank_called"] += len(missing_urls)
            
            # Batch MixRank calls
            mr_results = await self.mixrank_enricher.enrich_batch(list(missing_urls.keys()))
            
            for url, mr_result in mr_results.items():
                record_id = url_to_id.get(url, url)
                
                results[url] = EnrichmentResult(
                    linkedin_url=url,
                    success=mr_result.success,
                    data=mr_result.data,
                    error=mr_result.error,
                    http_status=mr_result.http_status,
                )
                
                if mr_result.success:
                    self.stats["mixrank_success"] += 1
                    self.snowflake_missing_urls.append({
                        'record_id': record_id,
                        'linkedin_url': url,
                        'source': 'mixrank_api',
                        'mixrank_success': True,
                    })
                else:
                    self.stats["mixrank_failed"] += 1
                    self.snowflake_missing_urls.append({
                        'record_id': record_id,
                        'linkedin_url': url,
                        'source': 'mixrank_api',
                        'mixrank_success': False,
                        'mixrank_error': mr_result.error,
                    })
        else:
            # No MixRank fallback
            for url in missing_urls:
                record_id = url_to_id.get(url, url)
                results[url] = EnrichmentResult(
                    linkedin_url=url,
                    success=False,
                    error="Not found in Snowflake (MixRank fallback disabled)",
                )
                self.snowflake_missing_urls.append({
                    'record_id': record_id,
                    'linkedin_url': url,
                    'source': 'snowflake_only',
                    'mixrank_success': False,
                })
        
        # Update totals
        self.stats["total_success"] = self.stats["snowflake_found"] + self.stats["mixrank_success"]
        self.stats["total_failed"] = self.stats["mixrank_failed"] + (
            self.stats["snowflake_missing"] - self.stats["mixrank_called"]
        )
        
        return results
    
    async def enrich_profile(self, linkedin_url: str) -> EnrichmentResult:
        """Enrich a single profile."""
        results = await self.enrich_batch({linkedin_url: linkedin_url})
        return results.get(linkedin_url, EnrichmentResult(
            linkedin_url=linkedin_url,
            success=False,
            error="Unknown error",
        ))
    
    def get_stats(self) -> dict:
        """Get combined stats."""
        stats = self.stats.copy()
        stats["snowflake_stats"] = self.snowflake_enricher.get_stats()
        if self.mixrank_enricher:
            stats["mixrank_stats"] = self.mixrank_enricher.get_stats()
        return stats
    
    async def close(self):
        """Close connections."""
        self.snowflake_enricher.close()
        if self.mixrank_enricher:
            await self.mixrank_enricher.close()
    
    def print_stats(self):
        """Print summary of enrichment sources."""
        total = self.stats["total_urls"]
        if total == 0:
            return
        
        sf_found = self.stats["snowflake_found"]
        sf_missing = self.stats["snowflake_missing"]
        mr_success = self.stats["mixrank_success"]
        mr_failed = self.stats["mixrank_failed"]
        
        print(f"\n📊 Enrichment Source Summary:")
        print(f"   Total URLs: {total:,}")
        print(f"   ✓ Snowflake: {sf_found:,} ({sf_found/total*100:.1f}%)")
        if self.enable_mixrank_fallback:
            print(f"   ✓ MixRank async: {mr_success:,} ({mr_success/total*100:.1f}%)")
            print(f"   ✗ Failed: {mr_failed:,} ({mr_failed/total*100:.1f}%)")
        else:
            print(f"   ✗ Not in Snowflake: {sf_missing:,} ({sf_missing/total*100:.1f}%)")
    
    def get_snowflake_missing_urls(self) -> list[dict]:
        """Get list of URLs not found in Snowflake."""
        return self.snowflake_missing_urls.copy()
    
    def export_snowflake_missing(self, filepath: str) -> int:
        """Export URLs not found in Snowflake to CSV."""
        import csv
        
        if not self.snowflake_missing_urls:
            return 0
        
        fieldnames = ['record_id', 'linkedin_url', 'source', 'mixrank_success', 'mixrank_error']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.snowflake_missing_urls)
        
        return len(self.snowflake_missing_urls)


# Testing
if __name__ == "__main__":
    print("Snowflake Enrichment Module Test")
    print("=" * 60)
    
    # Test URL normalization
    print("\n1. URL Normalization Tests:")
    enricher = SnowflakeEnricher.__new__(SnowflakeEnricher)
    enricher.stats = {"parse_errors": 0}
    
    test_urls = [
        "https://www.linkedin.com/in/john-doe",
        "https://linkedin.com/in/jane-smith",
        "http://linkedin.com/in/test-user",
        "https://www.linkedin.com/in/user-123abc",
    ]
    
    for url in test_urls:
        normalized = enricher._normalize_url_for_snowflake(url)
        print(f"  {url}")
        print(f"    -> {normalized}")
    
    # Test experience parsing
    print("\n2. Experience Parsing Tests:")
    
    # Double-encoded JSON (actual Snowflake format)
    double_encoded = '''[
        "{\\"company_name\\":\\"Acme Corp\\",\\"title\\":\\"Engineer\\",\\"is_current\\":true}",
        "{\\"company_name\\":\\"Old Job\\",\\"title\\":\\"Intern\\",\\"is_current\\":false}"
    ]'''
    
    parsed = enricher._parse_experience_arr(double_encoded)
    print(f"  Double-encoded JSON -> {len(parsed)} experiences")
    for exp in parsed:
        print(f"    - {exp.get('company')}: {exp.get('title')} (current={exp.get('is_current')})")
    
    # Empty cases
    empty_cases = ['', '[]', 'null', None]
    for case in empty_cases:
        result = enricher._parse_experience_arr(case)
        print(f"  '{case}' -> {result}")
    
    # Test connection (only if env vars are set)
    print("\n3. Connection Test:")
    if os.environ.get("SNOWFLAKE_ACCOUNT"):
        try:
            real_enricher = SnowflakeEnricher()
            success, msg = real_enricher.test_connection()
            print(f"  {'✓' if success else '✗'} {msg}")
            real_enricher.close()
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("  Skipped (SNOWFLAKE_ACCOUNT not set)")
    
    print("\n" + "=" * 60)
    print("✓ Module tests completed!")
