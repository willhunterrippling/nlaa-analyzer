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
