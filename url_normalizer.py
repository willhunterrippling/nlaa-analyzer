"""
LinkedIn URL Normalization Module

Handles the 4 main URL issues found in the SFDC data:
1. missing_domain_prefix (57.7%): `in/-erica-avalos` -> full URL
2. wrong_path_structure (21.8%): `/pub/` URLs -> `/in/` format
3. missing_protocol (20.2%): `//www.linkedin.com/...` -> add https:
4. double_url (<1%): doubled URLs -> extract valid path
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormalizationResult:
    """Result of URL normalization attempt."""
    original: str
    normalized: Optional[str]
    issue_type: Optional[str]
    success: bool
    error: Optional[str] = None


def normalize_linkedin_url(url: str) -> NormalizationResult:
    """
    Normalize a LinkedIn URL to the standard format:
    https://www.linkedin.com/in/<slug>
    
    Returns a NormalizationResult with details about the transformation.
    """
    if not url or not isinstance(url, str):
        return NormalizationResult(
            original=url or "",
            normalized=None,
            issue_type=None,
            success=False,
            error="Empty or invalid URL"
        )
    
    original = url.strip()
    
    # Handle NULL-like values
    if original in ('\\N', 'NULL', 'null', 'None', ''):
        return NormalizationResult(
            original=original,
            normalized=None,
            issue_type=None,
            success=False,
            error="NULL value"
        )
    
    # Issue 1: Double URL (extract the real /in/ path)
    # Example: https://www.linkedin.com/https://www.linkedin.com/in/john-doe
    double_url_match = re.search(r'linkedin\.com.*?(linkedin\.com/in/[^"\s]+)', original, re.IGNORECASE)
    if double_url_match:
        extracted = double_url_match.group(1)
        normalized = f"https://www.{extracted}"
        # Clean up any trailing issues
        normalized = re.sub(r'/+$', '', normalized)
        return NormalizationResult(
            original=original,
            normalized=normalized,
            issue_type="double_url",
            success=True
        )
    
    # Issue 2: Wrong path structure (/pub/ URLs)
    # Example: linkedin.com/pub/kobi-yehiel/16/1bb/a75 -> /in/kobi-yehiel-a751bb16
    pub_match = re.search(r'linkedin\.com/pub/([^/]+)/([^/]+)/([^/]+)/([^/\s"]+)', original, re.IGNORECASE)
    if pub_match:
        name, part1, part2, part3 = pub_match.groups()
        # Convert /pub/name/a/b/c to /in/name-cba format
        slug = f"{name}-{part3}{part2}{part1}"
        normalized = f"https://www.linkedin.com/in/{slug}"
        return NormalizationResult(
            original=original,
            normalized=normalized,
            issue_type="wrong_path_structure",
            success=True
        )
    
    # Issue 3: Missing protocol (starts with //)
    # Example: //www.linkedin.com/in/aaron-freed
    if original.startswith('//'):
        normalized = f"https:{original}"
        return NormalizationResult(
            original=original,
            normalized=normalized,
            issue_type="missing_protocol",
            success=True
        )
    
    # Issue 4: Missing domain prefix (bare slug)
    # Example: in/-erica-avalos or in/john-doe
    if original.lower().startswith('in/') and 'linkedin.com' not in original.lower():
        normalized = f"https://www.linkedin.com/{original}"
        return NormalizationResult(
            original=original,
            normalized=normalized,
            issue_type="missing_domain_prefix",
            success=True
        )
    
    # Issue 3b: Missing protocol but has domain
    # Example: www.linkedin.com/in/john or linkedin.com/in/john
    if 'linkedin.com' in original.lower() and not original.lower().startswith('http'):
        normalized = f"https://{original.lstrip('/')}"
        return NormalizationResult(
            original=original,
            normalized=normalized,
            issue_type="missing_protocol",
            success=True
        )
    
    # Check if it's already a valid LinkedIn /in/ URL
    if re.search(r'https?://[^/]*linkedin\.com/in/', original, re.IGNORECASE):
        # Already valid, just clean up
        normalized = re.sub(r'/+$', '', original)  # Remove trailing slashes
        return NormalizationResult(
            original=original,
            normalized=normalized,
            issue_type=None,
            success=True
        )
    
    # Unable to normalize - not a recognized format
    return NormalizationResult(
        original=original,
        normalized=None,
        issue_type=None,
        success=False,
        error=f"Unrecognized URL format: {original[:100]}"
    )


def extract_linkedin_slug(url: str) -> Optional[str]:
    """
    Extract the LinkedIn profile slug from a URL.
    Example: https://www.linkedin.com/in/john-doe-123 -> john-doe-123
    """
    match = re.search(r'linkedin\.com/in/([^/?#\s"]+)', url, re.IGNORECASE)
    if match:
        return match.group(1).rstrip('/')
    return None


def is_valid_linkedin_url(url: str) -> bool:
    """Check if a URL is a valid LinkedIn profile URL."""
    if not url:
        return False
    return bool(re.search(r'https?://[^/]*linkedin\.com/in/[^/\s"]+', url, re.IGNORECASE))


# Testing
if __name__ == "__main__":
    test_urls = [
        # Double URL
        "https://www.linkedin.com/https://www.linkedin.com/in/john-larney-b3986b6",
        # Pub URL
        "http://il.linkedin.com/pub/kobi-yehiel/16/1bb/a75",
        "LinkedIn.com/pub/andre-tyagi/7/57/998",
        # Missing protocol
        "//www.linkedin.com/in/aaron-freed",
        "www.linkedin.com/in/chris-corson/",
        # Bare slug
        "in/-erica-avalos",
        "in/john-doe-123",
        # Already valid
        "https://www.linkedin.com/in/kami-stehman",
        "https://linkedin.com/in/komal-sahani-274543191",
        # Invalid
        "",
        "\\N",
        "not-a-url",
    ]
    
    print("LinkedIn URL Normalization Test Results")
    print("=" * 80)
    
    for url in test_urls:
        result = normalize_linkedin_url(url)
        status = "✓" if result.success else "✗"
        print(f"\n{status} Original: {url}")
        if result.success:
            print(f"  Normalized: {result.normalized}")
            if result.issue_type:
                print(f"  Issue fixed: {result.issue_type}")
        else:
            print(f"  Error: {result.error}")

