"""
Hybrid Company Name Matching Engine

Matches SFDC account names to LinkedIn experience company names using:
1. Programmatic matching (fast, handles ~80% of cases)
2. LLM fallback via GPT-5-mini (for ambiguous cases)

Matching Strategy:
- Normalize names: lowercase, strip common suffixes
- Exact match on normalized names
- Substring match (bidirectional)
- Token overlap scoring
- Confidence levels: HIGH, MEDIUM, LOW
"""

import os
import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class MatchConfidence(Enum):
    """Confidence level of a match."""
    HIGH = "high"       # Auto-approve: exact match or >90% similarity
    MEDIUM = "medium"   # Auto-approve: substring match with good token overlap
    LOW = "low"         # Needs LLM review
    NO_MATCH = "no_match"  # No current experience matches


@dataclass
class MatchResult:
    """Result of matching an account name to LinkedIn experiences."""
    account_name: str
    matched: bool
    confidence: MatchConfidence
    matching_experience: Optional[dict] = None
    match_reason: str = ""
    similarity_score: float = 0.0
    llm_used: bool = False
    llm_error: bool = False  # True if LLM call failed
    all_current_companies: list = field(default_factory=list)


# Common company suffixes to strip for normalization
COMPANY_SUFFIXES = [
    # Legal entity types
    r'\bInc\.?$', r'\bIncorporated$', r'\bCorp\.?$', r'\bCorporation$',
    r'\bLLC$', r'\bL\.L\.C\.?$', r'\bLtd\.?$', r'\bLimited$',
    r'\bLLP$', r'\bL\.L\.P\.?$', r'\bLP$', r'\bL\.P\.?$',
    r'\bPLC$', r'\bP\.L\.C\.?$', r'\bGmbH$', r'\bAG$',
    r'\bCo\.?$', r'\bCompany$', r'\b& Co\.?$',
    # Common descriptors
    r'\bGroup$', r'\bHoldings?$', r'\bEnterprises?$',
    r'\bInternational$', r'\bIntl\.?$', r'\bGlobal$',
    r'\bServices?$', r'\bSolutions?$', r'\bTechnolog(?:y|ies)$',
    r'\bConsulting$', r'\bPartners?$', r'\bAssociates?$',
    # Regional/division descriptors (often in SFDC but not LinkedIn)
    r'\bFleet\s*(?:&|and)?\s*Business\s*Solutions?$',
    r'\bNorth\s*America$', r'\bNA$', r'\bEMEA$', r'\bAPAC$',
    r'\bUS$', r'\bUSA$', r'\bAmericas?$',
]

# Compile suffix patterns
SUFFIX_PATTERN = re.compile(
    r'[\s,]*(?:' + '|'.join(COMPANY_SUFFIXES) + r')[\s,]*',
    re.IGNORECASE
)


def normalize_company_name(name: str) -> str:
    """
    Normalize a company name for comparison.
    
    Steps:
    1. Lowercase
    2. Remove common suffixes
    3. Remove punctuation except hyphens
    4. Normalize whitespace
    """
    if not name:
        return ""
    
    # Lowercase
    normalized = name.lower().strip()
    
    # Remove suffixes (repeatedly in case of multiple)
    for _ in range(3):  # Max 3 passes
        prev = normalized
        normalized = SUFFIX_PATTERN.sub(' ', normalized).strip()
        if normalized == prev:
            break
    
    # Remove punctuation except hyphens and spaces
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def get_tokens(name: str) -> set:
    """Get word tokens from a company name."""
    normalized = normalize_company_name(name)
    # Split on spaces and hyphens
    tokens = set(re.split(r'[\s-]+', normalized))
    # Remove empty and very short tokens
    return {t for t in tokens if len(t) > 1}


def get_first_significant_token(name: str) -> str:
    """Get the first significant token (likely the core company name)."""
    normalized = normalize_company_name(name)
    tokens = re.split(r'[\s-]+', normalized)
    # Skip very short tokens and common prefixes
    skip_words = {'the', 'a', 'an', 'one', 'new'}
    for token in tokens:
        if len(token) > 2 and token not in skip_words:
            return token
    return tokens[0] if tokens else ""


def calculate_token_overlap(name1: str, name2: str) -> float:
    """
    Calculate token overlap score between two names.
    Returns a score from 0.0 to 1.0.
    """
    tokens1 = get_tokens(name1)
    tokens2 = get_tokens(name2)
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2
    
    # Jaccard similarity
    jaccard = len(intersection) / len(union) if union else 0.0
    
    # Also consider: what % of smaller set is covered?
    smaller = min(len(tokens1), len(tokens2))
    coverage = len(intersection) / smaller if smaller else 0.0
    
    # Base score (weight coverage higher for substring-like matches)
    base_score = (jaccard + coverage * 2) / 3
    
    # TARGETED BOOST for specific patterns where first-token matching is reliable:
    # 1. Law firms: "DeWitt LLP" should match "DeWitt · Giger, LLP"
    # 2. Simple names: "Google" should match "Google LLC"
    # 
    # But NOT for industry-differentiated names:
    # - "Winco Foods" vs "Winco Mfg." (different industries)
    # - "Clover Management" vs "One Clover" (management indicates business type)
    
    first1 = get_first_significant_token(name1)
    first2 = get_first_significant_token(name2)
    
    if first1 and first2 and first1 == first2 and len(first1) >= 4:
        # Check for business-type words that differentiate companies
        business_type_words = {
            # Industry descriptors
            'foods', 'food', 'mfg', 'manufacturing', 'retail', 'tech', 
            'software', 'hardware', 'bio', 'pharma', 'medical', 'health',
            'management', 'consulting', 'capital', 'ventures', 'financial',
            'insurance', 'bank', 'banking', 'realty', 'properties',
            'construction', 'engineering', 'logistics', 'transport',
            'energy', 'oil', 'gas', 'mining', 'steel', 'auto', 'motors',
            # Business structure words that matter
            'store', 'stores', 'shop', 'restaurant', 'hotels', 'club',
        }
        
        other_tokens1 = tokens1 - {first1}
        other_tokens2 = tokens2 - {first2}
        
        # If EITHER side has a business-type word that the other doesn't share,
        # they're likely different companies (e.g., "Winco Foods" vs "Winco Mfg")
        type_words1 = other_tokens1 & business_type_words
        type_words2 = other_tokens2 & business_type_words
        
        has_business_type_conflict = bool(
            (type_words1 or type_words2) and  # At least one has a type word
            type_words1 != type_words2         # And they don't match
        )
        
        if not has_business_type_conflict:
            base_score = max(base_score, 0.55)
    
    return base_score


def programmatic_match(
    account_name: str,
    experiences: list[dict]
) -> MatchResult:
    """
    Attempt to match account name to experiences using programmatic rules.
    
    Returns MatchResult with confidence level indicating if LLM fallback needed.
    """
    # Filter to current experiences only
    current_experiences = [
        exp for exp in experiences
        if exp.get('is_current', False) or exp.get('end_date') is None
    ]
    
    if not current_experiences:
        return MatchResult(
            account_name=account_name,
            matched=False,
            confidence=MatchConfidence.NO_MATCH,
            match_reason="No current experiences found",
            all_current_companies=[]
        )
    
    # Get all current company names for logging
    all_current = [exp.get('company', '') for exp in current_experiences]
    
    # Normalize account name
    norm_account = normalize_company_name(account_name)
    
    best_match = None
    best_score = 0.0
    best_reason = ""
    best_confidence = MatchConfidence.NO_MATCH
    
    for exp in current_experiences:
        company = exp.get('company', '')
        if not company:
            continue
        
        norm_company = normalize_company_name(company)
        
        # Check 1: Exact match (after normalization)
        if norm_account == norm_company:
            return MatchResult(
                account_name=account_name,
                matched=True,
                confidence=MatchConfidence.HIGH,
                matching_experience=exp,
                match_reason=f"Exact match: '{norm_account}'",
                similarity_score=1.0,
                all_current_companies=all_current
            )
        
        # Check 2: One contains the other (substring match)
        if norm_account in norm_company or norm_company in norm_account:
            # Calculate token overlap for ranking
            score = calculate_token_overlap(account_name, company)
            
            if score > best_score:
                best_score = score
                best_match = exp
                best_confidence = MatchConfidence.HIGH if score > 0.7 else MatchConfidence.MEDIUM
                best_reason = f"Substring match: '{norm_account}' <-> '{norm_company}' (score: {score:.2f})"
        
        # Check 3: High token overlap without substring match
        else:
            score = calculate_token_overlap(account_name, company)
            
            if score > 0.5 and score > best_score:
                best_score = score
                best_match = exp
                best_confidence = MatchConfidence.MEDIUM if score > 0.7 else MatchConfidence.LOW
                best_reason = f"Token overlap: '{norm_account}' <-> '{norm_company}' (score: {score:.2f})"
    
    # Return best match found
    if best_match and best_confidence != MatchConfidence.NO_MATCH:
        return MatchResult(
            account_name=account_name,
            matched=True,
            confidence=best_confidence,
            matching_experience=best_match,
            match_reason=best_reason,
            similarity_score=best_score,
            all_current_companies=all_current
        )
    
    # No good match found - might still be a match but needs LLM
    # Check if there are any partial signals
    for exp in current_experiences:
        company = exp.get('company', '')
        score = calculate_token_overlap(account_name, company)
        
        if score > 0.3:  # Some overlap, might be worth LLM review
            return MatchResult(
                account_name=account_name,
                matched=False,
                confidence=MatchConfidence.LOW,
                matching_experience=exp,
                match_reason=f"Weak signal, needs LLM review: '{account_name}' <-> '{company}' (score: {score:.2f})",
                similarity_score=score,
                all_current_companies=all_current
            )
    
    return MatchResult(
        account_name=account_name,
        matched=False,
        confidence=MatchConfidence.NO_MATCH,
        match_reason="No matching company found in current experiences",
        all_current_companies=all_current
    )


class LLMMatcher:
    """
    LLM-based company name matcher for ambiguous cases.
    Uses GPT-5-mini for cost efficiency.
    """
    
    SYSTEM_PROMPT = """You are an expert at matching company names between Salesforce account names and LinkedIn experience entries.

Your task: Determine if the given account name matches any company in the experience list.

Rules:
1. Account names may have additional text (divisions, regions) not in LinkedIn
2. Company names may be abbreviated or shortened
3. Focus on the core company name, ignoring suffixes like Inc, LLC, Corp
4. A match means the person works at the same company, even if titles differ

Return JSON only:
{"match": true/false, "company": "matching company name or null", "reason": "brief explanation"}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.stats = {"calls": 0, "matches": 0, "no_matches": 0, "errors": 0}
    
    def match(
        self,
        account_name: str,
        experiences: list[dict],
        current_only: bool = True
    ) -> MatchResult:
        """
        Use LLM to determine if account name matches any experience.
        """
        # Filter to current experiences
        if current_only:
            exp_list = [
                exp for exp in experiences
                if exp.get('is_current', False) or exp.get('end_date') is None
            ]
        else:
            exp_list = experiences
        
        if not exp_list:
            return MatchResult(
                account_name=account_name,
                matched=False,
                confidence=MatchConfidence.NO_MATCH,
                match_reason="No experiences to match against",
                llm_used=True
            )
        
        # Build experience list for prompt
        exp_text = "\n".join([
            f"- {exp.get('company', 'Unknown')} (Title: {exp.get('title', 'Unknown')})"
            for exp in exp_list
        ])
        
        user_prompt = f"""Account name: {account_name}

Current LinkedIn experiences:
{exp_text}

Does the account name match any of these companies?"""

        self.stats["calls"] += 1
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            matched = result.get("match", False)
            matched_company = result.get("company")
            reason = result.get("reason", "")
            
            if matched:
                self.stats["matches"] += 1
                # Find the matching experience object
                matching_exp = None
                for exp in exp_list:
                    if exp.get('company') == matched_company:
                        matching_exp = exp
                        break
                
                return MatchResult(
                    account_name=account_name,
                    matched=True,
                    confidence=MatchConfidence.HIGH,  # LLM confirmed
                    matching_experience=matching_exp,
                    match_reason=f"LLM: {reason}",
                    similarity_score=0.9,  # High confidence from LLM
                    llm_used=True,
                    all_current_companies=[e.get('company', '') for e in exp_list]
                )
            else:
                self.stats["no_matches"] += 1
                return MatchResult(
                    account_name=account_name,
                    matched=False,
                    confidence=MatchConfidence.NO_MATCH,
                    match_reason=f"LLM: {reason}",
                    llm_used=True,
                    all_current_companies=[e.get('company', '') for e in exp_list]
                )
                
        except Exception as ex:
            self.stats["errors"] += 1
            return MatchResult(
                account_name=account_name,
                matched=False,
                confidence=MatchConfidence.LOW,
                match_reason=f"LLM error: {str(ex)}",
                llm_used=True,
                llm_error=True,
                all_current_companies=[exp.get('company', '') for exp in exp_list]
            )
    
    def get_stats(self) -> dict:
        return self.stats.copy()


class AsyncLLMMatcher:
    """
    Async LLM-based company name matcher for ambiguous cases.
    Uses AsyncOpenAI for non-blocking API calls.
    """
    
    SYSTEM_PROMPT = """You are an expert at matching company names between Salesforce account names and LinkedIn experience entries.

Your task: Determine if the given account name matches any company in the experience list.

Rules:
1. Account names may have additional text (divisions, regions) not in LinkedIn
2. Company names may be abbreviated or shortened
3. Focus on the core company name, ignoring suffixes like Inc, LLC, Corp
4. A match means the person works at the same company, even if titles differ

Return JSON only:
{"match": true/false, "company": "matching company name or null", "reason": "brief explanation"}"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.stats = {"calls": 0, "matches": 0, "no_matches": 0, "errors": 0}
    
    async def match(
        self,
        account_name: str,
        experiences: list[dict],
        current_only: bool = True
    ) -> MatchResult:
        """
        Use LLM to determine if account name matches any experience.
        Non-blocking async implementation.
        """
        # Filter to current experiences
        if current_only:
            exp_list = [
                exp for exp in experiences
                if exp.get('is_current', False) or exp.get('end_date') is None
            ]
        else:
            exp_list = experiences
        
        if not exp_list:
            return MatchResult(
                account_name=account_name,
                matched=False,
                confidence=MatchConfidence.NO_MATCH,
                match_reason="No experiences to match against",
                llm_used=True
            )
        
        # Build experience list for prompt
        exp_text = "\n".join([
            f"- {exp.get('company', 'Unknown')} (Title: {exp.get('title', 'Unknown')})"
            for exp in exp_list
        ])
        
        user_prompt = f"""Account name: {account_name}

Current LinkedIn experiences:
{exp_text}

Does the account name match any of these companies?"""

        self.stats["calls"] += 1
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=200
            )
            
            result = json.loads(response.choices[0].message.content)
            
            matched = result.get("match", False)
            matched_company = result.get("company")
            reason = result.get("reason", "")
            
            if matched:
                self.stats["matches"] += 1
                # Find the matching experience object
                matching_exp = None
                for exp in exp_list:
                    if exp.get('company') == matched_company:
                        matching_exp = exp
                        break
                
                return MatchResult(
                    account_name=account_name,
                    matched=True,
                    confidence=MatchConfidence.HIGH,  # LLM confirmed
                    matching_experience=matching_exp,
                    match_reason=f"LLM: {reason}",
                    similarity_score=0.9,  # High confidence from LLM
                    llm_used=True,
                    all_current_companies=[e.get('company', '') for e in exp_list]
                )
            else:
                self.stats["no_matches"] += 1
                return MatchResult(
                    account_name=account_name,
                    matched=False,
                    confidence=MatchConfidence.NO_MATCH,
                    match_reason=f"LLM: {reason}",
                    llm_used=True,
                    all_current_companies=[e.get('company', '') for e in exp_list]
                )
                
        except Exception as ex:
            self.stats["errors"] += 1
            return MatchResult(
                account_name=account_name,
                matched=False,
                confidence=MatchConfidence.LOW,
                match_reason=f"LLM error: {str(ex)}",
                llm_used=True,
                llm_error=True,
                all_current_companies=[exp.get('company', '') for exp in exp_list]
            )
    
    def get_stats(self) -> dict:
        return self.stats.copy()


class HybridMatcher:
    """
    Hybrid matching engine combining programmatic and LLM approaches.
    
    Strategy:
    1. Try programmatic match first
    2. If HIGH or MEDIUM confidence -> return result
    3. If LOW confidence -> use LLM fallback
    4. If NO_MATCH with partial signals -> use LLM fallback
    """
    
    def __init__(self, use_llm: bool = True, model: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_matcher = LLMMatcher(model=model) if use_llm else None
        
        self.stats = {
            "total": 0,
            "programmatic_matches": 0,
            "llm_matches": 0,
            "no_matches": 0,
            "llm_calls": 0,
        }
    
    def match(
        self,
        account_name: str,
        experiences: list[dict]
    ) -> MatchResult:
        """
        Match account name to experiences using hybrid approach.
        """
        self.stats["total"] += 1
        
        # Step 1: Try programmatic match
        result = programmatic_match(account_name, experiences)
        
        # Step 2: Check if we need LLM fallback
        if result.confidence in (MatchConfidence.HIGH, MatchConfidence.MEDIUM):
            # Good programmatic match - use it
            if result.matched:
                self.stats["programmatic_matches"] += 1
            return result
        
        # Step 3: Use LLM for low confidence or ambiguous cases
        if self.use_llm and result.confidence == MatchConfidence.LOW:
            self.stats["llm_calls"] += 1
            llm_result = self.llm_matcher.match(account_name, experiences)
            
            if llm_result.matched:
                self.stats["llm_matches"] += 1
            else:
                self.stats["no_matches"] += 1
            
            return llm_result
        
        # No match found
        self.stats["no_matches"] += 1
        return result
    
    def get_stats(self) -> dict:
        stats = self.stats.copy()
        if self.llm_matcher:
            stats["llm_stats"] = self.llm_matcher.get_stats()
        return stats


class AsyncHybridMatcher:
    """
    Async hybrid matching engine combining programmatic and LLM approaches.
    
    Strategy:
    1. Try programmatic match first (synchronous, fast)
    2. If HIGH or MEDIUM confidence -> return result
    3. If LOW confidence -> use async LLM fallback
    4. If NO_MATCH with partial signals -> use async LLM fallback
    """
    
    def __init__(self, use_llm: bool = True, model: str = "gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_matcher = AsyncLLMMatcher(model=model) if use_llm else None
        
        self.stats = {
            "total": 0,
            "programmatic_matches": 0,
            "llm_matches": 0,
            "no_matches": 0,
            "llm_calls": 0,
        }
    
    async def match(
        self,
        account_name: str,
        experiences: list[dict]
    ) -> MatchResult:
        """
        Match account name to experiences using hybrid approach.
        Async method that yields to event loop during LLM calls.
        """
        self.stats["total"] += 1
        
        # Step 1: Try programmatic match (fast, synchronous)
        result = programmatic_match(account_name, experiences)
        
        # Step 2: Check if we need LLM fallback
        if result.confidence in (MatchConfidence.HIGH, MatchConfidence.MEDIUM):
            # Good programmatic match - use it
            if result.matched:
                self.stats["programmatic_matches"] += 1
            return result
        
        # Step 3: Use async LLM for low confidence or ambiguous cases
        if self.use_llm and result.confidence == MatchConfidence.LOW:
            self.stats["llm_calls"] += 1
            llm_result = await self.llm_matcher.match(account_name, experiences)
            
            if llm_result.matched:
                self.stats["llm_matches"] += 1
            else:
                self.stats["no_matches"] += 1
            
            return llm_result
        
        # No match found
        self.stats["no_matches"] += 1
        return result
    
    def get_stats(self) -> dict:
        stats = self.stats.copy()
        if self.llm_matcher:
            stats["llm_stats"] = self.llm_matcher.get_stats()
        return stats


# Testing
if __name__ == "__main__":
    print("Company Name Matching Engine Tests")
    print("=" * 60)
    
    # Test normalization
    print("\n1. Normalization Tests:")
    test_names = [
        "Stellantis Fleet & Business Solutions",
        "Stellantis, Inc.",
        "STELLANTIS",
        "stellantis",
        "ABC Company, LLC",
        "ABC Co.",
        "XYZ International Holdings Ltd.",
    ]
    for name in test_names:
        print(f"  '{name}' -> '{normalize_company_name(name)}'")
    
    # Test token overlap
    print("\n2. Token Overlap Tests:")
    pairs = [
        ("Stellantis Fleet & Business Solutions", "Stellantis"),
        ("ABC Company LLC", "ABC Corp"),
        ("Microsoft Corporation", "Google Inc"),
        ("Acme Inc", "Acme Industries"),
    ]
    for n1, n2 in pairs:
        score = calculate_token_overlap(n1, n2)
        print(f"  '{n1}' <-> '{n2}': {score:.2f}")
    
    # Test programmatic matching
    print("\n3. Programmatic Matching Tests:")
    
    test_experiences = [
        {
            "company": "Stellantis",
            "title": "Sales Manager",
            "is_current": True,
            "start_date": "2022-01-01",
            "end_date": None
        },
        {
            "company": "Ford Motor Company",
            "title": "Account Executive",
            "is_current": False,
            "start_date": "2018-01-01",
            "end_date": "2021-12-31"
        },
        {
            "company": "Tesla",
            "title": "Regional Director",
            "is_current": True,
            "start_date": "2023-06-01",
            "end_date": None
        }
    ]
    
    test_accounts = [
        "Stellantis Fleet & Business Solutions",  # Should match Stellantis
        "Stellantis",                              # Exact match
        "Ford Motor Company",                      # Not current
        "General Motors",                          # No match
        "Tesla, Inc.",                             # Should match Tesla
    ]
    
    for account in test_accounts:
        result = programmatic_match(account, test_experiences)
        status = "✓ MATCH" if result.matched else "✗ NO MATCH"
        print(f"\n  Account: '{account}'")
        print(f"  Result: {status} ({result.confidence.value})")
        print(f"  Reason: {result.match_reason}")
        if result.matching_experience:
            print(f"  Matched to: {result.matching_experience.get('company')}")
    
    print("\n" + "=" * 60)
    print("✓ All matching tests completed!")
    print("\nNote: LLM tests require OPENAI_API_KEY environment variable.")

