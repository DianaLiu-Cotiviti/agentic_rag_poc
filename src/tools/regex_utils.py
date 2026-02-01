"""
Regular expression tools for extracting from NCCI manual text:
- CPT codes (single 5-digit codes)
- CPT ranges (e.g., 64400-64530)
- Modifiers (e.g., 24, 25, 59, XE, LT, RT, etc.)
"""
import regex as re
from typing import List, Set, Dict, Any


# CPT codes: 5-digit numbers, typically between 10000-99999
# Avoid false matches with years, page numbers, etc.
CPT_SINGLE_RE = re.compile(
    r'\b([1-9]\d{4})\b'  # 5 digits starting with 1-9
)

# CPT ranges: format like "64400-64530" or "64400 through 64530"
CPT_RANGE_RE = re.compile(
    r'\b([1-9]\d{4})\s*(?:-|through|to)\s*([1-9]\d{4})\b',
    re.IGNORECASE
)

# Common NCCI modifiers (comprehensive list)
MODIFIER_RE = re.compile(
    r'\b(modifier\s+)?('
    r'24|25|26|27|50|51|52|53|54|55|56|57|58|59|62|63|66|73|74|76|77|78|79|80|81|82|'
    r'90|91|92|95|96|97|99|'
    r'E[1-4]|'  # Eyelid modifiers E1-E4
    r'F[1-9A]|FA|'  # Finger modifiers F1-F9, FA
    r'T[1-9A]|TA|'  # Toe modifiers T1-T9, TA
    r'LC|LD|LT|RC|RT|'  # Anatomic modifiers
    r'XE|XP|XS|XU|'  # X-EPSU modifiers (subset of 59)
    r'LM|LN|'
    r'GG|GH|GJ|GK|GL|GN|GO|GP|GQ|GR|GV|GW|GX|GY|GZ|'
    r'Q[0-9]|QS|QX|QY|QZ|'
    r'SG|TC|'
    r'A[1-9]|AA|AD|AE|AF|AG|AH|AI|AJ|AK|AM|AP|AQ|AR|AS|AT|AW|AY|AZ|'
    r'BA|BL|BO|BP|BR|BU|'
    r'CA|CB|CC|CD|CE|CF|CG|CH|CI|CJ|CK|CL|CM|CN|CO|CP|CR|CS|CT|'
    r'GC|GE|GF|'
    r'HA|HB|HC|HD|HE|HF|HG|HH|HI|HJ|HK|HL|HM|HN|HO|HP|HQ|HR|HS|HT|HU|HV|HW|HX|HY|HZ|'
    r'JA|JB|JC|JD|JE|JW|'
    r'K[0-4]|KA|KB|KC|KD|KE|KF|KG|KH|KI|KJ|KK|KL|KM|KN|KO|KP|KQ|KR|KS|KT|KU|KV|KW|KX|KY|KZ|'
    r'M[2S]|MS|'
    r'PA|PB|PC|PD|PE|PI|PL|PM|PN|PO|'
    r'Q[0-9A-Z]|'
    r'SA|SB|SC|SD|SE|SF|SG|SH|SI|SJ|SK|SL|SM|SN|SQ|SS|ST|SU|SV|SW|SY|'
    r'U[1-9A-Z]|'
    r'V[1-9A-Z]|'
    r'VP|'
    r'X[1-5E-Z]'
    r')\b',
    re.IGNORECASE
)


def extract_cpt_codes(text: str) -> List[int]:
    """
    Extract individual CPT codes (5-digit numbers) from text
    
    Returns a deduplicated and sorted list
    Filters out common false matches (such as years)
    """
    matches = CPT_SINGLE_RE.findall(text)
    codes = set()
    
    for match in matches:
        code = int(match)
        # Filter out obvious false positives
        # - Years: 2020-2026, 1990-2050
        # - Common page ranges: < 1000
        # - Valid CPT range: 00100-99607 (but starting from 10000 is safer)
        if 10000 <= code <= 99999:
            # Additional heuristic: avoid years
            if not (1990 <= code <= 2050):
                codes.add(code)
    
    return sorted(codes)


def extract_cpt_ranges(text: str) -> List[Dict[str, int]]:
    """
    Extract CPT code ranges from text
    
    Return format: [{"start": 64400, "end": 64530}, ...]
    """
    matches = CPT_RANGE_RE.findall(text)
    ranges = []
    
    for start_str, end_str in matches:
        start = int(start_str)
        end = int(end_str)
        
        # Validate: start < end, both in valid CPT range
        if 10000 <= start < end <= 99999:
            ranges.append({"start": start, "end": end})
    
    return ranges


def mentions_modifiers(text: str) -> bool:
    """
    Check if text mentions modifiers
    
    Simple boolean check for quick filtering
    """
    return bool(MODIFIER_RE.search(text))


def extract_modifiers(text: str) -> List[str]:
    """
    Extract all modifier codes from text
    
    Returns a deduplicated and sorted list
    """
    matches = MODIFIER_RE.findall(text)
    # matches is list of tuples: (prefix, modifier)
    # We want the actual modifier (second element)
    mods = set()
    for prefix, modifier in matches:
        if modifier:
            mods.add(modifier.upper())
    
    return sorted(mods)


# Additional utility: check if a code is within any range
def code_in_ranges(code: int, ranges: List[Dict[str, int]]) -> bool:
    """
    Check if a given CPT code is within any of the provided ranges
    """
    for r in ranges:
        if r["start"] <= code <= r["end"]:
            return True
    return False


# Pattern for CCMI/Modifier Indicator (0, 1, 9, etc.)
MODIFIER_INDICATOR_RE = re.compile(
    r'(?:modifier\s+indicator|CCMI)[\s:]*([0-9])',
    re.IGNORECASE
)


def extract_modifier_indicators(text: str) -> List[str]:
    """
    Extract modifier indicator values (0, 1, 9)
    """
    matches = MODIFIER_INDICATOR_RE.findall(text)
    return list(set(matches))


# Keyword patterns for topic detection (used in topic_tags_for)
TOPIC_KEYWORDS = {
    "PTP": [r'\bPTP\b', r'procedure[- ]to[- ]procedure', r'column 1', r'column 2'],
    "MUE": [r'\bMUE\b', r'medically unlikely edit'],
    "BYPASS": [r'\bbypass\b', r'override', r'CCMI'],
    "MODIFIER": [r'\bmodifier\b', r'\b(25|59|XE|XP|XS|XU|LT|RT|50)\b'],
    "GLOBAL_SURGERY": [r'global surgery', r'global period', r'post[- ]?operative'],
    "ANATOMIC": [r'\b(LT|RT)\b', r'left|right', r'anatomic', r'bilateral', r'finger|toe'],
    "E&M": [r'\bE&M\b', r'evaluation (and|&) management', r'office visit'],
    "GENERAL_POLICY": [r'general (policy|correct coding)', r'principles?'],
    "DISTINCT": [r'distinct procedural service', r'separate (encounter|session|site)', r'different (anatomic|provider)'],
    "DOCUMENTATION": [r'documentation', r'medical record', r'medical necessity'],
}


def detect_topics(text: str) -> Set[str]:
    """
    Detect topic tags related to the text based on keyword patterns
    """
    text_lower = text.lower()
    topics = set()
    
    for topic, patterns in TOPIC_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                topics.add(topic)
                break
    
    return topics


# Test function
if __name__ == "__main__":
    # Test samples
    test_text = """
    CPT codes 64400-64530 describe nerve blocks.
    Modifier 59 may be appropriate when reporting code 31629 with another procedure.
    The CCMI indicator is 1 for codes 12001-12007.
    Modifiers LT and RT are anatomic modifiers.
    Modifier 25 with E&M service on the same day requires separate documentation.
    """
    
    print("CPT Codes:", extract_cpt_codes(test_text))
    print("CPT Ranges:", extract_cpt_ranges(test_text))
    print("Modifiers:", extract_modifiers(test_text))
    print("Modifier Indicators:", extract_modifier_indicators(test_text))
    print("Topics:", detect_topics(test_text))
    print("Mentions modifiers?", mentions_modifiers(test_text))
