"""
Keyword parsing utilities
Extract specific entities (CPT codes, modifiers, etc.) from keywords list
"""
import re
from typing import List, Tuple


def extract_cpt_codes(keywords: List[str]) -> List[str]:
    """
    Extract CPT codes from keywords list
    
    Args:
        keywords: List of keywords that may contain CPT codes
        
    Returns:
        List of CPT codes (5-digit strings)
        
    Examples:
        >>> extract_cpt_codes(["CPT 14301", "modifier 59", "tissue transfer"])
        ["14301"]
        >>> extract_cpt_codes(["14301", "14302", "adjacent tissue"])
        ["14301", "14302"]
        >>> extract_cpt_codes(["CPT 14000-14350", "range"])
        ["14000", "14350"]
    """
    codes = []
    
    for keyword in keywords:
        # Pattern 1: "CPT 14301" or "CPT14301"
        if match := re.search(r'(?:CPT\s*)?(\d{5})', keyword, re.IGNORECASE):
            codes.append(match.group(1))
        
        # Pattern 2: CPT code ranges "14000-14350"
        elif match := re.search(r'(\d{5})\s*-\s*(\d{5})', keyword):
            codes.extend([match.group(1), match.group(2)])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_codes = []
    for code in codes:
        if code not in seen:
            seen.add(code)
            unique_codes.append(code)
    
    return unique_codes


def extract_modifiers(keywords: List[str]) -> List[str]:
    """
    Extract modifier numbers from keywords list
    
    Args:
        keywords: List of keywords that may contain modifiers
        
    Returns:
        List of modifier codes (2-digit strings)
        
    Examples:
        >>> extract_modifiers(["modifier 59", "CPT 14301"])
        ["59"]
        >>> extract_modifiers(["mod 25", "modifier 51"])
        ["25", "51"]
    """
    modifiers = []
    
    for keyword in keywords:
        # Pattern: "modifier 59" or "mod 59" or "-59"
        if match := re.search(r'(?:modifier|mod)\s*(\d{2})', keyword, re.IGNORECASE):
            modifiers.append(match.group(1))
        elif match := re.search(r'-(\d{2})\b', keyword):
            modifiers.append(match.group(1))
    
    # Remove duplicates
    return list(dict.fromkeys(modifiers))


def extract_code_range(keywords: List[str]) -> Tuple[str, str] | None:
    """
    Extract CPT code range from keywords
    
    Args:
        keywords: List of keywords
        
    Returns:
        Tuple of (start_code, end_code) or None if no range found
        
    Examples:
        >>> extract_code_range(["CPT 14000-14350", "range"])
        ("14000", "14350")
        >>> extract_code_range(["CPT 14301"])
        None
    """
    for keyword in keywords:
        if match := re.search(r'(\d{5})\s*-\s*(\d{5})', keyword):
            return (match.group(1), match.group(2))
    return None


def has_cpt_codes(keywords: List[str]) -> bool:
    """
    Quick check if keywords contain any CPT codes
    
    Args:
        keywords: List of keywords
        
    Returns:
        True if any CPT code found
    """
    return len(extract_cpt_codes(keywords)) > 0


# Example usage
if __name__ == "__main__":
    # Test examples
    keywords1 = ["CPT 14301", "modifier 59", "adjacent tissue transfer"]
    print(f"Keywords: {keywords1}")
    print(f"CPT codes: {extract_cpt_codes(keywords1)}")
    print(f"Modifiers: {extract_modifiers(keywords1)}")
    print()
    
    keywords2 = ["CPT 14000-14350", "range", "modifier 25", "billing guidelines"]
    print(f"Keywords: {keywords2}")
    print(f"CPT codes: {extract_cpt_codes(keywords2)}")
    print(f"Code range: {extract_code_range(keywords2)}")
    print(f"Modifiers: {extract_modifiers(keywords2)}")
    print()
    
    keywords3 = ["adjacent tissue transfer", "complex repair", "separately reportable"]
    print(f"Keywords: {keywords3}")
    print(f"Has CPT codes: {has_cpt_codes(keywords3)}")
    print(f"CPT codes: {extract_cpt_codes(keywords3)}")
