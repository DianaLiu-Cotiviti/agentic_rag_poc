"""
Search Guidance Templates for Different Retrieval Strategies

This module contains guidance templates that enhance retrieval quality by providing
search hints, target document types, key terms, and other metadata to guide the
semantic search and hybrid search processes.

These templates can be easily modified to improve retrieval performance.
"""

from typing import Dict, List, Optional


# ============================================================================
# SEMANTIC SEARCH GUIDANCE TEMPLATES
# ============================================================================

def get_ncci_semantic_guidance(cpt_codes: List[str]) -> str:
    """
    Semantic search guidance for NCCI-related queries
    
    Args:
        cpt_codes: List of CPT codes to search for
        
    Returns:
        Detailed guidance text to enhance semantic search
    """
    cpt_str = ", ".join(cpt_codes) if cpt_codes else "the specified CPT code"
    
    return (
        f"Find all relevant NCCI documentation for CPT code {cpt_str}, including:\n"
        f"1. PTP (Procedure-to-Procedure) edits and bundling rules for {cpt_str}\n"
        f"2. Modifier usage policies (59, XE, XP, XS, XU, anatomic modifiers LT/RT)\n"
        f"3. CCMI (Column 1/Column 2 Correct Coding Modifier Indicator) and bypass indicators\n"
        f"4. MUE (Medically Unlikely Edits) policies if applicable\n"
        f"5. General policies from Chapter I that apply to this procedure\n"
        f"6. Global surgery packages, anatomic considerations, and distinct procedural services\n"
        f"7. Add-on codes, bilateral procedures, and separate encounter documentation requirements"
    )


def get_cpt_definition_semantic_guidance(cpt_codes: List[str]) -> str:
    """
    Semantic search guidance for CPT definition lookups
    
    Args:
        cpt_codes: List of CPT codes to look up
        
    Returns:
        Guidance text for CPT definition retrieval
    """
    cpt_str = ", ".join(cpt_codes) if cpt_codes else "the specified CPT code"
    
    return (
        f"Find comprehensive CPT code documentation for {cpt_str}, including:\n"
        f"1. Official CPT code description and definition\n"
        f"2. Anatomical location and surgical approach\n"
        f"3. Procedure scope and typical components included\n"
        f"4. Clinical indications and typical use cases\n"
        f"5. Related codes in the same family or section\n"
        f"6. Any parenthetical notes or special instructions\n"
        f"7. Code-specific guidelines from the CPT manual"
    )


def get_modifier_semantic_guidance(modifiers: List[str], cpt_codes: Optional[List[str]] = None) -> str:
    """
    Semantic search guidance for modifier-related queries
    
    Args:
        modifiers: List of modifier numbers (e.g., ["59", "25"])
        cpt_codes: Optional list of CPT codes for context
        
    Returns:
        Guidance text for modifier retrieval
    """
    mod_str = ", ".join(modifiers) if modifiers else "the specified modifier"
    cpt_context = f" with CPT {', '.join(cpt_codes)}" if cpt_codes else ""
    
    return (
        f"Find comprehensive modifier documentation for modifier {mod_str}{cpt_context}, including:\n"
        f"1. Official modifier definition and purpose\n"
        f"2. When and how to apply modifier {mod_str}\n"
        f"3. Modifier usage rules and restrictions\n"
        f"4. Documentation requirements for proper modifier use\n"
        f"5. Common scenarios and examples of correct application\n"
        f"6. Payer-specific policies or coverage considerations\n"
        f"7. Relationship to other modifiers (substitutions, combinations)"
    )


def get_billing_policy_semantic_guidance(topic: str) -> str:
    """
    Semantic search guidance for billing policy queries
    
    Args:
        topic: The billing policy topic (e.g., "global surgery", "bundling")
        
    Returns:
        Guidance text for billing policy retrieval
    """
    return (
        f"Find comprehensive billing policy documentation for {topic}, including:\n"
        f"1. Official policy statements and guidelines\n"
        f"2. Applicable CPT code ranges or specific codes\n"
        f"3. Coverage criteria and medical necessity requirements\n"
        f"4. Documentation requirements and claim submission guidelines\n"
        f"5. Common billing errors and how to avoid them\n"
        f"6. Examples of correct and incorrect billing scenarios\n"
        f"7. Updates or changes to the policy in recent years"
    )


# ============================================================================
# HYBRID SEARCH GUIDANCE TEMPLATES
# ============================================================================

def get_ncci_hybrid_guidance(cpt_codes: List[str]) -> Dict[str, any]:
    """
    Hybrid search guidance for NCCI-related queries
    Combines semantic guidance with keyword boosting
    
    Args:
        cpt_codes: List of CPT codes to search for
        
    Returns:
        Dictionary with semantic_guidance, boost_terms, and filters
    """
    cpt_str = ", ".join(cpt_codes) if cpt_codes else "the specified CPT code"
    
    return {
        "semantic_guidance": (
            f"Find all relevant NCCI documentation for CPT code {cpt_str}, including:\n"
            f"1. PTP (Procedure-to-Procedure) edits and bundling rules for {cpt_str}\n"
            f"2. Modifier usage policies (59, XE, XP, XS, XU, anatomic modifiers LT/RT)\n"
            f"3. CCMI (Column 1/Column 2 Correct Coding Modifier Indicator) and bypass indicators\n"
            f"4. MUE (Medically Unlikely Edits) policies if applicable\n"
            f"5. General policies from Chapter I that apply to this procedure\n"
            f"6. Global surgery packages, anatomic considerations, and distinct procedural services\n"
            f"7. Add-on codes, bilateral procedures, and separate encounter documentation requirements"
        ),
        "boost_terms": cpt_codes + ["NCCI", "PTP", "bundling", "modifier", "edits"],
        "metadata_filters": {
            "doc_type": ["ncci_policy", "ncci_edits", "billing_guide"]
        }
    }


def get_cpt_definition_hybrid_guidance(cpt_codes: List[str]) -> Dict[str, any]:
    """
    Hybrid search guidance for CPT definition lookups
    
    Args:
        cpt_codes: List of CPT codes to look up
        
    Returns:
        Dictionary with semantic_guidance, boost_terms, and filters
    """
    cpt_str = ", ".join(cpt_codes) if cpt_codes else "the specified CPT code"
    
    return {
        "semantic_guidance": (
            f"Find comprehensive CPT code documentation for {cpt_str}, including:\n"
            f"1. Official CPT code description and definition\n"
            f"2. Anatomical location and surgical approach\n"
            f"3. Procedure scope and typical components included\n"
            f"4. Clinical indications and typical use cases\n"
            f"5. Related codes in the same family or section\n"
            f"6. Any parenthetical notes or special instructions\n"
            f"7. Code-specific guidelines from the CPT manual"
        ),
        "boost_terms": cpt_codes + ["CPT", "procedure", "description", "definition"],
        "metadata_filters": {
            "doc_type": ["cpt_manual", "procedure_description"]
        }
    }


def get_modifier_hybrid_guidance(modifiers: List[str], cpt_codes: Optional[List[str]] = None) -> Dict[str, any]:
    """
    Hybrid search guidance for modifier-related queries
    
    Args:
        modifiers: List of modifier numbers
        cpt_codes: Optional list of CPT codes for context
        
    Returns:
        Dictionary with semantic_guidance, boost_terms, and filters
    """
    mod_str = ", ".join(modifiers) if modifiers else "the specified modifier"
    cpt_context = f" with CPT {', '.join(cpt_codes)}" if cpt_codes else ""
    
    boost_terms = [f"modifier {m}" for m in modifiers] + modifiers
    if cpt_codes:
        boost_terms.extend(cpt_codes)
    
    return {
        "semantic_guidance": (
            f"Find comprehensive modifier documentation for modifier {mod_str}{cpt_context}, including:\n"
            f"1. Official modifier definition and purpose\n"
            f"2. When and how to apply modifier {mod_str}\n"
            f"3. Modifier usage rules and restrictions\n"
            f"4. Documentation requirements for proper modifier use\n"
            f"5. Common scenarios and examples of correct application\n"
            f"6. Payer-specific policies or coverage considerations\n"
            f"7. Relationship to other modifiers (substitutions, combinations)"
        ),
        "boost_terms": boost_terms,
        "metadata_filters": {
            "doc_type": ["modifier_guide", "billing_policy", "ncci_policy"]
        }
    }


# ============================================================================
# GUIDANCE SELECTOR - Automatically choose appropriate guidance
# ============================================================================

def select_guidance_for_query(
    query_type: str,
    question_type: str,
    cpt_codes: List[str],
    modifiers: List[str],
    retrieval_method: str = "semantic"
) -> any:
    """
    Automatically select appropriate guidance based on query characteristics
    
    Args:
        query_type: Type of query candidate (original, synonym, section_specific, etc.)
        question_type: Question type from orchestrator (PTP, modifier, definition, etc.)
        cpt_codes: Extracted CPT codes
        modifiers: Extracted modifiers
        retrieval_method: "semantic" or "hybrid"
        
    Returns:
        Guidance string (for semantic) or dict (for hybrid)
    """
    # For hybrid search, return structured guidance
    if retrieval_method == "hybrid":
        if question_type in ["PTP", "guideline"] or "bundling" in query_type.lower():
            return get_ncci_hybrid_guidance(cpt_codes)
        elif question_type == "modifier" or modifiers:
            return get_modifier_hybrid_guidance(modifiers, cpt_codes)
        elif question_type == "definition" or query_type == "original":
            return get_cpt_definition_hybrid_guidance(cpt_codes)
        else:
            # Default: NCCI guidance for general queries
            return get_ncci_hybrid_guidance(cpt_codes)
    
    # For semantic search, return guidance string
    else:
        if question_type in ["PTP", "guideline"] or "bundling" in query_type.lower():
            return get_ncci_semantic_guidance(cpt_codes)
        elif question_type == "modifier" or modifiers:
            return get_modifier_semantic_guidance(modifiers, cpt_codes)
        elif question_type == "definition" or query_type == "original":
            return get_cpt_definition_semantic_guidance(cpt_codes)
        else:
            # Default: NCCI guidance for general queries
            return get_ncci_semantic_guidance(cpt_codes)


# ============================================================================
# CUSTOM GUIDANCE BUILDER
# ============================================================================

def build_custom_guidance(
    target_info: str,
    aspects: List[str],
    retrieval_method: str = "semantic"
) -> any:
    """
    Build custom search guidance for specialized queries
    
    Args:
        target_info: What we're searching for (e.g., "CPT 14301 documentation")
        aspects: List of specific aspects to find (numbered list items)
        retrieval_method: "semantic" or "hybrid"
        
    Returns:
        Guidance string or dict depending on retrieval_method
    """
    aspects_text = "\n".join([f"{i+1}. {aspect}" for i, aspect in enumerate(aspects)])
    
    semantic_guidance = f"Find comprehensive information about {target_info}, including:\n{aspects_text}"
    
    if retrieval_method == "hybrid":
        return {
            "semantic_guidance": semantic_guidance,
            "boost_terms": [],
            "metadata_filters": {}
        }
    else:
        return semantic_guidance
