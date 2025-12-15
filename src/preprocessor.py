import json
import os
import re
from typing import Optional, Dict, Any, List


def _load_course_catalog(catalog_path: str) -> List[dict]:
    """
    Loads course catalog JSON from disk.
    Expected format: a list of objects with:
      department, course_code, course_name, aliases (list)
    """
    if not os.path.exists(catalog_path):
        return []
    with open(catalog_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_course_from_query(
    query: str,
    department: str,
    catalog_path: str = os.path.join("data", "course_catalog.json")
) -> Optional[Dict[str, Any]]:
    """
    Tries to detect a course from the user's query (Hebrew/English/aliases/course code).
    Returns the matched course dict (from catalog) or None.
    """
    if not query:
        return None

    catalog = _load_course_catalog(catalog_path)
    if not catalog:
        return None

    q = query.strip()
    q_lower = q.lower()

    # 1) Try course code pattern like 203.3834 or 2033834
    m = re.search(r"\b(\d{3}\.\d{3,4}|\d{7})\b", q)
    if m:
        code = m.group(1)
        # normalize 2033834 -> 203.3834 (simple heuristic)
        if "." not in code and len(code) == 7:
            code = f"{code[:3]}.{code[3:]}"
        for course in catalog:
            if course.get("department") == department and course.get("course_code") == code:
                return course

    # 2) Try aliases (case-insensitive)
    best_match = None
    best_len = 0

    for course in catalog:
        if course.get("department") != department:
            continue
        aliases = course.get("aliases", []) or []
        # also include course_name as alias automatically
        course_name = course.get("course_name")
        if course_name and course_name not in aliases:
            aliases = aliases + [course_name]

        for alias in aliases:
            if not alias:
                continue
            alias_lower = str(alias).lower().strip()
            if not alias_lower:
                continue

            # substring match (works for Hebrew/English)
            if alias_lower in q_lower:
                if len(alias_lower) > best_len:
                    best_match = course
                    best_len = len(alias_lower)

    return best_match


def augment_query_with_course_context(query: str, course: Optional[Dict[str, Any]]) -> str:
    """
    If we detected a course, we can add a small context hint to the query.
    This is optional but often improves retrieval.
    """
    if not course:
        return query

    course_name = course.get("course_name", "")
    course_code = course.get("course_code", "")

    # Keep it subtle—don't rewrite meaning, just add a hint.
    return f'{query} (Course: {course_name} {course_code})'
