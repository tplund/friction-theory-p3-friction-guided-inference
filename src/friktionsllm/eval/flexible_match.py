"""Flexible answer matching for SimpleQA evaluation.

Handles date formats, name ordering, punctuation differences, and numeric
answers while remaining conservative to avoid false positives.

Uses only regex for date parsing -- no external libraries.
"""

from __future__ import annotations

import re

# ── Month lookup ──────────────────────────────────────────────────────────────

_MONTH_MAP: dict[str, int] = {}
_MONTH_NAMES_LONG = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_NAMES_SHORT = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
for _i, (_long, _short) in enumerate(zip(_MONTH_NAMES_LONG, _MONTH_NAMES_SHORT), 1):
    _MONTH_MAP[_long] = _i
    _MONTH_MAP[_short] = _i

_MONTH_PAT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)


# ── Date extraction (regex only) ─────────────────────────────────────────────

def _month_to_int(name: str) -> int | None:
    return _MONTH_MAP.get(name.lower().rstrip("."))


def _extract_dates(text: str) -> list[tuple[int, int, int]]:
    """Return all (year, month, day) tuples found in *text*.

    Supported formats:
      7 June 2018 / 7 Jun 2018 / June 7, 2018 / Jun 7, 2018 / June 7 2018
      2018-06-07 / 07/06/2018 (ambiguous -- try both DMY and MDY)
    """
    results: list[tuple[int, int, int]] = []

    # Remove ordinal suffixes: "7th", "1st", "2nd", "3rd"
    cleaned = re.sub(r"(\d+)(?:st|nd|rd|th)\b", r"\1", text)

    # Pattern 1: "7 June 2018" / "7 Jun 2018" / "7 June, 2018"
    for m in re.finditer(
        r"\b(\d{1,2})\s+" + _MONTH_PAT + r",?\s+(\d{4})\b",
        cleaned, re.IGNORECASE,
    ):
        day = int(m.group(1))
        month_name = m.group(0).split()[1].rstrip(",")
        year = int(m.group(2))
        month = _month_to_int(month_name)
        if month and 1 <= day <= 31:
            results.append((year, month, day))

    # Pattern 2: "June 7, 2018" / "Jun 7, 2018" / "June 7 2018"
    for m in re.finditer(
        r"\b(" + _MONTH_PAT + r")\s+(\d{1,2}),?\s+(\d{4})\b",
        cleaned, re.IGNORECASE,
    ):
        month = _month_to_int(m.group(1))
        day = int(m.group(2))
        year = int(m.group(3))
        if month and 1 <= day <= 31:
            results.append((year, month, day))

    # Pattern 3: "2018-06-07"
    for m in re.finditer(r"\b(\d{4})-(\d{2})-(\d{2})\b", cleaned):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            results.append((year, month, day))

    # Pattern 4: "07/06/2018" -- ambiguous.
    # We try BOTH d/m/y and m/d/y and keep all valid ones so that the
    # caller can match against any interpretation.
    for m in re.finditer(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", cleaned):
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # DMY
        if 1 <= b <= 12 and 1 <= a <= 31:
            results.append((year, b, a))
        # MDY (only add if different from DMY)
        if 1 <= a <= 12 and 1 <= b <= 31 and (a != b):
            results.append((year, a, b))

    return results


# ── Text normalization ────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[.,;:!?\"'()\[\]{}\-/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _strip_all_punct(s: str) -> str:
    """Remove every non-alphanumeric character except space."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", s).lower().strip()


# ── Significant-word set matching ─────────────────────────────────────────────

def _significant_words(s: str) -> set[str]:
    """Words longer than 2 characters, lowercased."""
    return {w for w in _normalize(s).split() if len(w) > 2}


# ── Episode / season matching ────────────────────────────────────────────────

def _extract_episode(s: str) -> tuple[int, int] | None:
    s_lower = s.lower()
    season = re.search(r"season\s*(\d+)", s_lower)
    episode = re.search(r"episode\s*(\d+)", s_lower)
    if season and episode:
        return (int(season.group(1)), int(episode.group(1)))
    se = re.search(r"s(\d+)\s*e(\d+)", s_lower)
    if se:
        return (int(se.group(1)), int(se.group(2)))
    return None


# ── Number matching ──────────────────────────────────────────────────────────

def _extract_core_number(s: str) -> str | None:
    """If *s* is essentially a number (possibly with trailing units/period),
    return the number as a canonical string.  '36.' -> '36', '36 units' -> '36'.
    Handles comma-separated thousands: '26,364.02' -> '26364.02'.

    Does NOT match if the string contains slashes, colons, or other
    structure indicating it is NOT a simple number (e.g. DOIs, times)."""
    s_stripped = s.strip()
    # Reject if string looks like a DOI, URL path, time, or structured ID
    if re.search(r"[/:\\@#]", s_stripped):
        return None
    # First remove commas used as thousand separators (digits on both sides)
    cleaned = re.sub(r"(\d),(\d)", r"\1\2", s_stripped)
    # Match: number optionally followed by ONE short word (unit)
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z]{0,10})\.?\s*$", cleaned)
    if m:
        num = m.group(1).rstrip(".")
        return num
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def flexible_match(response: str, expected: str) -> bool:
    """Return True if *response* contains the answer given by *expected*.

    Matching strategies, applied in order (first hit wins):
      1. Exact substring (case-insensitive, current behaviour).
      2. Normalised match (strip punctuation, lowercase, collapse whitespace).
      3. Date parsing -- regex only, handles multiple formats. IMPORTANT:
         "11 June 1949" must NOT match "30 June 1949".
      4. Episode/season match.
      5. Number core match (e.g. "36." vs "36 units").
      6. Word-set match -- all significant words (>2 chars) from *expected*
         appear in *response*, order-independent. Only for short expected
         answers (<=5 words) to avoid false positives on long phrases.
      7. Repeat 1-6 on the *last line* of response (models often put the
         concise answer on the final line).
    """
    if not response or not expected:
        return False

    # Also try extracting an answer after common prefixes
    # in the full response, plus just the last line.
    response = response.strip()
    expected = expected.strip()

    if _match_core(response, expected):
        return True

    # Try last line separately
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    if len(lines) > 1:
        last_line = lines[-1]
        if _match_core(last_line, expected):
            return True

    # Try extracting answer after common prefixes
    for prefix in ("the answer is", "answer:", "result:", "born on",
                    "date is", "it is", "it was", "was on"):
        idx = response.lower().find(prefix)
        if idx >= 0:
            segment = response[idx + len(prefix):idx + len(prefix) + 120].strip()
            if _match_core(segment, expected):
                return True

    return False


def _match_core(response: str, expected: str) -> bool:
    """Run the ordered matching strategies on response vs expected."""

    resp_lower = response.lower()
    exp_lower = expected.lower()

    # 1. Exact substring (case-insensitive) -- current behaviour
    exp_stripped = expected.strip().lower().rstrip(".,;:!?")
    if exp_stripped in resp_lower:
        return True

    # 1b. Reverse substring: response is contained in expected (original
    #     check_match behaviour). E.g. response="15th" in expected="15th place".
    resp_stripped = response.strip().lower().rstrip(".,;:!?")
    if len(resp_stripped) >= 3 and resp_stripped in exp_lower:
        return True

    # 2. Normalised match (both directions)
    norm_exp = _normalize(expected)
    norm_resp = _normalize(response)
    if norm_exp and norm_exp in norm_resp:
        return True
    if norm_resp and len(norm_resp) >= 3 and norm_resp in norm_exp:
        return True

    # 3. Date matching (regex only)
    exp_dates = _extract_dates(expected)
    if exp_dates:
        resp_dates = _extract_dates(response)
        if resp_dates:
            for ed in exp_dates:
                for rd in resp_dates:
                    if ed == rd:
                        return True
        # Dates found in expected but no match in response -- do NOT fall
        # through to word-set matching (would falsely match different dates
        # that share month/year).
        return False

    # 4. Episode/season matching
    ep_exp = _extract_episode(expected)
    if ep_exp:
        ep_resp = _extract_episode(response)
        if ep_resp and ep_exp == ep_resp:
            return True
        # If expected has episode info, do NOT fall through to word-set
        # matching (would match wrong episode numbers since numbers are
        # filtered as "too short").
        return False

    # 5. Number core match  ("36." vs "36 units")
    exp_num = _extract_core_number(expected)
    if exp_num is not None:
        resp_num = _extract_core_number(response)
        if resp_num is not None and exp_num == resp_num:
            return True
        # Also check if the number appears as a standalone token in response
        if re.search(r"\b" + re.escape(exp_num) + r"\b", response):
            return True

    # 6. Word-set match (order-independent) -- only for short expected
    #    Skip if expected is primarily numeric (would match wrong numbers).
    has_digits = bool(re.search(r"\d", expected))
    exp_words = _significant_words(expected)
    if exp_words and len(exp_words) <= 5 and not has_digits:
        resp_words = _significant_words(response)
        if exp_words.issubset(resp_words):
            # Extra guard: response should not be excessively long relative
            # to expected (avoids matching a single word buried in an essay).
            if len(resp_words) <= len(exp_words) * 4:
                return True

    # 7. Punctuation-stripped exact match
    stripped_exp = _strip_all_punct(expected)
    stripped_resp = _strip_all_punct(response)
    if stripped_exp and stripped_exp == stripped_resp:
        return True
    # Substring only when response is not much longer
    if stripped_exp and stripped_exp in stripped_resp:
        if len(stripped_resp) < len(stripped_exp) * 3:
            return True

    return False
