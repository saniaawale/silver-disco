"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)

_CHARS_PER_SECOND = 15.0

# Common verbose Spanish phrases → shorter equivalents
_PHRASE_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\ben este momento\b", "ahora"),
    (r"\ben este instante\b", "ahora"),
    (r"\bactualmente\b", "hoy"),
    (r"\ba pesar de ello\b", "aun así"),
    (r"\bsin embargo\b", "pero"),
    (r"\bno obstante\b", "pero"),
    (r"\bpor lo tanto\b", "entonces"),
    (r"\bpor consiguiente\b", "así"),
    (r"\bcon el fin de\b", "para"),
    (r"\ba fin de\b", "para"),
    (r"\bes decir\b", "o sea"),
    (r"\bde todas formas\b", "igual"),
    (r"\bde todas maneras\b", "igual"),
    (r"\bpor supuesto\b", "claro"),
    (r"\bdesde luego\b", "claro"),
    (r"\bpor otra parte\b", "además"),
    (r"\bpor otro lado\b", "además"),
    (r"\ben realidad\b", "en verdad"),
    (r"\bmucho más\b", "más"),
    (r"\bmucho menos\b", "menos"),
    (r"\bverdaderamente\b", "muy"),
    (r"\bcompletamente\b", "del todo"),
    (r"\btotalmente\b", "del todo"),
    (r"\bsimplemente\b", "solo"),
    (r"\bsolamente\b", "solo"),
    (r"\bunicamente\b", "solo"),
    (r"\bgeneralmente\b", "en general"),
    (r"\bnormalmente\b", "usualmente"),
    (r"\bhabitualmente\b", "siempre"),
    (r"\bfinalmente\b", "al fin"),
    (r"\bpor último\b", "al fin"),
    (r"\bes importante\b", "importa"),
    (r"\blo que es\b", "lo"),
    (r"\bque es\b", "que"),
]

# Filler/hedge words safe to drop
_FILLERS = [
    r"\bbueno\b,?\s*",
    r"\bpues\b,?\s*",
    r"\bverdad\b,?\s*",
    r"\bclaro\b,?\s*",
    r"\beh\b,?\s*",
    r"\bum\b,?\s*",
    r"\bah\b,?\s*",
    r"\bde hecho\b,?\s*",
    r"\ben fin\b,?\s*",
]


def _apply_phrase_substitutions(text: str) -> str:
    for pattern, replacement in _PHRASE_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _strip_fillers(text: str) -> str:
    for pattern in _FILLERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip()


def _truncate_to_budget(text: str, budget_chars: int) -> str:
    if len(text) <= budget_chars:
        return text
    words = text.split()
    result = []
    length = 0
    for word in words:
        if length + len(word) + (1 if result else 0) > budget_chars:
            break
        result.append(word)
        length += len(word) + (1 if len(result) > 1 else 0)
    return " ".join(result)


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    budget_chars = int(target_duration_s * _CHARS_PER_SECOND)
    candidates: list[TranslationCandidate] = []

    # Level 1: phrase substitution
    substituted = _apply_phrase_substitutions(baseline_es).strip()
    if substituted != baseline_es:
        candidates.append(TranslationCandidate(
            text=substituted,
            char_count=len(substituted),
            brevity_rationale="replaced verbose phrases with shorter equivalents",
        ))

    # Level 2: filler removal (applied on top of substitutions)
    defilled = _strip_fillers(substituted)
    if defilled != substituted and defilled:
        candidates.append(TranslationCandidate(
            text=defilled,
            char_count=len(defilled),
            brevity_rationale="removed filler words",
        ))

    # Level 3: hard truncation to fit budget
    best_so_far = defilled if defilled else substituted if substituted != baseline_es else baseline_es
    truncated = _truncate_to_budget(best_so_far, budget_chars)
    if truncated != best_so_far and truncated:
        candidates.append(TranslationCandidate(
            text=truncated,
            char_count=len(truncated),
            brevity_rationale=f"truncated to fit {budget_chars}-char budget ({target_duration_s:.1f}s)",
        ))

    # Always include the baseline if nothing else fits better or as a fallback
    if not candidates:
        candidates.append(TranslationCandidate(
            text=baseline_es,
            char_count=len(baseline_es),
            brevity_rationale="no shortening possible; returning baseline",
        ))

    return sorted(candidates, key=lambda c: c.char_count)
