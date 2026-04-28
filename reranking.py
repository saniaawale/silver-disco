"""Deterministic failure analysis and translation re-ranking stubs.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  The translation re-ranking function is a **student assignment**
— see the docstring for inputs, outputs, and implementation guidance.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)


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

    Strategy: rule-based phrase contraction followed by word-boundary truncation
    as a hard fallback. Candidates are deduplicated and sorted shortest first.

    The duration heuristic is 15 chars/second for Spanish TTS.
    """
    char_budget = int(target_duration_s * _CHARS_PER_SECOND)

    if len(baseline_es) <= char_budget:
        return []

    candidates: list[TranslationCandidate] = []
    seen: set[str] = set()

    def _add(text: str, rationale: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(TranslationCandidate(
                text=text,
                char_count=len(text),
                brevity_rationale=rationale,
            ))

    # Pass 1: apply multi-word phrase substitutions in order (longest first so
    # more specific patterns match before their sub-strings do).
    contracted = baseline_es
    applied: list[str] = []
    for phrase, replacement in _PHRASE_CONTRACTIONS:
        import re as _re
        pattern = _re.compile(r'\b' + _re.escape(phrase) + r'\b', _re.IGNORECASE)
        new = pattern.sub(replacement, contracted)
        if new != contracted:
            applied.append(f'"{phrase}"→"{replacement}"')
            contracted = new

    if contracted != baseline_es:
        _add(contracted, "phrase contractions: " + ", ".join(applied))

    # Pass 2: strip standalone filler words from the contracted text.
    import re as _re
    stripped = contracted
    filler_removed: list[str] = []
    for filler in _FILLER_WORDS:
        pattern = _re.compile(
            r'(?<![^\s,;])' + _re.escape(filler) + r'(?![^\s,;.])',
            _re.IGNORECASE,
        )
        new = pattern.sub("", stripped).strip(" ,;")
        # collapse multiple spaces
        new = _re.sub(r' {2,}', ' ', new)
        if new != stripped:
            filler_removed.append(filler)
            stripped = new

    if stripped != contracted:
        _add(stripped, "filler removal: " + ", ".join(filler_removed))

    # Pass 3: word-boundary truncation of the best candidate so far, or the
    # baseline if no contraction helped.  Produces one candidate that fits
    # exactly within the budget by cutting at the last word boundary.
    best_so_far = stripped if stripped != baseline_es else baseline_es
    if len(best_so_far) > char_budget:
        truncated = _truncate_at_word_boundary(best_so_far, char_budget)
        if truncated:
            _add(truncated, f"truncated to {char_budget}-char budget")

    candidates.sort(key=lambda c: c.char_count)
    logger.info(
        "get_shorter_translations: budget=%d chars (%.1fs), baseline=%d chars, "
        "produced %d candidate(s).",
        char_budget,
        target_duration_s,
        len(baseline_es),
        len(candidates),
    )
    return candidates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CHARS_PER_SECOND = 15  # empirical TTS rate for Spanish

# Multi-word phrase → shorter equivalent, ordered longest-match first.
_PHRASE_CONTRACTIONS: list[tuple[str, str]] = [
    ("en este momento", "ahora"),
    ("en este instante", "ahora"),
    ("en estos momentos", "ahora"),
    ("en el momento actual", "actualmente"),
    ("a pesar de todo", "con todo"),
    ("a pesar de ello", "pero"),
    ("a pesar de esto", "pero"),
    ("debido a que", "porque"),
    ("debido a ello", "por eso"),
    ("debido a esto", "por esto"),
    ("a causa de", "por"),
    ("con el fin de", "para"),
    ("con el objetivo de", "para"),
    ("con la finalidad de", "para"),
    ("con el propósito de", "para"),
    ("por medio de", "mediante"),
    ("por parte de", "por"),
    ("por supuesto que", "claro,"),
    ("por supuesto", "claro"),
    ("por lo tanto", "así"),
    ("por lo que", "entonces"),
    ("por lo menos", "al menos"),
    ("sin embargo", "pero"),
    ("no obstante", "pero"),
    ("a continuación", "luego"),
    ("en primer lugar", "primero"),
    ("en segundo lugar", "segundo"),
    ("en último lugar", "por último"),
    ("a continuación", "luego"),
    ("en realidad", "realmente"),
    ("de hecho", "incluso"),
    ("de todas formas", "igual"),
    ("de todas maneras", "igual"),
    ("de todos modos", "igual"),
    ("hay que", "se debe"),
    ("lo que significa", "o sea"),
    ("lo cual significa", "es decir"),
    ("es decir", "o sea"),
    ("es necesario que", "hay que"),
    ("es importante que", "hay que"),
    ("es posible que", "quizás"),
    ("es probable que", "probablemente"),
    ("quiero decir", "o sea"),
    ("me refiero a", "o sea"),
    ("se trata de", "es"),
    ("se puede decir", "podría decirse"),
    ("se podría decir", "podría decirse"),
    ("todo el mundo", "todos"),
    ("mucha gente", "muchos"),
    ("gran cantidad de", "muchos"),
    ("un gran número de", "muchos"),
    ("actualmente", "hoy"),
    ("anteriormente", "antes"),
    ("posteriormente", "después"),
    ("finalmente", "al fin"),
]

# Standalone filler words/phrases that add length without content.
_FILLER_WORDS: list[str] = [
    "pues",
    "bueno",
    "entonces",
    "básicamente",
    "literalmente",
    "evidentemente",
    "obviamente",
    "simplemente",
    "claramente",
    "definitivamente",
    "absolutamente",
    "totalmente",
    "completamente",
    "realmente",
]


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Return *text* cut at the last whitespace that fits within *max_chars*."""
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 0:
        cut = cut[:last_space]
    return cut.rstrip(" ,;:")
