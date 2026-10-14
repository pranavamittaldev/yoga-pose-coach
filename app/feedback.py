"""Turn per-joint deviations into plain-language correction tips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import config
from app.scorer import JointRule, JointSummary


@dataclass(frozen=True)
class Tip:
    """One correction tip for a joint."""

    joint: str
    text: str
    avg_score: float
    avg_signed_deviation: float


def choose_tip_text(rule: JointRule, signed_deviation: float) -> str:
    """Pick ``tip_low`` for a negative deviation (angle too small), ``tip_high`` otherwise."""
    return rule.tip_low if signed_deviation < 0 else rule.tip_high


def select_tips(
    summaries: dict[str, JointSummary],
    rules: Sequence[JointRule],
    min_deviation: float = config.MIN_TIP_DEVIATION_DEGREES,
    max_tips: int = config.MAX_TIPS,
) -> list[Tip]:
    """Return up to ``max_tips`` tips for the worst joints, most severe first.

    A joint is a candidate only if it lost points (average score below 100)
    and its average deviation is at least ``min_deviation`` degrees, so tiny
    errors are ignored. Candidates are ordered by lowest average score, then
    by largest deviation. A tip whose text was already chosen (e.g. the same
    wording for left and right) is skipped so each tip appears once.
    """
    rules_by_name = {r.name: r for r in rules}
    candidates = [
        (name, s)
        for name, s in summaries.items()
        if name in rules_by_name and s.avg_score < 100.0 and abs(s.avg_signed_deviation) >= min_deviation
    ]
    candidates.sort(key=lambda item: (item[1].avg_score, -abs(item[1].avg_signed_deviation)))
    tips: list[Tip] = []
    for name, s in candidates:
        text = choose_tip_text(rules_by_name[name], s.avg_signed_deviation)
        if any(t.text == text for t in tips):
            continue
        tips.append(Tip(joint=name, text=text, avg_score=s.avg_score, avg_signed_deviation=s.avg_signed_deviation))
        if len(tips) == max_tips:
            break
    return tips
