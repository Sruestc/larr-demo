"""Phone-logit to policy-token alignment utilities.

This module assumes frame-level phone logits are already available. The logits
may come from a frame-level phoneme recognizer or a CTC phoneme recognizer; the
module does not require either model to be present. It does not load or run a
CTC/Frame-CE model. The first stage is a monotonic forced-path search over the
supplied phone sequence; the second stage projects frame intervals to
policy-token intervals by temporal overlap.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Optional, Sequence

Span = tuple[int, int]

@dataclass(frozen=True)
class ForcedAlignment:
    frame_phone_positions: tuple[int, ...]
    frame_emit: tuple[bool, ...]
    kept_phone_indices: tuple[int, ...]
    score: float

@dataclass(frozen=True)
class PhoneTokenAlignment:
    forced: ForcedAlignment
    phone_frame_spans: tuple[Optional[Span], ...]
    token_to_phone: tuple[int, ...]
    phone_token_spans: tuple[Optional[Span], ...]
    reliable: bool
    reason: str

def _log_softmax(rows: Sequence[Sequence[float]]) -> list[list[float]]:
    result = []
    for row in rows:
        values = [float(x) for x in row]
        if not values or not all(isfinite(x) for x in values):
            raise ValueError("logits must be a non-empty finite rectangular matrix")
        maximum = max(values)
        denominator = sum(exp(x - maximum) for x in values)
        result.append([x - maximum - log(denominator) for x in values])
    return result

def _allowed_skip(optional: Sequence[bool], n: int) -> tuple[list[list[bool]], int]:
    max_run = 0; run = 0
    for value in optional:
        run = run + 1 if value else 0
        max_run = max(max_run, run)
    max_jump = 1 + max_run
    skip_ok = [[False] * n for _ in range(max_jump + 1)]
    for jump in range(1, max_jump + 1):
        for position in range(jump, n):
            skip_ok[jump][position] = all(optional[index] for index in range(position - jump + 1, position))
    return skip_ok, max_jump

def _leading_optional(optional: Sequence[bool]) -> list[bool]:
    result = [False] * (len(optional) + 1); allowed = True
    for index in range(len(optional)):
        result[index] = allowed; allowed = allowed and bool(optional[index])
    result[len(optional)] = allowed
    return result

def _trailing_optional(optional: Sequence[bool]) -> list[bool]:
    result = [False] * len(optional); allowed = True
    for index in range(len(optional) - 1, -1, -1):
        result[index] = allowed; allowed = allowed and bool(optional[index])
    return result

def _best(values: Sequence[float]) -> tuple[int, float]:
    index, value = max(enumerate(values), key=lambda item: item[1])
    return index, value

def constrained_phone_alignment(
    logits: Sequence[Sequence[float]],
    phone_ids: Sequence[int],
    *,
    blank_index: int = 0,
    optional: Optional[Sequence[bool]] = None,
    kept_phone_indices: Optional[Sequence[int]] = None,
) -> Optional[ForcedAlignment]:
    """Find the best monotonic path for `phone_ids` under frame logits.

    The path has two states per phone position: `new` (the frame is a blank)
    and `emit` (the frame emits the current phone). A phone may repeat across
    frames, and an advance may skip only positions marked `optional`. This is
    the position-aware forced alignment used by Dense LARR; unlike a free CTC
    decode, it returns the phone position responsible for every frame.

    `kept_phone_indices` maps the supplied sequence back to an original phone
    sequence when the caller removed out-of-vocabulary entries. If omitted,
    positions are returned unchanged.
    """
    if len(logits) == 0 or len(phone_ids) == 0:
        return None
    if any(len(row) != len(logits[0]) for row in logits):
        raise ValueError("logits must be rectangular")
    if blank_index < 0 or blank_index >= len(logits[0]):
        raise ValueError("blank_index is outside the vocabulary")
    if any(int(phone) < 0 or int(phone) >= len(logits[0]) for phone in phone_ids):
        raise ValueError("phone_ids contain an index outside the logits vocabulary")
    optional = list(optional) if optional is not None else [False] * len(phone_ids)
    if len(optional) != len(phone_ids):
        raise ValueError("optional must match phone_ids")
    kept = list(kept_phone_indices) if kept_phone_indices is not None else list(range(len(phone_ids)))
    if len(kept) != len(phone_ids):
        raise ValueError("kept_phone_indices must match phone_ids")

    log_probs = _log_softmax(logits)
    T, n = len(logits), len(phone_ids)
    phone_scores = [[log_probs[t][int(phone_ids[p])] for p in range(n)] for t in range(T)]
    blank_scores = [log_probs[t][blank_index] for t in range(T)]
    skip_ok, max_jump = _allowed_skip(optional, n)
    leading = _leading_optional(optional)
    trailing = _trailing_optional(optional)
    neg_inf = float("-inf")

    dp_new = [neg_inf] * n
    dp_emit = [neg_inf] * n
    back_new = [[0] * n for _ in range(T)]
    back_emit = [[0] * n for _ in range(T)]
    for p in range(n):
        if leading[p]:
            dp_emit[p] = phone_scores[0][p]
            if optional[p]:
                dp_new[p] = blank_scores[0]
    dp_new[0] = blank_scores[0]

    for t in range(1, T):
        blank_candidates = [max(dp_new[p] + blank_scores[t], dp_emit[p] + blank_scores[t]) for p in range(n)]
        for p in range(n):
            back_new[t][p] = 1 if dp_emit[p] > dp_new[p] else 0
        next_new = blank_candidates
        emit_candidates: list[list[float]] = [
            list(dp_new), list(dp_emit)
        ]
        for jump in range(1, max_jump + 1):
            from_new = [neg_inf] * n; from_emit = [neg_inf] * n
            for p in range(jump, n):
                if skip_ok[jump][p]:
                    from_new[p] = dp_new[p - jump]
                    from_emit[p] = dp_emit[p - jump]
            emit_candidates.extend([from_new, from_emit])
        next_emit = [neg_inf] * n
        for p in range(n):
            choices = [emit_candidates[choice][p] for choice in range(len(emit_candidates))]
            choice, value = _best(choices)
            next_emit[p] = value + phone_scores[t][p]
            back_emit[t][p] = choice
        dp_new, dp_emit = next_new, next_emit

    final_new = [dp_new[p] if trailing[p] else neg_inf for p in range(n)]
    final_emit = [dp_emit[p] if trailing[p] else neg_inf for p in range(n)]
    state_new, score_new = _best(final_new)
    state_emit, score_emit = _best(final_emit)
    if score_emit >= score_new:
        position, state, score = state_emit, 1, score_emit
    else:
        position, state, score = state_new, 0, score_new
    if not isfinite(score):
        return None

    frame_positions = [0] * T; frame_emit = [False] * T
    for t in range(T - 1, -1, -1):
        frame_positions[t] = position; frame_emit[t] = state == 1
        if t == 0: break
        if state == 0:
            state = back_new[t][position]
        else:
            choice = back_emit[t][position]
            if choice <= 1:
                state = choice
            else:
                jump = (choice - 2) // 2 + 1
                state = (choice - 2) % 2
                position = max(position - jump, 0)
    return ForcedAlignment(tuple(frame_positions), tuple(frame_emit), tuple(kept), float(score))

def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))

def project_frames_to_policy_tokens(
    forced: ForcedAlignment,
    *,
    phone_count: int,
    policy_token_count: int,
    frame_hop_seconds: float,
    policy_token_rate: float,
    policy_start_seconds: float = 0.0,
) -> tuple[tuple[Optional[Span], ...], tuple[int, ...], tuple[Optional[Span], ...]]:
    """Project forced frame positions to policy tokens by overlap duration.

    Each policy token receives the phone position with the largest temporal
    overlap. Ties are resolved toward the earlier phone. A token with no frame
    overlap is `-1`; returned phone spans are half-open token intervals.
    """
    if phone_count <= 0 or policy_token_count < 0 or frame_hop_seconds <= 0 or policy_token_rate <= 0:
        raise ValueError("invalid timeline or token-count argument")
    if len(forced.frame_phone_positions) != len(forced.frame_emit):
        raise ValueError("forced alignment arrays must have equal length")
    frame_spans: list[Optional[Span]] = [None] * phone_count
    for phone in range(phone_count):
        indices = [i for i, value in enumerate(forced.frame_phone_positions) if value == phone]
        if indices: frame_spans[phone] = (min(indices), max(indices) + 1)
    token_to_phone = [-1] * policy_token_count
    for token in range(policy_token_count):
        t0 = policy_start_seconds + token / policy_token_rate
        t1 = policy_start_seconds + (token + 1) / policy_token_rate
        overlaps = [0.0] * phone_count
        for frame, phone in enumerate(forced.frame_phone_positions):
            if 0 <= phone < phone_count:
                f0, f1 = frame * frame_hop_seconds, (frame + 1) * frame_hop_seconds
                overlaps[phone] += _interval_overlap(t0, t1, f0, f1)
        best = max(range(phone_count), key=lambda phone: overlaps[phone], default=-1)
        if best >= 0 and overlaps[best] > 0.0: token_to_phone[token] = best
    token_spans: list[Optional[Span]] = [None] * phone_count
    for phone in range(phone_count):
        indices = [t for t, value in enumerate(token_to_phone) if value == phone]
        if indices: token_spans[phone] = (min(indices), max(indices) + 1)
    return tuple(frame_spans), tuple(token_to_phone), tuple(token_spans)

def align_phone_logits_to_policy_tokens(
    logits: Sequence[Sequence[float]],
    phone_ids: Sequence[int],
    *,
    policy_token_count: int,
    frame_hop_seconds: float,
    policy_token_rate: float,
    blank_index: int = 0,
    optional: Optional[Sequence[bool]] = None,
    kept_phone_indices: Optional[Sequence[int]] = None,
    policy_start_seconds: float = 0.0,
) -> Optional[PhoneTokenAlignment]:
    """Run constrained phone alignment and temporal phone-to-token projection."""
    forced = constrained_phone_alignment(logits, phone_ids, blank_index=blank_index, optional=optional, kept_phone_indices=kept_phone_indices)
    if forced is None: return None
    frame_spans, token_to_phone, token_spans = project_frames_to_policy_tokens(forced, phone_count=len(phone_ids), policy_token_count=policy_token_count, frame_hop_seconds=frame_hop_seconds, policy_token_rate=policy_token_rate, policy_start_seconds=policy_start_seconds)
    optional_flags = list(optional) if optional is not None else [False] * len(phone_ids)
    required_spans = [token_spans[i] for i, is_optional in enumerate(optional_flags) if not is_optional]
    reliable = all(span is not None for span in required_spans) and any(value >= 0 for value in token_to_phone)
    reason = "ok" if reliable else "no_phone_token_overlap"
    return PhoneTokenAlignment(forced, frame_spans, token_to_phone, token_spans, reliable, reason)
