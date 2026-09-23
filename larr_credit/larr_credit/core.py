"""Pure-Python LARR reward and credit assignment.

The functions in this module start after phone recognition and acoustic/frame
alignment. They intentionally return ordinary Python values so that a caller
can convert the result to PyTorch/JAX tensors inside its own training loop.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Callable, Optional, Sequence

Phone = str
Span = tuple[int, int]
PhoneEqual = Callable[[Phone, Phone], bool]

def _same_phone(a: Phone, b: Phone, phone_equal: Optional[PhoneEqual]) -> bool:
    return phone_equal(a, b) if phone_equal is not None else a == b

@dataclass(frozen=True)
class EditOp:
    kind: str
    hypothesis_index: Optional[int]
    target_index: Optional[int]
    target_indices: tuple[int, ...]

@dataclass(frozen=True)
class RewardResult:
    edit_distance: int
    per: float
    reward: float
    operations: tuple[EditOp, ...]

@dataclass(frozen=True)
class LocalCreditResult:
    token_advantage: tuple[float, ...]
    token_weights: tuple[float, ...]
    error_token_mask: tuple[bool, ...]
    valid: bool
    reason: str

@dataclass(frozen=True)
class DenseCreditResult:
    token_advantages: tuple[tuple[float, ...], ...]
    target_advantages: tuple[tuple[float, ...], ...]
    target_labels: tuple[tuple[int, ...], ...]
    token_to_target: tuple[tuple[int, ...], ...]
    valid: tuple[bool, ...]
    fallback: tuple[bool, ...]
    reasons: tuple[str, ...]

def edit_alignment(hypothesis: Sequence[Phone], target: Sequence[Phone], phone_equal: Optional[PhoneEqual] = None) -> tuple[EditOp, ...]:
    """Return deterministic hypothesis-to-target Levenshtein operations."""
    n, m = len(hypothesis), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if _same_phone(hypothesis[i - 1], target[j - 1], phone_equal):
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and _same_phone(hypothesis[i - 1], target[j - 1], phone_equal) and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(EditOp("match", i - 1, j - 1, (j - 1,))); i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(EditOp("substitution", i - 1, j - 1, (j - 1,))); i, j = i - 1, j - 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(EditOp("deletion", None, j - 1, (j - 1,))); j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            anchors = () if m == 0 else ((0,) if j == 0 else ((m - 1,) if j == m else (j - 1, j)))
            ops.append(EditOp("insertion", i - 1, None, anchors)); i -= 1
        else:
            raise RuntimeError("Levenshtein backtrace did not match its DP table")
    ops.reverse()
    return tuple(ops)

def _labels_from_ops(target_length: int, ops: Sequence[EditOp]) -> list[int]:
    labels = [1] * target_length
    for op in ops:
        if op.kind != "match":
            for index in op.target_indices:
                if 0 <= index < target_length: labels[index] = -1
    return labels

def phoneme_reward(hypothesis: Sequence[Phone], target: Sequence[Phone], *, scale: float = 100.0, phone_equal: Optional[PhoneEqual] = None) -> RewardResult:
    """Compute reward ``-scale * PER`` with unit-cost edits."""
    ops = edit_alignment(hypothesis, target, phone_equal)
    distance = sum(op.kind != "match" for op in ops)
    per = distance / max(len(target), 1)
    return RewardResult(distance, per, -scale * per, ops)

def group_relative_advantages(rewards: Sequence[float], *, eps_reward: float = 1e-6, eps_advantage: float = 1e-6) -> tuple[float, ...]:
    """Normalize a same-prompt reward group and compute sample-std advantages."""
    if not rewards: return ()
    if not all(isfinite(float(x)) for x in rewards): raise ValueError("rewards must be finite")
    denominator = max(abs(float(x)) for x in rewards) + eps_reward
    normalized = [float(x) / denominator for x in rewards]
    mean = sum(normalized) / len(normalized)
    std = sqrt(sum((x - mean) ** 2 for x in normalized) / (len(normalized) - 1)) if len(normalized) > 1 else 0.0
    return tuple((x - mean) / (std + eps_advantage) for x in normalized)

def _validate_spans(spans: Sequence[Span], token_count: int) -> bool:
    previous_end = 0
    for start, end in spans:
        if not (0 <= start < end <= token_count and start >= previous_end): return False
        previous_end = end
    return True

def local_credit(*, scalar_advantage: float, target_labels: Sequence[int], token_to_target: Sequence[int], token_mask: Optional[Sequence[bool]] = None, error_weight: float = 3.0) -> LocalCreditResult:
    """Redistribute a scalar advantage toward error-bearing phone spans."""
    if error_weight < 0 or not isfinite(float(error_weight)): raise ValueError("error_weight must be finite and non-negative")
    n = len(token_to_target); mask = list(token_mask) if token_mask is not None else [True] * n
    if len(mask) != n: raise ValueError("token_mask and token_to_target must have equal length")
    if not isfinite(float(scalar_advantage)): raise ValueError("scalar_advantage must be finite")
    error_mask = [bool(mask[t]) and 0 <= int(k) < len(target_labels) and int(target_labels[k]) < 0 for t, k in enumerate(token_to_target)]
    raw = [1.0 + error_weight if bad else 1.0 for bad in error_mask]
    valid_count = sum(mask)
    if valid_count == 0: return LocalCreditResult(tuple(0.0 for _ in range(n)), tuple(raw), tuple(error_mask), False, "empty_token_mask")
    mean = sum(raw[t] for t in range(n) if mask[t]) / valid_count
    weights = [raw[t] / mean if mask[t] else 0.0 for t in range(n)]
    values = [float(scalar_advantage) * weights[t] if mask[t] else 0.0 for t in range(n)]
    return LocalCreditResult(tuple(values), tuple(weights), tuple(error_mask), True, "ok")

def _target_map_for_row(ops: Sequence[EditOp], spans: Sequence[Span], token_count: int, *, deletion_policy: str) -> tuple[list[int], str]:
    if len(spans) != sum(op.kind != "deletion" for op in ops): return [-1] * token_count, "span_count_mismatch"
    if not _validate_spans(spans, token_count): return [-1] * token_count, "invalid_spans"
    token_to_target = [-1] * token_count; span_by_hypothesis = {}; span_index = 0
    for op in ops:
        if op.hypothesis_index is None: continue
        span = spans[span_index]; span_index += 1; span_by_hypothesis[op.hypothesis_index] = span
        mapped = op.target_index if op.kind in {"match", "substitution"} else (op.target_indices[0] if op.target_indices else -1)
        for token in range(span[0], span[1]): token_to_target[token] = mapped
    if deletion_policy not in {"neighbor", "scalar"}: raise ValueError("deletion_policy must be 'neighbor' or 'scalar'")
    if deletion_policy == "scalar" and any(op.kind == "deletion" for op in ops): return [-1] * token_count, "deletion_requires_scalar"
    if deletion_policy == "neighbor" and any(op.kind == "deletion" for op in ops):
        # Deletions do not have an observed hypothesis span. The reference
        # implementation uses a conservative group-level scalar fallback.
        return [-1] * token_count, "deletion_requires_scalar"
    for op_index, op in enumerate(ops):
        if op.kind != "deletion" or op.target_index is None: continue
        previous_hyp = next((x.hypothesis_index for x in reversed(ops[:op_index]) if x.hypothesis_index in span_by_hypothesis), None)
        next_hyp = next((x.hypothesis_index for x in ops[op_index + 1:] if x.hypothesis_index in span_by_hypothesis), None)
        boundary = span_by_hypothesis[previous_hyp][1] - 1 if previous_hyp is not None else (span_by_hypothesis[next_hyp][0] if next_hyp is not None else None)
        if boundary is None: return [-1] * token_count, "deletion_without_neighbor_span"
        token_to_target[boundary] = op.target_index
    return token_to_target, "ok"

def dense_credit(hypotheses: Sequence[Sequence[Phone]], targets: Sequence[Sequence[Phone]], hypothesis_spans: Sequence[Sequence[Span]], *, token_counts: Optional[Sequence[int]] = None, token_masks: Optional[Sequence[Sequence[bool]]] = None, scalar_advantages: Optional[Sequence[float]] = None, phone_equal: Optional[PhoneEqual] = None, eps: float = 1e-6, deletion_policy: str = "neighbor") -> DenseCreditResult:
    """Compute phone-wise group-relative credit for one same-prompt group."""
    bsz = len(hypotheses)
    if not (len(targets) == len(hypothesis_spans) == bsz): raise ValueError("hypotheses, targets, and spans must align")
    if bsz == 0: return DenseCreditResult((), (), (), (), (), (), ())
    if scalar_advantages is not None and len(scalar_advantages) != bsz: raise ValueError("scalar_advantages must align")
    if token_counts is None: token_counts = [max((end for _, end in spans), default=0) for spans in hypothesis_spans]
    if len(token_counts) != bsz: raise ValueError("token_counts must align")
    if token_masks is not None and len(token_masks) != bsz: raise ValueError("token_masks must align")
    if token_masks is not None:
        for index, mask in enumerate(token_masks):
            if len(mask) != token_counts[index]:
                raise ValueError("each token mask must match its token count")
    def fallback(reason: str, maps: Optional[list[list[int]]] = None) -> DenseCreditResult:
        values = tuple(tuple(float(scalar_advantages[i]) if scalar_advantages is not None and (token_masks is None or token_masks[i][t]) else 0.0 for t in range(token_counts[i])) for i in range(bsz))
        return DenseCreditResult(values, tuple(() for _ in range(bsz)), tuple(() for _ in range(bsz)), tuple(tuple(x) for x in (maps or [[-1] * n for n in token_counts])), tuple(False for _ in range(bsz)), tuple(True for _ in range(bsz)), tuple(reason for _ in range(bsz)))
    if bsz < 2: return fallback("singleton_group")
    if len({len(target) for target in targets}) != 1 or not targets[0]: return fallback("target_length_mismatch")
    ops_rows = [edit_alignment(hypotheses[i], targets[i], phone_equal) for i in range(bsz)]
    maps = []; reasons = []
    for i in range(bsz):
        mapping, reason = _target_map_for_row(ops_rows[i], hypothesis_spans[i], token_counts[i], deletion_policy=deletion_policy)
        maps.append(mapping); reasons.append(reason)
    if any(reason != "ok" for reason in reasons): return fallback("group_fallback:" + "+".join(sorted(set(reasons))), maps)
    labels = [_labels_from_ops(len(targets[i]), ops_rows[i]) for i in range(bsz)]
    target_length = len(targets[0]); target_advantages = [[0.0] * target_length for _ in range(bsz)]
    for position in range(target_length):
        vals = [float(labels[i][position]) for i in range(bsz)]; mean = sum(vals) / bsz
        std = sqrt(sum((v - mean) ** 2 for v in vals) / (bsz - 1))
        for i in range(bsz): target_advantages[i][position] = (vals[i] - mean) / (std + eps)
    token_advantages = []
    for i, mapping in enumerate(maps):
        mask = list(token_masks[i]) if token_masks is not None else [True] * token_counts[i]
        if len(mask) != token_counts[i]: raise ValueError("each token mask must match token count")
        token_advantages.append([target_advantages[i][k] if mask[t] and 0 <= k < target_length else 0.0 for t, k in enumerate(mapping)])
    return DenseCreditResult(tuple(tuple(x) for x in token_advantages), tuple(tuple(x) for x in target_advantages), tuple(tuple(x) for x in labels), tuple(tuple(x) for x in maps), tuple(True for _ in range(bsz)), tuple(False for _ in range(bsz)), tuple("ok" for _ in range(bsz)))
