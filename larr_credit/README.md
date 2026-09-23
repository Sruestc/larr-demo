# LARR credit assignment (local release candidate)

This directory contains the model-independent part of LARR after a phone recognizer has produced a free phone hypothesis. It implements:

- acoustic phoneme reward and group-relative scalar advantage;
- Local LARR span reweighting; and
- hypothesis-aware Dense LARR phone credit.

The package deliberately does **not** include a phone recognizer, acoustic codec extractor, CTC/Frame-CE aligner, TTS model, checkpoint, training data, or experiment paths. The caller supplies recognized phones and phone-to-policy-token spans. This keeps the release small and makes the algorithm boundary explicit.

## Inputs

For each rollout, provide `hypothesis_phones`, `target_phones`, and `hypothesis_spans`, where each half-open span `(start, end)` covers the policy-token positions aligned to one recognized phone. `token_mask` is optional. Phone sequences should already use the same silence/boundary filtering and tone-equivalence rules as the recognizer.

## Minimal example

```python
from larr_credit import phoneme_reward, group_relative_advantages, local_credit, dense_credit

hypotheses = [["zh", "ang", "san"], ["zh", "an", "san"]]
targets = [["zh", "ang", "san"], ["zh", "ang", "san"]]
spans = [[(0, 2), (2, 4), (4, 6)], [(0, 2), (2, 4), (4, 6)]]
rewards = [phoneme_reward(h, g).reward for h, g in zip(hypotheses, targets)]
scalar = group_relative_advantages(rewards)
local = local_credit(scalar_advantage=scalar[1], target_labels=[1, -1, 1], token_to_target=[0, 0, 1, 1, 2, 2], error_weight=3.0)
dense = dense_credit(hypotheses, targets, spans)
```

## Safety behavior

Dense credit is group-atomic: a singleton group, inconsistent target lengths, invalid spans, or a non-finite alignment triggers scalar fallback when scalar advantages are supplied. This prevents scalar and phone-wise advantages from being mixed within one GRPO group. Insertions mark neighboring target phones. Deletions use group-level scalar fallback by default, matching the conservative Dense LARR implementation; `deletion_policy="scalar"` makes this explicit.

The implementation uses only the Python standard library. Before publication, choose and add the repository license, author information, and citation notice. This directory is the model-independent reference implementation included with the LARR demo repository. The project license is pending author/institution confirmation.

## Phone-to-token alignment from existing logits

The optional alignment module accepts a finite `T x V` matrix of frame-level
phone logits and a phone-ID sequence. The logits can come from either a
frame-level phoneme recognition model or a CTC-based phoneme recognition model.
For a CTC model, `blank_index` identifies the CTC blank class. The alignment
code does not depend on the encoder, acoustic frontend, or model architecture;
it only requires the logits, the phone-vocabulary indices, and the frame/time
rate. It exposes:

```python
from larr_credit import align_phone_logits_to_policy_tokens

alignment = align_phone_logits_to_policy_tokens(
    logits, phone_ids,
    policy_token_count=policy_length,
    frame_hop_seconds=0.04,
    policy_token_rate=25.0,
    blank_index=0,
)
# alignment.token_to_phone[t] is the aligned phone position, or -1.
```

`constrained_phone_alignment` performs a position-aware monotonic dynamic
program. It allows repeated frames for one phone, blank frames, and optional
phone positions. `project_frames_to_policy_tokens` then uses temporal overlap
between frame intervals and policy-token intervals. Ties go to the earlier
phone, and tokens outside the aligned timeline receive `-1`.

For Dense LARR, `phone_ids` should be the recognized free hypothesis, while the
Levenshtein step in `dense_credit` transfers hypothesis-phone positions to
target-phone positions. The code assumes logits are already computed; it does
not include the vocabulary, acoustic encoder, CTC head, or frame-CE model.
