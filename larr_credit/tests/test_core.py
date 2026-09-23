import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from larr_credit import dense_credit, edit_alignment, group_relative_advantages, local_credit, phoneme_reward

class LARRCreditTest(unittest.TestCase):
    def test_reward_and_edit_ops(self):
        result = phoneme_reward(["a", "x", "c"], ["a", "b", "c"])
        self.assertEqual(result.edit_distance, 1)
        self.assertAlmostEqual(result.per, 1 / 3)
        self.assertAlmostEqual(result.reward, -100 / 3)
        self.assertEqual([op.kind for op in result.operations], ["match", "substitution", "match"])

    def test_insertion_marks_adjacent_targets(self):
        ops = edit_alignment(["a", "x", "b"], ["a", "b"])
        insertion = next(op for op in ops if op.kind == "insertion")
        self.assertEqual(insertion.target_indices, (0, 1))

    def test_local_credit_is_unit_mean(self):
        result = local_credit(scalar_advantage=2.0, target_labels=[1, -1], token_to_target=[0, 0, 1, 1], token_mask=[True] * 4, error_weight=3.0)
        self.assertEqual(result.error_token_mask, (False, False, True, True))
        self.assertAlmostEqual(sum(result.token_weights) / 4, 1.0)
        self.assertEqual(result.token_advantage[:2], (0.8, 0.8))
        self.assertEqual(result.token_advantage[2:], (3.2, 3.2))

    def test_dense_group_relative_phone_credit(self):
        result = dense_credit([["a", "b"], ["x", "b"]], [["a", "b"], ["a", "b"]], [[(0, 2), (2, 4)], [(0, 2), (2, 4)]])
        self.assertTrue(all(result.valid))
        self.assertEqual(result.target_labels, ((1, 1), (-1, 1)))
        self.assertAlmostEqual(result.target_advantages[0][0], 1 / math.sqrt(2), places=5)
        self.assertAlmostEqual(result.target_advantages[1][0], -1 / math.sqrt(2), places=5)
        self.assertEqual(result.token_to_target[0], (0, 0, 1, 1))

    def test_dense_group_fallback_on_mismatched_targets(self):
        result = dense_credit([["a"], ["a", "b"]], [["a"], ["a", "b"]], [[(0, 1)], [(0, 1), (1, 2)]], scalar_advantages=[0.5, -0.5])
        self.assertTrue(all(result.fallback))
        self.assertEqual(result.token_advantages, ((0.5,), (-0.5, -0.5)))

    def test_deletion_uses_scalar_fallback(self):
        result = dense_credit([["a"], ["a", "b"]], [["a", "b"], ["a", "b"]], [[(0, 2)], [(0, 2), (2, 4)]], deletion_policy="neighbor")
        self.assertTrue(all(result.fallback))
        self.assertTrue(result.reasons[0].startswith("group_fallback:deletion_requires_scalar"))

    def test_scalar_group_advantage_is_centered(self):
        values = group_relative_advantages([-1.0, -0.5, 0.0])
        self.assertAlmostEqual(sum(values), 0.0, places=6)

if __name__ == "__main__":
    unittest.main()

class AlignmentTest(unittest.TestCase):
    def test_forced_alignment_returns_monotonic_phone_positions(self):
        from larr_credit import constrained_phone_alignment
        logits = [
            [0.0, 5.0, -2.0], [0.0, 5.0, -2.0],
            [5.0, 0.0, -2.0], [0.0, -2.0, 5.0],
            [0.0, -2.0, 5.0], [5.0, -2.0, 0.0],
        ]
        result = constrained_phone_alignment(logits, [1, 2], blank_index=0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.frame_phone_positions), 6)
        self.assertEqual(sorted(set(result.frame_phone_positions)), [0, 1])
        self.assertEqual(list(result.frame_phone_positions), sorted(result.frame_phone_positions))
        self.assertEqual(result.kept_phone_indices, (0, 1))

    def test_projection_uses_temporal_overlap(self):
        from larr_credit import align_phone_logits_to_policy_tokens
        logits = [
            [0.0, 5.0, -2.0], [0.0, 5.0, -2.0],
            [5.0, 0.0, -2.0], [0.0, -2.0, 5.0],
            [0.0, -2.0, 5.0], [5.0, -2.0, 0.0],
        ]
        result = align_phone_logits_to_policy_tokens(logits, [1, 2], policy_token_count=6, frame_hop_seconds=0.04, policy_token_rate=25.0)
        self.assertIsNotNone(result)
        self.assertEqual(result.token_to_phone, (0, 0, 0, 1, 1, 1))
        self.assertEqual(result.phone_token_spans, ((0, 3), (3, 6)))
        self.assertTrue(result.reliable)

    def test_optional_phone_may_have_no_token_span(self):
        from larr_credit import align_phone_logits_to_policy_tokens
        logits = [[0.0, 5.0, -2.0], [0.0, -2.0, 5.0], [5.0, -2.0, 0.0]]
        result = align_phone_logits_to_policy_tokens(logits, [1, 2], optional=[False, True], policy_token_count=2, frame_hop_seconds=0.04, policy_token_rate=25.0)
        self.assertIsNotNone(result)
        self.assertTrue(result.reliable)
