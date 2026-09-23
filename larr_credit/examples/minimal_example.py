"""Run from this directory with: python examples/minimal_example.py"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from larr_credit import dense_credit, group_relative_advantages, local_credit, phoneme_reward

hypotheses = [["zh", "ang", "san"], ["zh", "an", "san"]]
targets = [["zh", "ang", "san"], ["zh", "ang", "san"]]
spans = [[(0, 2), (2, 4), (4, 6)], [(0, 2), (2, 4), (4, 6)]]
rewards = [phoneme_reward(h, g).reward for h, g in zip(hypotheses, targets)]
scalar = group_relative_advantages(rewards)
local = local_credit(scalar_advantage=scalar[1], target_labels=[1, -1, 1], token_to_target=[0, 0, 1, 1, 2, 2], token_mask=[True] * 6, error_weight=3.0)
dense = dense_credit(hypotheses, targets, spans)
print("rewards:", rewards)
print("scalar advantages:", scalar)
print("local token advantages:", local.token_advantage)
print("dense token advantages:", dense.token_advantages)
