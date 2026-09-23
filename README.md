# LARR Demo

**LARR: Language-Model-Attenuated Recognition Reward and Localized Credit Assignment for TTS Reinforcement Learning**

This repository contains the reviewer-facing static demo for LARR and a small, model-independent reference implementation of the reward and credit-assignment logic.

- Demo page: [sruestc.github.io/larr-demo](https://sruestc.github.io/larr-demo/)
- Demo source: [`demo/index.html`](demo/index.html)
- Reference implementation: [`larr_credit/`](larr_credit/)

The demo includes ASR Repair examples, six-system Zero-shot TTS comparisons, three-system SFT comparisons, and a compact Dense LARR phone-alignment overview. The reference code assumes that phone hypotheses and frame-level phone logits are already available. It implements acoustic phoneme reward, Local/Dense credit assignment, constrained monotonic phone alignment, and phone-to-policy-token temporal projection.

The repository does not include the phone recognizer, acoustic encoder, CTC/Frame-CE model, TTS model, checkpoints, training data, paper sources, or experiment caches. The project license is pending author/institution confirmation.
