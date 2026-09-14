# LARR paper demo

This is a self-contained static demo for the ICASSP 2027 paper:

> LARR: LM-Agnostic Recognition Reward and Hypothesis-Aware Credit Assignment for TTS Reinforcement Learning

The page has two listening sections: four paired ASR-repair examples from the 2,000-utterance diagnostic set, six Zero-shot TTS examples across six systems, and six SFT TTS examples across three systems. The short WAV examples are bundled under `assets/audio/`, so the page does not depend on lab-server paths or external runtime services. The page is fully static and has no runtime service dependency.

Open `index.html` locally, or publish the `demo/` directory with GitHub Pages.

## GitHub Pages

The repository includes `.github/workflows/pages.yml`. Push the repository to GitHub, then choose **Settings → Pages → Source: GitHub Actions**. Every push to `master` publishes the `demo/` directory and produces a reviewer-facing `github.io` URL.
