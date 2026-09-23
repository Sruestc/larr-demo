# LARR paper demo

This is a self-contained static demo for the ICASSP 2027 paper:

> LARR: Language-Model-Attenuated Recognition Reward and Localized Credit Assignment for TTS Reinforcement Learning

The page contains four paired ASR-repair examples, six Zero-shot TTS examples across six systems, six SFT TTS examples across three systems, and a compact Dense LARR phone-alignment overview. The WAV examples are bundled under `assets/`, so the page has no lab-server or runtime-service dependency.

Open `index.html` locally, or publish the `demo/` directory with GitHub Pages. The footer links to `https://github.com/Sruestc/larr-demo`.

## GitHub Pages

The repository includes `.github/workflows/pages.yml`. Push the repository to GitHub, then choose **Settings → Pages → Source: GitHub Actions**. Every push to `master` publishes the `demo/` directory and produces a reviewer-facing `github.io` URL.
