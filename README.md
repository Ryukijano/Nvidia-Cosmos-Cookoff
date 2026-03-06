# Cosmos Sentinel

Cosmos Sentinel is a demo-first traffic safety pipeline that combines three stages:

- **BADAS** for early predictive collision gating
- **Cosmos Reason 2** for incident understanding and severity narration
- **Cosmos Predict 2.5** for optional future-world continuation rollouts

The repository is organized so you can demo the full experience from a single Streamlit app while keeping the core pipeline scripts reusable from the command line.

## What the demo does

1. Runs BADAS on an input traffic video to find the highest-risk moment.
2. Extracts a focused clip around that alert window.
3. Runs Cosmos Reason 2 on the full video plus the focused clip.
4. Surfaces structured outputs, visual diagnostics, and artifacts in Streamlit.
5. Optionally runs Cosmos Predict to generate observed or prevented continuation videos.

## Main entry points

- `demo_streamlit.py` — primary Streamlit demo UI
- `main.py` — convenience launcher for the Streamlit app
- `main_pipeline.py` — CLI orchestration for BADAS + clip extraction + Reason
- `test_full_pipeline.py` — validation script for staged or full pipeline checks

## Repository structure

```text
.
├── demo_streamlit.py
├── main.py
├── main_pipeline.py
├── badas_detector.py
├── cosmos_risk_narrator.py
├── cosmos_predict_runner.py
├── extract_clip.py
├── test_full_pipeline.py
├── .streamlit/config.toml
├── 1_first.mp4
├── nexar_data/
├── cosmos-predict2.5/
├── cosmos-reason2/
└── cosmos-cookbook/
```

## Quickstart

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure access for gated models or datasets

If your BADAS assets or model access require Hugging Face authentication, export a token before running the pipeline:

```bash
export HF_TOKEN=your_token_here
```

You can also start from `.env.example` and load it manually in your shell.

### 3. Make sure required assets are available

This repo expects local assets rather than re-downloading everything automatically during each run.

Typical requirements are:

- BADAS model files under `nexar_data/badas_model/`
- access to the selected Cosmos Reason model checkpoint
- vendored `cosmos-predict2.5/` code if you want Predict rollouts

## Run the app

### Streamlit demo

```bash
python main.py
```

or

```bash
streamlit run demo_streamlit.py
```

### Pipeline only

```bash
python main_pipeline.py 1_first.mp4
```

### Validation

Run the fast staged validation:

```bash
python test_full_pipeline.py 1_first.mp4 --stages-only
```

Run the full validation path:

```bash
python test_full_pipeline.py 1_first.mp4
```

## Outputs

The pipeline produces structured payloads and artifacts such as:

- `BADAS_JSON:` detector payloads
- `REASON_JSON:` narrator payloads
- `PIPELINE_JSON:` end-to-end summary payloads
- extracted focus clips
- saliency and risk visualizations
- frame strips and rollout videos

Generated artifacts are ignored by default in `.gitignore` so the repo stays clean.

## Notes for submission

- The demo title is **Cosmos Sentinel**.
- The sample video path is repo-relative so the project is easier to clone and run elsewhere.
- Hardcoded tokens were removed from the public scripts; use environment variables instead.
- Large model weights and dataset assets are not bundled directly in the repo.

## Known operational requirements

- CUDA is strongly recommended for Cosmos Reason and BADAS performance.
- Cosmos Predict is optional and is triggered manually from the UI.
- Some upstream assets may be gated and require accepted licenses or authenticated access.

## Recommended submission checklist

- Add a project description and demo link in the GitHub repo header.
- Add screenshots or a short GIF of the Streamlit dashboard.
- Initialize git and verify `git status` only shows source files you want to commit.
- Confirm no secrets are present before pushing.
