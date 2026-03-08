---
title: DINO-ENDO Phase Recognition
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# DINO-ENDO Streamlit Space

This folder is an isolated Hugging Face Space scaffold for the phase-recognition models in this repository.
It is intentionally separate from the existing FastAPI webapp and defaults to a **DINO-Endo demo** on paid GPU hardware such as **1x A10G (24 GB VRAM)**.
The same code can still expose AI-Endo and V-JEPA2 when you opt into them through environment variables.

## Supported model families

- **AI-Endo**
  - `resnet50.pth`
  - `fusion.pth`
  - `transformer.pth`
- **DINO-Endo**
  - `dinov2_vit14s_latest_checkpoint.pth`
  - `fusion_transformer_decoder_best_model.pth`
  - optional `dinov2_decoder.pth`
  - vendored `dinov2/` source tree
- **V-JEPA2**
  - `vjepa_encoder_human.pt`
  - `mlp_decoder_human.pth`
  - vendored `vjepa2/` source tree

## Weight delivery strategy

The default design is:

1. Keep the **Space repo mostly code-only**.
2. Upload weights to one or more **Hugging Face model repos**.
3. Let the Space populate `model/` (or `SPACE_MODEL_DIR`) on demand via `huggingface_hub`.

This works better than checking all weights directly into the Space repo because code and weights stay versioned separately and Space rebuilds stay lighter.
A fully local `model/` folder is still supported as a fallback.

## Default Space behavior

The Docker Space is configured to boot as a **DINO-Endo-first demo**:

- `SPACE_ENABLED_MODELS=dinov2`
- `SPACE_DEFAULT_MODEL=dinov2`

If you want the same Space build to expose multiple model families again, override those environment variables in Space Settings, for example:

```text
SPACE_ENABLED_MODELS=dinov2,aiendo,vjepa2
SPACE_DEFAULT_MODEL=dinov2
```

The Dockerfile is also set up to be **HF Dev Mode compatible**:

- app code lives under `/app`
- `/app` is owned by uid `1000`
- the required Dev Mode packages (`bash`, `curl`, `wget`, `procps`, `git`, `git-lfs`) are installed

## Runtime configuration

The app looks for model files in `SPACE_MODEL_DIR` first (default: `./model`).
If a required checkpoint is missing locally, it will try to download it from the configured model repo(s).

### Common environment variables

- `SPACE_ENABLED_MODELS` — comma-separated list of model families to expose in the UI
- `SPACE_DEFAULT_MODEL` — default selected model when multiple model families are enabled
- `SPACE_MODEL_DIR` — local directory where checkpoints should live (default: `./model`)
- `PHASE_MODEL_REPO_ID` — shared HF model repo for all weights
- `PHASE_MODEL_REVISION` — optional shared revision/tag/commit
- `HF_TOKEN` — only needed for private or gated repos

If `HF_HOME` / `HF_HUB_CACHE` are not set explicitly, the app will automatically use persistent `/data` storage when it exists and otherwise fall back to a local cache inside the Space folder.

### Per-model overrides

- `AIENDO_MODEL_REPO_ID`, `DINO_MODEL_REPO_ID`, `VJEPA2_MODEL_REPO_ID`
- `AIENDO_MODEL_REVISION`, `DINO_MODEL_REVISION`, `VJEPA2_MODEL_REVISION`
- `AIENDO_MODEL_SUBFOLDER`, `DINO_MODEL_SUBFOLDER`, `VJEPA2_MODEL_SUBFOLDER`

Use subfolder env vars if you store multiple model families in one repo under different directories.

## Local development vs. publishing

The required vendored `dinov2/` and `vjepa2/` source trees are now staged inside this folder, so the Space scaffold is self-contained.
If those upstream source trees change and you want to refresh the copies here, run:

```bash
python scripts/stage_vendor_sources.py --overwrite
```

That script refreshes the vendored source copies inside this folder before publishing.

## Publishing checklist

1. Populate the Space folder files here.
2. Run `python scripts/stage_vendor_sources.py --overwrite` if you need to refresh the vendored source copies.
3. Push the contents of this folder to a Hugging Face **Docker Space**.
4. Upload your checkpoints to HF **model repo(s)**.
5. Configure the relevant repo IDs (and `HF_TOKEN` only if the repos are private).

## Local smoke test

Once the Space dependencies are installed, you can smoke test a predictor directly:

```bash
python scripts/smoke_test.py --model dinov2 --model-dir /path/to/model
python scripts/smoke_test.py --model aiendo --model-dir /path/to/model
python scripts/smoke_test.py --model vjepa2 --model-dir /path/to/model
```

## Scope of v1

- Streamlit UI
- DINO-Endo demo by default, with optional multi-model selector when enabled
- image upload and video upload
- per-frame phase timeline output for video
- JSON / CSV export

Not included in v1:

- auth / user management
- SQL database
- PDF/HTML report generation
- background queue processing
- polyp segmentation
