---
title: "Ryukijano's Project Portfolio"
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Ryukijano's Project Portfolio

This folder is an isolated Hugging Face Space scaffold for the phase-recognition models in this repository.
It is intentionally separate from the existing FastAPI webapp and is designed to expose **DINO-Endo, AI-Endo, and V-JEPA2** on paid GPU hardware such as **1x A10G (24 GB VRAM)**.
The public UI now behaves like a small **project portfolio**: DINO-Endo Surgery is the first featured workspace, and the same landing page can later host additional projects without rebuilding the overall shell.
The default featured model remains **DINO-Endo**, but the same Space can load and unload all three model families one at a time.

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

The Docker Space is configured to boot as a **three-model public demo** with **DINO-Endo** selected by default:

- `SPACE_ENABLED_MODELS=dinov2,aiendo,vjepa2`
- `SPACE_DEFAULT_MODEL=dinov2`

If you want to narrow the public picker to a subset of models, override those environment variables in Space Settings, for example:

```text
SPACE_ENABLED_MODELS=dinov2
SPACE_DEFAULT_MODEL=dinov2
```

The Dockerfile is also set up to be **HF Dev Mode compatible**:

- app code lives under `/app`
- `/app` is owned by uid `1000`
- the required Dev Mode packages (`bash`, `curl`, `wget`, `procps`, `git`, `git-lfs`) are installed

## Runtime configuration

The app looks for model files in `SPACE_MODEL_DIR` first. When that env var is unset, the runtime prefers persistent `/data/model`, then falls back to `~/.cache/huggingface/models`, and only uses a local bundled `model/` directory when actual checkpoint files are already present there.
If a required checkpoint is missing locally, it will try to download it from the configured model repo(s).

### Upload and dashboard behavior

- The top of the app is a reusable portfolio landing section, with DINO-Endo Surgery as the current live workspace.
- The active model family is selected through a visible **model slider** in the workspace rather than a hidden picker.
- The Space now keeps a single active predictor loaded at a time and unloads the previous model when the model slider changes.
- MP4 is the primary video upload format, while `mov`, `avi`, `mkv`, `webm`, and `m4v` remain enabled as fallback containers.
- `.streamlit/config.toml` raises the default Streamlit single-file upload ceiling to **4096 MB** and disables file watching / usage telemetry so runtime cache writes do not trigger restart loops.
- Uploaded videos are immediately spooled to local disk for metadata probing and analysis, instead of repeatedly reading the in-memory upload object on every rerun.
- The UI shows file size, duration, fps, frame count, resolution, working-storage headroom, and suppresses inline preview for very large uploads to keep the browser path lighter.
- V-JEPA2 is labeled as a slower first load so users understand the cold-cache cost of its very large encoder checkpoint.

### Explainability behavior

- The sidebar includes an opt-in live explainability toggle for encoder/decoder visualizations.
- DINO-Endo and V-JEPA2 use true encoder self-attention maps, while AI-Endo uses a labeled proxy encoder overlay from ResNet activations.
- AI-Endo and DINO-Endo render decoder-side temporal attention strips from the custom Transformer path.
- V-JEPA2 renders a labeled proxy temporal strip from decoder feature energy because its classifier head is an MLP, not an attention block.
- Encoder controls expose **layer/head sliders** when the loaded model supports true encoder attention.

### Common environment variables

- `SPACE_ENABLED_MODELS` — comma-separated list of model families to expose in the UI
- `SPACE_DEFAULT_MODEL` — default selected model when multiple model families are enabled
- `SPACE_MODEL_DIR` — local directory where checkpoints should live (default: `/data/model` when writable, otherwise `~/.cache/huggingface/models`)
- `PHASE_MODEL_REPO_ID` — shared HF model repo for all weights
- `PHASE_MODEL_REVISION` — optional shared revision/tag/commit
- `HF_TOKEN` — only needed for private or gated repos

If `HF_HOME` / `HF_HUB_CACHE` are not set explicitly, the app will automatically use persistent `/data` storage when it exists and otherwise fall back to the standard cache under the app user's home directory.

## Dependency files

- `runtime-requirements.txt` is the actual Docker runtime dependency set used by the app image.
- Root `requirements.txt` is kept intentionally minimal so Hugging Face Spaces Dev Mode bootstrap layers do not try to reinstall the full CUDA/PyTorch stack outside the Docker image.

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

### Deployment helper scripts

- `python scripts/stage_space_bundle.py --overwrite --output-dir /tmp/dino_space_minimal_upload`
  - stages a code-only upload bundle for the current multi-model Space without local caches or checkpoints.
- `python scripts/publish_model_repo.py --family aiendo --repo-id <owner/repo> --model-dir /path/to/model`
  - publishes one model family to a Hugging Face **model repo** and automatically switches to `upload_large_folder()` for very large bundles.
- `python scripts/publish_space_repo.py --repo-id <owner/space> --dino-model-repo-id <owner/dino-repo> --aiendo-model-repo-id <owner/aiendo-repo> --vjepa2-model-repo-id <owner/vjepa2-repo>`
  - stages/uploads the Docker Space bundle and updates the key Space environment variables for the three-model demo.

## Local smoke test

Once the Space dependencies are installed, you can smoke test a predictor directly:

```bash
python scripts/smoke_test.py --model dinov2 --model-dir /path/to/model
python scripts/smoke_test.py --model aiendo --model-dir /path/to/model
python scripts/smoke_test.py --model vjepa2 --model-dir /path/to/model
```

## Scope of v1

- Streamlit UI
- project-portfolio landing page with DINO-Endo Surgery as the first hosted workspace
- three-model slider for DINO-Endo, AI-Endo, and V-JEPA2, with DINO-Endo selected by default
- image upload and video upload
- dashboard-style model/runtime status
- robust video metadata probing with OpenCV + ffprobe fallback
- large single-file uploads up to the configured Streamlit cap
- per-frame phase timeline output for video
- optional live encoder/decoder explainability sidebar with true attention where available and labeled proxies elsewhere
- JSON / CSV export

Not included in v1:

- auth / user management
- SQL database
- PDF/HTML report generation
- background queue processing
- polyp segmentation
