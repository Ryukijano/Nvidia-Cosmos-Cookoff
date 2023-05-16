---
title: CatCon Controlnet SD 1 5 B2
emoji: 🐨
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.28.0
app_file: app.py
pinned: true
license: mit
tags:
- jax-diffusers-event
- jax
- Text-To-Image
- Diffusers
- Controlnet
- Stable-Diffusion
datasets:
- animelover/danbooru2022
models:
- Ryukijano/CatCon-Controlnet-WD-1-5-b2R
---

Experimental proof of concept made for the [Huggingface JAX/Diffusers community sprint](https://github.com/huggingface/community-events/tree/main/jax-controlnet-sprint)

[Demo available here](https://huggingface.co/spaces/Ryukijano/CatCon-One-Shot-Controlnet-SD-1-5-b2)
[My teammate's demo is available here] (https://huggingface.co/spaces/Cognomen/CatCon-Controlnet-WD-1-5-b2) 

This is a controlnet for the Stable Diffusion checkpoint [Waifu Diffusion 1.5 beta 2](https://huggingface.co/waifu-diffusion/wd-1-5-beta2) which aims to guide image generation by conditioning outputs with patches of images from a common category of the training target examples. The current checkpoint has been trained for approx. 100k steps on a filtered subset of [Danbooru 2021](https://gwern.net/danbooru2021) using artists as the conditioned category with the aim of learning robust style transfer from an image example.

Major limitations:

 - The current checkpoint was trained on 768x768 crops without aspect ratio checkpointing. Loss in coherence for non-square aspect ratios can be expected.
 - The training dataset is extremely noisy and used without filtering stylistic outliers from within each category, so performance may be less than ideal. A more diverse dataset with a larger variety of styles and categories would likely have better performance.
 - The Waifu Diffusion base model is a hybrid anime/photography model, and can unpredictably jump between those modalities.
 - As styling is sensitive to divergences in model checkpoints, the capabilities of this controlnet are not expected to predictably apply to other SD 2.X checkpoints.

Waifu Diffusion 1.5 beta 2 is licensed under [Fair AI Public License 1.0-SD](https://freedevproject.org/faipl-1.0-sd/). This controlnet imposes no restrictions beyond the MIT license, but it cannot be used independently of a base model.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference