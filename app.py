import gradio as gr
import jax.numpy as jnp
from diffusers import FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel
from diffusers import FlaxScoreSdeVeScheduler, FlaxDPMSolverMultistepScheduler
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torchvision
import torchvision.transforms as T
from flax.jax_utils import replicate
from flax.training.common_utils import shard
import cv2
import PIL
from PIL import Image
import numpy as np
import jax

def create_key(seed=0):
    return jax.random.PRNGKey(seed)


output_res = (768,768)


conditioning_image_transforms = T.Compose(
    [
        #T2.ScaleJitter(target_size=output_res, scale_range=(0.5, 3.0))),
        T.RandomCrop(size=output_res, pad_if_needed=True, padding_mode="symmetric"),
        T.ToTensor(),
        #T.Normalize([0.5], [0.5]),
    ]
)

cnet, cnet_params = FlaxControlNetModel.from_pretrained("Ryukijano/catcon-controlnet-wd", dtype=jnp.bfloat16, from_flax=True)
pipe, params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        "./models/wd-1-5-b2-flax",
        controlnet=cnet,
        revision="flax",
        dtype=jnp.bfloat16,
        )

key = jax.random.PRNGKey(0)

# inference function takes prompt, negative prompt and image
# inference function takes prompt, negative prompt and image
def infer(prompt, negative_prompt, image):
    # implement your inference function here
    params["controlnet"] = cnet_params
    num_samples = 1

    inp = Image.fromarray(image)

    cond_input = conditioning_image_transforms(inp)
    cond_input = T.ToPILImage()(cond_input)

    cond_img_in = pipe.prepare_image_inputs([cond_input] * num_samples)
    cond_img_in = shard(cond_img_in)

    prompt_in = pipe.prepare_text_inputs([prompt] * num_samples)
    prompt_in = shard(prompt_in)

    n_prompt_in = pipe.prepare_text_inputs([negative_prompt] * num_samples)
    n_prompt_in = shard(n_prompt_in)

    rng = get_random(0)
    rng = jax.random.split(rng, jax.device_count())

    p_params = replicate(params)
    
    output = pipe(
        prompt_ids=prompt_in,
        image=cond_img_in,
        params=p_params,
        prng_seed=rng,
        num_inference_steps=70,
        neg_prompt_ids=n_prompt_in,
        jit=True,
            ).images

    output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
    return output_images

gr.Interface(
    infer,
    inputs=[
        gr.Textbox(
            label="Enter prompt",
            max_lines=1,
            placeholder="1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck",
        ),
        gr.Textbox(
            label="Enter negative prompt",
            max_lines=1,
            placeholder="low quality",
        ),
        gr.Image(type="pil"),
        gr.Slider(minimum=0, maximum=100, step=1, default=0, label="Seed (0 for random seed)")
    ],
    outputs=gr.Gallery().style(grid=[2], height="auto"),
    title="Generate controlled outputs with Categorical Conditioning on Stable Diffusion 1.5 beta 2.",
    description="This Space uses image examples as style conditioning.",
    examples=[["1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck", "low quality", "wikipe_cond_1.png"]],
    allow_flagging=False,
).launch(enable_queue=True)
