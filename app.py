import gradio as gr
import jax
import numpy as np
import jax.numpy as jnp
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from PIL import Image
from diffusers import FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel
from diffusers import UniPCMultistepScheduler


def create_key(seed=0):
    return jax.random.PRNGKey(seed)


output_res = (768,768)



def conditioning_image_transforms(image):
    image = jnp.array(image) / 255.0 #convert the image into JAX array and normalize to [0,1]
    image = (image - 0.5) * 2 # normalize to [-1, 1]
    return image


controlnet, controlnet_params = FlaxControlNetModel.from_pretrained("./models/catcon-controlnet-wd", dtype=jnp.bfloat16
)
pipe = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        "./models/wd-1-5-b2", dtype=jnp.bfloat16
        )
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

key = jax.random.PRNGKey(0)

# inference function takes prompt, negative prompt and image
def infer(prompt, negative_prompt, image):
    if seed == 0:
        seed = jax.random.randint(jax.random.PRNGKey(0), (), 1, 1001)[0] #generate a random seed if slider is set to 0

    key = create_key(seed) #using the create_key function created above
    
    output = pipe(
        prompt,
        image,
        generator=key, #using the JAX random number generator defined above
        num_images_per_prompt=1,
        num_inference_steps=20
    )

    return output.images

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