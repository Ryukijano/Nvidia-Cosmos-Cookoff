import gradio as gr
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
import torchvision.transforms as T
#import torchvision.transforms.v2 as T2
import cv2
from PIL import Image
import numpy as np

output_res = (768,768)

conditioning_image_transforms = T.Compose(
    [
        #T2.ScaleJitter(target_size=output_res, scale_range=(0.5, 3.0)),
        #T2.RandomCrop(size=output_res, pad_if_needed=True, padding_mode="symmetric"),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

cnet = ControlNetModel.from_pretrained("./models/catcon-controlnet-wd", torch_dtype=torch.float32, from_flax=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "./models/wd-1-5-b2",
        controlnet=cnet,
        torch_dtype=torch.float32,
        )
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

generator = torch.manual_seed(0)

# inference function takes prompt, negative prompt and image
def infer(prompt, negative_prompt, image):
    # implement your inference function here

    #cond_input = conditioning_image_transforms(np.array(image)))
    
    output = pipe(
        prompt,
        image,
        generator=generator,
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
    ],
    outputs=gr.Gallery().style(grid=[2], height="auto"),
    title="Generate controlled outputs with Categorical Conditioning on Waifu Diffusion 1.5 beta 2.",
    description="This Space uses image examples as style conditioning.",
    examples=[["1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, watercolor, night, turtleneck", "low quality", "wikipe_cond_1.png"]],
    allow_flagging=False,
).launch(enable_queue=True)

