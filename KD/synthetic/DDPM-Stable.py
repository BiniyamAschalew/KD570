#pip install transformers diffusers mediapy accelerate

import mediapy as media
import torch
from diffusers import StableDiffusionPipeline
import random

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
scheduler = None

device = "cuda"

if model_id.startswith("stabilityai/"):
    model_revision = "fp16"
else:
    model_revision = None

if scheduler is None:
    pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision=model_revision,
            )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision=model_revision,
            )

pipe = pipe.to(device)
pipe.enable_xformers_memory_efficient_attention()

if model_id.endswith('-base'):
    image_length = 512
else:
    image_length = 768

prompt = "a photo of a sugar glider eating fish"
num_images = 4
seed = 1000

images = pipe(
    prompt,
    height = image_length,
    width = image_length,
    num_inference_steps = 50,
    guidance_scale = 9,
    num_images_per_prompt = num_images,
    generator = torch.Generator("cuda").manual_seed(seed)
    ).images

media.show_images(images)
display(f"Seed: {seed}")
images[0].save("../samples/stdm-output.jpg")