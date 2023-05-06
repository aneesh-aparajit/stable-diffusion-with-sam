import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry

MODEL_CHECKPOINT = './weights/sam_vit_h_4b8939.pth'
MODEL_TYPE = 'vit_h'
DEVICE = 'cpu'

selected_pixels = []

sam = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_CHECKPOINT)
sam.to(device=DEVICE)

predictor = SamPredictor(sam_model=sam)
pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
)
pipeline = pipeline.to(DEVICE)


def on_select(image, evt: gr.SelectData):  # SelectData is a subclass of EventData
    selected_pixels.append(evt.index)
    input_points = np.array(selected_pixels)
    input_labels = np.ones(shape=(input_points.shape[0]))

    image = np.array(image.convert('RGB'))
    predictor.set_image(image=image)

    masks, _, _ = predictor.predict(
        point_coords=input_points, 
        point_labels=input_labels, 
        multimask_output=False
    )
    mask = Image.fromarray(masks[0, :, :])
    return mask
    
def get_inpainting(prompt, image, mask):
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    image = pipeline(prompt=prompt, image=image, mask_image=mask).images[0]
    return image


with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion with Segment Anything!")
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil')
        with gr.Column():
            mask = gr.Image(type='pil')
        with gr.Column():
            output = gr.Image(type='pil')
    with gr.Row():
        prompt = gr.Textbox(placeholder="Prompt")

    image.select(fn=on_select, inputs=image, outputs=mask)
    
    btn = gr.Button("Run")
    btn.click(fn=get_inpainting, inputs=[prompt, image, mask], outputs=output)

demo.launch()
