from typing import Optional

import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from multiprocessing import cpu_count
import requests
import io
import os
from PIL import Image

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    DEISMultistepScheduler,
    HeunDiscreteScheduler,
    EulerDiscreteScheduler,
)

qrcode_generator = qrcode.QRCode(
    version=1,
    error_correction=qrcode.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)

controlnet = ControlNetModel.from_pretrained(
    "./control_v1p_sd15_qrcode", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "./stable-diffusion-v1-5",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda")
pipe.enable_xformers_memory_efficient_attention()


def resize_for_condition_image(input_image: Image.Image, resolution: int):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=Image.LANCZOS)
    return img


SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "DPM++ Karras": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True),
    "Heun": lambda config: HeunDiscreteScheduler.from_config(config),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
    "DDIM": lambda config: DDIMScheduler.from_config(config),
    "DEIS": lambda config: DEISMultistepScheduler.from_config(config),
}


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    strength: float = 0.8,
    seed: int = -1,
    init_image: Optional[Image.Image] = None,
    qrcode_image: Optional[Image.Image] = None,
    use_qr_code_as_init_image = True,
    sampler = "DPM++ Karras SDE",
):
    if prompt is None or prompt == "":
        raise gr.Error("Prompt is required")

    if qrcode_image is None and qr_code_content == "":
        raise gr.Error("QR Code Image or QR Code Content is required")

    pipe.scheduler = SAMPLER_MAP[sampler](pipe.scheduler.config)

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    if qr_code_content != "" or qrcode_image.size == (1, 1):
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color="white")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)
    else:
        print("Using QR Code Image")
        qrcode_image = resize_for_condition_image(qrcode_image, 768)

    # hack due to gradio examples
    init_image = qrcode_image

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        control_image=qrcode_image,  # type: ignore
        width=768,  # type: ignore
        height=768,  # type: ignore
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),  # type: ignore
        generator=generator,
        strength=float(strength),
        num_inference_steps=40,
    )
    return out.images[0]  # type: ignore


with gr.Blocks() as blocks:
    gr.Markdown(
        """
#                                 YumChina QRCode AI Art Generator
## 💡 如何使用百胜中国艺术二维码生成器生成漂亮的二维码
我们使用二维码图像作为初始图像和控制图像，这允许您生成与您提供的提示非常自然地融合在一起的二维码。
强度参数定义了添加到二维码中的噪声量，噪声二维码随后通过Controlnet引导向您的提示和二维码图像。
使用高强度值在0.8到0.95之间，并选择一个调节尺度在0.6到2.0之间。
这种模式可以说实现了美观上最吸引人的二维码图像，但也需要更多地调整控制网的调节尺度和强度值。如果生成的图像看起来太像原始二维码，请确保轻轻增加强度值并减少调节尺度。同时，请查看下面的examples。

                """
    )

    with gr.Row():
        with gr.Column():
            qr_code_content = gr.Textbox(
                label="二维码内容",
                info="文本 或者 URL",
                value="",
            )
            with gr.Accordion(label="QR Code Image (Optional)", open=False):
                qr_code_image = gr.Image(
                    label="QR Code Image (Optional). Leave blank to automatically generate QR code",
                    type="pil",
                )

            prompt = gr.Textbox(
                label="提示词语",
                info="Prompt that guides the generation towards",
            )
            negative_prompt = gr.Textbox(
                label="消极提示词",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            use_qr_code_as_init_image = gr.Checkbox(label="Use QR code as init image", value=True, interactive=False, info="Whether init image should be QR code. Unclick to pass init image or generate init image with Stable Diffusion 2.1")

            with gr.Accordion(label="Init Images (Optional)", open=False, visible=False) as init_image_acc:
                init_image = gr.Image(label="Init Image (Optional). Leave blank to generate image with SD 2.1", type="pil")


            with gr.Accordion(
                label="Params: The generated QR Code functionality is largely influenced by the parameters detailed below",
                open=True,
            ):
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.01,
                    value=1.1,
                    label="Controlnet Conditioning Scale",
                )
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.01, value=0.9, label="Strength"
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    step=0.25,
                    value=7.5,
                    label="Guidance Scale",
                )
                sampler = gr.Dropdown(choices=list(SAMPLER_MAP.keys()), value="DPM++ Karras SDE", label="Sampler")
                seed = gr.Slider(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                )
            with gr.Row():
                run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Result Image")
    run_btn.click(
        inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
            init_image,
            qr_code_image,
            use_qr_code_as_init_image,
            sampler,
        ],
        outputs=[result_image],
        concurrency_limit=1
    )

    gr.Examples(
        examples=[
            [
                "http://www.yumchina.com/",
                "A sky view of a colorful lakes and rivers flowing through the desert",
                "ugly, disfigured, low quality, blurry, nsfw",
                7.5,
                1.3,
                0.9,
                5392011833,
                None,
                None,
                True,
                "DPM++ Karras SDE",
            ],
            [
                "http://www.yumchina.com/",
                "Bright sunshine coming through the cracks of a wet, cave wall of big rocks",
                "ugly, disfigured, low quality, blurry, nsfw",
                7.5,
                1.11,
                0.9,
                2523992465,
                None,
                None,
                True,
                "DPM++ Karras SDE",
            ],
            [
                "http://www.yumchina.com/",
                "Sky view of highly aesthetic, ancient greek thermal baths  in beautiful nature",
                "ugly, disfigured, low quality, blurry, nsfw",
                7.5,
                1.5,
                0.9,
                2523992465,
                None,
                None,
                True,
                "DPM++ Karras SDE",
            ],
        ],
        fn=inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            strength,
            seed,
            init_image,
            qr_code_image,
            use_qr_code_as_init_image,
            sampler,
        ],
        outputs=[result_image],
        cache_examples=True,
    )

blocks.queue(max_size=20,api_open=False)

blocks.launch(share=bool(os.environ.get("SHARE", False)), show_api=False)