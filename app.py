import json
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



API_KEY = "o5mxxS7YRCjTF3S1ZOKqNLBb"
SECRET_KEY = "Xw7tX3CNw3Nm1V4YOyFwpHhywypXjTBX"


def call_wenxin_api(text):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token=" + get_access_token()


    headers = {
        'Content-Type': 'application/json'
    }

    # 序列化 JSON 请求体
    json_payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ]
    })

    # 检查 JSON 序列化后的长度，并确保其为奇数
    if len(json_payload) % 2 == 0:
        # 在 JSON 字符串的末尾添加一个空格（或其他字符）
        json_payload += " "

        # 发送请求，这里直接使用 json_payload 而不是 payload
    response = requests.post(url, headers=headers, data=json_payload)

    # 输出响应内容
    print(response.text)
    return response.text  # 返回响应内容而不是response对象


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def generate_prompt(name):
    prompt_template = "假设你是提示词生成专家，请模仿下面的几个提示词：1.A sky view of a colorful lakes and rivers flowing through the desert  2.Bright sunshine coming through the cracks of a wet, cave wall of big rocks  3.Sky view of highly aesthetic, ancient greek thermal baths  in beautiful nature   每当我输入一个词语的时候 请帮我生成一段类似上面格式之一的英语提示词,要求长度不超过15个单词，每次只能生成一句话开头不能携带序号，我输入的是{}"
    input_for_api = prompt_template.format(name)
    print('-------开始准备调用接口')
    # 假设call_wenxin_api函数返回的是一个JSON格式的字符串
    response_str = call_wenxin_api(input_for_api)

    # 解析JSON字符串为Python字典
    response_dict = json.loads(response_str)

    # 提取result字段的内容
    result_content = response_dict.get('result', '未找到result字段')

    # 打印提取到的内容
    print(result_content)

    # 返回提取到的内容
    return result_content,result_content

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
            fast_prompt = gr.Textbox(
                label="快速提示词",
                info="Prompt that guides the generation towards",
            )
            generate_btn = gr.Button("快速帮你生成提示语")  # 添加一个生成提示词的按钮
            prompt = gr.Textbox(
                label="提示词语",
                info="Prompt that guides the generation towards",
            )
            negative_prompt = gr.Textbox(
                label="消极提示词",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            use_qr_code_as_init_image = gr.Checkbox(label="Use QR code as init image", value=True, interactive=False, info="Whether init image should be QR code. Unclick to pass init image or generate init image with Stable Diffusion 2.1")


            # 添加一个隐藏的输出组件，用于触发更新（这里不会在界面上显示）
            dummy_output = gr.Textbox(visible=False)

            # 设置按钮点击事件
            generate_btn.click(
                generate_prompt,
                inputs=fast_prompt,
                outputs=[fast_prompt, dummy_output]
            )
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