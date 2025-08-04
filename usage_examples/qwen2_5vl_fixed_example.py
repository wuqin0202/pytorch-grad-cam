#!/usr/bin/env python3
"""
修复版本的CAM，专门处理Transformer层的输出tuple问题
"""

import os
import argparse
import cv2
import numpy as np
import torch
from torch import nn

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("The Qwen2_5_VLForConditionalGeneration package is not installed. Please install it.")
    exit(1)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

from vggt_utils import load_and_preprocess_images
from utils import tensor2pil

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='/data25/wuqin/projects/FlagEmbedding/research/visual_bge/vl_query_samples/imgs/test1.jpg',
        help='Input image path')
    parser.add_argument(
        '--text-query',
        type=str,
        default="识别图片所有绿植的位置？给出框位置：(xy, xy)",
        help='Text query'
    )
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle component'
             'of cam_weights*activations')
    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')

    return args


class ImageClassifier(nn.Module):
    def __init__(self, model_dir=None):
        super(ImageClassifier, self).__init__()
        if model_dir is None:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="cuda"
            )
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        else:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir, torch_dtype="auto", device_map="cuda"
            )
            processor = AutoProcessor.from_pretrained(model_dir)
        self.model = model
        self.processor = processor
        self.current_text = None

    def set_text_query(self, image_path, text_query):
        """设置当前的文本查询"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": text_query},
                ],
            }
        ]
        self.current_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def forward(self, img_tensor):
        if self.current_text is None:
            raise ValueError("请先使用 set_text_query 方法设置文本查询")

        img_pil = tensor2pil(img_tensor[0])
        inputs = self.processor(
            text=[self.current_text],
            images=img_pil,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Forward pass through the model
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = logits
        return probs.float()


class FixedActivationsAndGradients(ActivationsAndGradients):
    """修复版本的ActivationsAndGradients，正确处理Transformer层的tuple输出"""

    def save_activation(self, module, input, output):
        # 处理Transformer层的tuple输出
        if isinstance(output, tuple):
            activation = output[0]  # 通常hidden_states是第一个元素
        else:
            activation = output

        if self.detach:
            if self.reshape_transform is not None:
                activation = self.reshape_transform(activation)
            self.activations.append(activation.cpu().detach())
        else:
            self.activations.append(activation)

    def save_gradient(self, module, input, output):
        # 处理Transformer层的tuple输出
        if isinstance(output, tuple):
            target_output = output[0]  # hidden_states
        else:
            target_output = output

        if not hasattr(target_output, "requires_grad") or not target_output.requires_grad:
            print(f"⚠️ 输出不需要梯度: requires_grad={getattr(target_output, 'requires_grad', None)}")
            return

        print(f"✓ 注册梯度hook到输出tensor: {target_output.shape}")

        # Gradients are computed in reverse order
        def _store_grad(grad):
            print(f"✓ 捕获到梯度: {grad.shape}")
            if self.detach:
                if self.reshape_transform is not None:
                    grad = self.reshape_transform(grad)
                self.gradients = [grad.cpu().detach()] + self.gradients
            else:
                self.gradients = [grad] + self.gradients

        target_output.register_hook(_store_grad)


class FixedGradCAM(GradCAM):
    """修复版本的GradCAM，使用修复的ActivationsAndGradients"""

    def __init__(self, model, target_layers, reshape_transform=None, **kwargs):
        # 不调用父类的__init__，自己初始化
        self.model = model.eval()
        self.target_layers = target_layers
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = False
        self.uses_gradients = True
        self.detach = True

        # 使用修复版本的ActivationsAndGradients
        self.activations_and_grads = FixedActivationsAndGradients(
            self.model, target_layers, reshape_transform, self.detach
        )


H, W = None, None
def reshape_transform(tensor):
    """Reshape transform for Vision Transformer features"""
    global H, W
    p_h, p_w = (H//28), (W//28)

    print(f"reshape_transform 输入: {type(tensor)}")

    if isinstance(tensor, tuple):
        tensor = tensor[0]  # 取hidden_states
        print(f"  从tuple中取第一个元素: {tensor.shape}")

    print(f"  处理tensor形状: {tensor.shape}")

    if len(tensor.shape) == 3:
        # 对于transformer层，我们需要处理[batch, seq, hidden]格式
        batch, seq_len, hidden_dim = tensor.shape
        print(f"  3D tensor: batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}")
        print(f"  需要的patch数量: {p_h * p_w}")

        if seq_len != p_h * p_w:
            # 对于language model层，我们需要提取图像相关的tokens
            if seq_len >= p_h * p_w + 15:
                print(f"  执行切片: [:, 15:{p_h*p_w+15}, :]")
                tensor = tensor[:, 15:p_h*p_w+15, :]
                print(f"  切片后形状: {tensor.shape}")
            else:
                print(f"  ⚠️ 序列长度不足，无法切片")
                available_len = min(seq_len, p_h * p_w)
                tensor = tensor[:, :available_len, :]
                p_h = int(np.sqrt(available_len))
                p_w = available_len // p_h
                print(f"  调整后的patch尺寸: {p_h}x{p_w}")

    print(f"  最终tensor形状: {tensor.shape}")
    result = tensor.reshape(tensor.size(0), p_h, p_w, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)
    print(f"  reshape后形状: {result.shape}")
    return result


if __name__ == '__main__':
    args = get_args()

    # Initialize model
    model = ImageClassifier()

    # 使用language model的倒数第二层
    target_layers = [model.model.model.language_model.layers[-2]]
    print(f"使用目标层: {type(target_layers[0])}")

    # 确保参数需要梯度
    for layer in target_layers:
        for param in layer.parameters():
            param.requires_grad = True

    # 设置文本查询
    model.set_text_query(args.image_path, args.text_query)
    print(f"文本查询已设置: {args.text_query}")

    # 加载和预处理图片
    img_tensor = load_and_preprocess_images([args.image_path])
    H, W = img_tensor.shape[2:]
    print(f"图片尺寸 H×W: {H}×{W}")

    # 使用修复版本的GradCAM
    print(f"初始化修复版本的GradCAM...")
    cam = FixedGradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )

    print(f"开始CAM计算...")
    try:
        grayscale_cam = cam(input_tensor=img_tensor,
                            targets=None,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)
        print(f"CAM计算成功，形状: {grayscale_cam.shape}")

        # Extract single image from batch
        grayscale_cam = grayscale_cam[0, :]

        # Create visualization
        cam_image = show_cam_on_image(img_tensor[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam)

        # Save result
        output_path = f'qwen2_5vl_{args.method}_fixed_cam.jpg'
        cv2.imwrite(output_path, cam_image)
        print(f"CAM visualization saved to {output_path}")

    except Exception as e:
        print(f"CAM计算失败: {e}")
        print(f"错误类型: {type(e)}")

        # 调试信息
        print("调试CAM内部状态...")
        if hasattr(cam, 'activations_and_grads'):
            print(f"激活数量: {len(cam.activations_and_grads.activations) if cam.activations_and_grads.activations else 0}")
            print(f"梯度数量: {len(cam.activations_and_grads.gradients) if cam.activations_and_grads.gradients else 0}")

            if cam.activations_and_grads.activations:
                for i, act in enumerate(cam.activations_and_grads.activations):
                    if act is not None:
                        print(f"  激活[{i}]: {act.shape}")
                    else:
                        print(f"  激活[{i}]: None")

            if cam.activations_and_grads.gradients:
                for i, grad in enumerate(cam.activations_and_grads.gradients):
                    if grad is not None:
                        print(f"  梯度[{i}]: {grad.shape}")
                    else:
                        print(f"  梯度[{i}]: None")

        raise e
