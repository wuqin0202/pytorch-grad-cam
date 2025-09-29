#!/usr/bin/env python3
"""
修复版本的CAM，专门处理Transformer层的输出tuple问题
"""

import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

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
        default='testset/test1.jpg',
        help='Input image path')
    parser.add_argument(
        '--text-query',
        type=str,
        default="识别图片所有绿植的位置？给出框位置：(xy, xy)",
        help='Text query'
    )
    parser.add_argument('--output-dir', type=str, default='./output/qwen2_5vl_decoder', help='Output directory')
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
    """
    Qwen2.5-VL模型的关键token提取器（Decoder版本）

    主要功能：
    1. 加载Qwen2.5-VL模型和处理器
    2. 基于blur方法识别关键token
    3. 返回关键token的logits用于解释性分析

    使用示例：
        >>> classifier = ImageClassifier()
        >>> classifier.set_text_query("识别图片中的物体")
        >>> img_tensor = load_and_preprocess_images([image_path])[0]
        >>> positions, token_ids = classifier.find_key_tokens(img_tensor)
        >>> logits = classifier.forward(img_tensor)
        >>> print(f"关键token位置: {positions}")
        >>> print(f"Logits形状: {logits.shape}")
    """

    def __init__(self, model_dir=None):
        """
        初始化Qwen2.5-VL模型

        Args:
            model_dir (str, optional): 模型路径，如果为None则使用默认的在线模型
        """
        super(ImageClassifier, self).__init__()

        # 模型加载逻辑
        if model_dir is None:
            # 使用默认的在线模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                torch_dtype="auto",
                device_map="cuda"
            )
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        else:
            # 使用本地模型
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype="auto",
                device_map="cuda"
            )
            self.processor = AutoProcessor.from_pretrained(model_dir)

        # 存储当前处理的文本查询模板
        self.current_text = None
        # 存储关键token的位置和ID
        self.key_token_positions = []
        self.key_token_ids = []
        # 存储生成的token ID用于forward方法
        self.generated_ids = None

    def _tensor2pil(self, tensor):
        """
        将tensor转换为PIL图像

        Args:
            tensor: 图像张量，形状为 [C, H, W] 或 [B, C, H, W]

        Returns:
            PIL.Image: PIL图像对象
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        # 假设tensor已经是[0,1]范围，转换为[0,255]
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0

        # 转换为numpy并调整维度顺序
        img_np = tensor.cpu().numpy().astype(np.uint8)
        if img_np.shape[0] == 3:  # CHW -> HWC
            img_np = np.transpose(img_np, (1, 2, 0))

        return Image.fromarray(img_np)

    def _pil2tensor(self, pil_image):
        """
        将PIL图像转换为tensor

        Args:
            pil_image: PIL图像对象

        Returns:
            torch.Tensor: 图像张量
        """
        # 转换为numpy数组
        img_np = np.array(pil_image)

        # 如果是RGB图像，调整维度顺序 HWC -> CHW
        if len(img_np.shape) == 3:
            img_np = np.transpose(img_np, (2, 0, 1))

        # 转换为tensor并归一化到[0,1]
        tensor = torch.from_numpy(img_np).float() / 255.0

        return tensor

    def set_text_query(self, text_query):
        """
        设置当前的文本查询，生成对话模板

        Args:
            text_query (str): 用户的文本查询
        """
        # 构建消息格式，图像路径会在后续处理中填充
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "",  # 占位符，实际图像会在处理时填充
                    },
                    {"type": "text", "text": text_query},
                ],
            }
        ]

        # 生成对话模板
        self.current_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def _generate_response(self, img_tensor):
        """
        第一步：生成模型回答（无梯度推理）

        Args:
            img_tensor: 图像张量

        Returns:
            str: 生成的文本回答
            torch.Tensor: 生成的token IDs
        """
        # 将tensor转换为PIL图像
        img_pil = self._tensor2pil(img_tensor)

        # 准备输入
        inputs = self.processor(
            text=[self.current_text],
            images=img_pil,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # 生成回答
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # 解码生成的文本
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return output_text, generated_ids_trimmed[0]

    def _create_blurred_image(self, img_tensor, kernel_size=None):
        """
        创建高斯模糊图像

        Args:
            img_tensor: 原始图像张量
            kernel_size: 高斯核大小，如果为None则自动计算

        Returns:
            torch.Tensor: 模糊后的图像张量
        """
        # 转换为PIL图像
        img_pil = self._tensor2pil(img_tensor)

        # 计算高斯核大小
        if kernel_size is None:
            kernel_size = get_kernel_size(img_pil.size)

        # 应用高斯模糊
        blur_pil = apply_gaussian_blur(img_pil, kernel_size)
        blur_pil.save("blurred_image.jpg")  # 保存模糊图像用于调试

        # 转换回tensor格式
        blur_tensor = self._pil2tensor(blur_pil)

        return blur_tensor

    def reset_state(self):
        """
        重置内部状态，用于处理新的图像
        """
        self.key_token_positions = []
        self.key_token_ids = []
        self.generated_ids = None

    @torch.no_grad()
    def _compute_logits_with_blur(self, img_tensor, blur_tensor, full_input_ids):
        """
        第二步：使用模糊图像计算logits，识别关键token

        Args:
            img_tensor: 原始图像张量
            blur_tensor: 模糊图像张量
            full_input_ids: 包含生成回答的完整输入token序列

        Returns:
            list: 关键token的位置
            list: 关键token的ID
        """
        # 转换图像为PIL格式
        img_pil = self._tensor2pil(img_tensor)
        blur_pil = self._tensor2pil(blur_tensor)

        # 准备输入（原图）
        inputs_orig = self.processor(
            text=[self.current_text],
            images=img_pil,
            padding=True,
            return_tensors="pt",
        )
        inputs_orig = inputs_orig.to("cuda")

        # 准备输入（模糊图）
        inputs_blur = self.processor(
            text=[self.current_text],
            images=blur_pil,
            padding=True,
            return_tensors="pt",
        )
        inputs_blur = inputs_blur.to("cuda")

        start_idx = inputs_orig.input_ids.shape[1] - 1  # 生成token开始位置
        end_idx = start_idx + full_input_ids.shape[0]

        # 拼接完整的输入token序列（原输入 + 生成的回答）
        full_input_ids_tensor = torch.cat([inputs_orig.input_ids, full_input_ids.unsqueeze(0)], dim=1)
        inputs_orig['input_ids'] = full_input_ids_tensor
        inputs_orig['attention_mask'] = torch.ones_like(full_input_ids_tensor)
        inputs_blur['input_ids'] = full_input_ids_tensor
        inputs_blur['attention_mask'] = torch.ones_like(full_input_ids_tensor)

        # 计算原图的logits
        outputs_orig = self.model(
            **inputs_orig,
            return_dict=True
        )
        logits_orig = outputs_orig.logits

        # 计算模糊图的logits
        outputs_blur = self.model(
            **inputs_blur,
            return_dict=True
        )
        logits_blur = outputs_blur.logits

        # 计算概率
        probs_orig = F.softmax(logits_orig, dim=-1)
        probs_blur = F.softmax(logits_blur, dim=-1)

        # 获取生成token对应的概率
        generated_token_ids = full_input_ids
        probs_orig_tokens = torch.gather(
            probs_orig[0, start_idx:end_idx],
            1,
            generated_token_ids.unsqueeze(1)
        ).squeeze()
        probs_blur_tokens = torch.gather(
            probs_blur[0, start_idx:end_idx],
            1,
            generated_token_ids.unsqueeze(1)
        ).squeeze()

        # 识别关键token（log概率差值大于阈值）
        special_ids = [self.processor.tokenizer.pad_token_id,
                      self.processor.tokenizer.eos_token_id,
                      self.processor.tokenizer.bos_token_id]
        special_ids = [id for id in special_ids if id is not None]

        # 确保概率张量维度正确
        if probs_orig_tokens.dim() == 0:
            probs_orig_tokens = probs_orig_tokens.unsqueeze(0)
        if probs_blur_tokens.dim() == 0:
            probs_blur_tokens = probs_blur_tokens.unsqueeze(0)

        # 避免log(0)的情况
        probs_orig_tokens = torch.clamp(probs_orig_tokens, min=1e-8)
        probs_blur_tokens = torch.clamp(probs_blur_tokens, min=1e-8)

        condition = (
            (torch.log(probs_orig_tokens) - torch.log(probs_blur_tokens) > 1.0) &
            (probs_orig_tokens >= 0.0) &
            (~torch.isin(generated_token_ids, torch.tensor(special_ids).to(generated_token_ids.device)))
        )

        positions = torch.where(condition)[0].tolist()
        token_ids = [generated_token_ids[idx].item() for idx in positions]

        return positions, token_ids

    def _get_key_token_logits(self, img_tensor, full_input_ids, positions):
        """
        第三步：使用原图获取关键token的logits

        Args:
            img_tensor: 原始图像张量
            full_input_ids: 完整输入token序列
            positions: 关键token位置列表

        Returns:
            torch.Tensor: 关键token的logits
        """
        if not positions:
            return torch.empty(0, self.model.config.vocab_size).to("cuda")

        # 转换图像为PIL格式
        img_pil = self._tensor2pil(img_tensor)

        # 准备输入
        inputs = self.processor(
            text=[self.current_text],
            images=img_pil,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        start_idx = inputs.input_ids.shape[1] - 1

        # 拼接完整的输入token序列
        full_input_ids_batch = torch.cat([inputs.input_ids, full_input_ids.unsqueeze(0)], dim=1)
        inputs['input_ids'] = full_input_ids_batch
        inputs['attention_mask'] = torch.ones_like(full_input_ids_batch)

        # 前向传播获取logits
        with torch.enable_grad():
            outputs = self.model(
                **inputs,
                return_dict=True
            )
            logits = outputs.logits

        # 提取关键token的logits
        key_token_logits = logits[0, start_idx:start_idx + len(full_input_ids)][positions]

        return key_token_logits.unsqueeze(0)

    def find_key_tokens(self, img_tensor):
        """
        找到关键token的位置和ID

        Args:
            img_tensor: 输入图像张量

        Returns:
            tuple: (关键token位置列表, 关键token ID列表)
        """
        # 确保已设置文本查询
        if self.current_text is None:
            raise ValueError("请先调用set_text_query()设置文本查询")

        # 第一步：生成回答
        generated_text, generated_ids = self._generate_response(img_tensor)
        print(f"生成的文本: {generated_text}")

        # 第二步：识别关键token
        blur_tensor = self._create_blurred_image(img_tensor)

        # 计算关键token位置
        positions, token_ids = self._compute_logits_with_blur(
            img_tensor, blur_tensor, generated_ids
        )

        # 存储结果
        self.key_token_positions = positions
        self.key_token_ids = token_ids
        self.generated_ids = generated_ids  # 保存生成的ID用于forward方法

        return positions, token_ids

    def forward(self, img_tensor):
        """
        前向传播：返回关键token的logits

        Args:
            img_tensor: 输入图像张量

        Returns:
            torch.Tensor: 关键token的logits，形状为 [num_key_tokens, vocab_size]
        """
        # 确保已找到关键token
        if not self.key_token_positions:
            self.find_key_tokens(img_tensor)

        # 第三步：获取关键token的logits
        key_token_logits = self._get_key_token_logits(
            img_tensor,
            self.generated_ids,
            self.key_token_positions
        )

        return key_token_logits


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


def get_kernel_size(image_size):
    """
    根据图像大小计算合适的高斯核大小

    Args:
        image_size: 图像尺寸 (width, height)

    Returns:
        int: 高斯核大小（奇数）
    """
    # 基于图像较大维度计算核大小
    max_dim = max(image_size)

    # 核大小约为图像最大维度的1/20，确保为奇数
    kernel_size = max(21, int(max_dim / 10))
    if kernel_size % 2 == 0:
        kernel_size += 1

    return kernel_size

def apply_gaussian_blur(image, kernel_size):
    """
    对PIL图像应用高斯模糊

    Args:
        image: PIL图像对象
        kernel_size: 高斯核大小

    Returns:
        PIL.Image: 模糊后的图像
    """
    # 转换为numpy数组
    img_array = np.array(image)

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(
        img_array,
        (kernel_size, kernel_size),
        sigmaX=kernel_size-1
    )

    # 转换回PIL图像
    return Image.fromarray(blurred.astype(np.uint8))


class Qwen2_5VLTargets(nn.Module):
    """
    用于处理关键token的目标类
    """
    def __init__(self, key_token_ids):
        super().__init__()
        self.key_token_ids = torch.Tensor(key_token_ids)

    def __call__(self, key_logits):
        self.key_token_ids = self.key_token_ids.to(key_logits.device)
        return torch.nn.functional.cross_entropy(
            key_logits.to(torch.float32),  # 确保为float32
            self.key_token_ids.to(torch.long)  # 确保为int64
        )


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
    if H is None or W is None:
        # 如果H, W未设置，使用默认值
        H, W = 224, 224
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
    model.set_text_query(args.text_query)
    print(f"文本查询已设置: {args.text_query}")

    # 加载和预处理图片
    img_tensor = load_and_preprocess_images([args.image_path])
    H, W = img_tensor.shape[2:]
    print(f"图片尺寸 H×W: {H}×{W}")

    # 第一步：找到关键token
    print("正在寻找关键token...")
    positions, token_ids = model.find_key_tokens(img_tensor[0])
    print(f"找到关键 token: 位置 {positions}, 文本 {model.processor.tokenizer.decode(token_ids)}")

    if not token_ids:
        print("未找到关键token，使用默认CAM计算")
        targets = None
    else:
        # 设置目标为关键token
        targets = [Qwen2_5VLTargets(token_ids)]  # type: ignore

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
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)
        print(f"CAM计算成功，形状: {grayscale_cam.shape}")

        # Extract single image from batch
        grayscale_cam = grayscale_cam[0, :]

        # Create visualization
        cam_image = show_cam_on_image(img_tensor[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam)

        # Save result
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f'{os.path.basename(args.image_path)}_{args.method}_key_tokens.jpg')
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
