# transformers==4.40.0
import argparse
import os
import cv2
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, SiglipProcessor
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pytorch_grad_cam import (
    GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM,
    XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device to use')
    parser.add_argument('--image-path', type=str,
                        default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--labels', type=str, nargs='+',
                        default=["a cat", "a dog", "a car", "a person", "a shoe", "green plant"],
                        help='Candidate labels')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Eigen-smooth the CAM')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'scorecam', 'gradcam++',
                                 'ablationcam', 'xgradcam', 'eigencam',
                                 'eigengradcam', 'layercam', 'fullgrad'])
    parser.add_argument('--output-dir', type=str,
                        default='./',
                        help='Output directory')
    parser.add_argument('--target-index', type=int, default=0,
                        help='Target label index for CAM visualization')
    return parser.parse_args()


def reshape_transform(tensor):
    """
    适配 ViT: tensor shape (B, L, C) -> (B, C, H, W)
    支持输入为 tuple（如 (hidden_states, )），只取第一个元素
    """
    # 兼容 tuple 输入
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    print("reshape_transform input shape:", tensor.shape)
    # 只处理 3D tensor
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.shape}")
    B, L, C = tensor.shape
    # 计算 patch 数量
    if int((L - 1) ** 0.5) ** 2 == L - 1:
        N = L - 1
        tokens = tensor[:, 1:, :]
    elif int(L ** 0.5) ** 2 == L:
        N = L
        tokens = tensor
    else:
        raise ValueError(
            f"Can't reshape {L} tokens (with/without cls) into square grid"
        )
    H = W = int(N ** 0.5)
    tokens = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
    print("reshape_transform output shape:", tokens.shape)
    return tokens


class ImageClassifier(nn.Module):
    def __init__(self, labels, model_dir="google/siglip2-large-patch16-512"):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            model_dir,
            attn_implementation="eager"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir)
        self.processor = SiglipProcessor(tokenizer=self.tokenizer, image_processor=self.image_processor)
        self.labels = labels

    def forward(self, x):
        texts = [f'This is a photo of {label}.' for label in self.labels]
        if isinstance(x, torch.Tensor):
            # 已经是 tensor，直接用
            tokenized = self.processor.tokenizer(
                texts,
                padding="max_length",
                max_length=64,
                return_tensors="pt",
                return_attention_mask=True
            )
            inputs = {
                "pixel_values": x,
                "input_ids": tokenized["input_ids"].to(self.model.device)
            }
            # 只有 attention_mask 存在时才加进去
            if "attention_mask" in tokenized:
                inputs["attention_mask"] = tokenized["attention_mask"].to(self.model.device)
        else:
            # 原始图片，正常处理
            inputs = self.processor(
                text=texts,
                images=x,
                padding="max_length",
                max_length=64,
                return_tensors="pt"
            ).to(self.model.device)

        outputs = self.model(**inputs)
        logits = outputs.logits_per_image / self.model.logit_scale.exp()
        probs = torch.sigmoid(logits)

        # 2. 如果想对所有类别同时做 Grad-CAM，可以返回 logits.sum()
        #    这里演示返回 logits，cam 库会自动对每个 target 做 loss=logit[target]
        return logits          # <-- 关键：返回 logits 而不是 prob


if __name__ == '__main__':
    args = get_args()

    methods = {
        "gradcam": GradCAM, "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus, "ablationcam": AblationCAM,
        "xgradcam": XGradCAM, "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM, "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    labels = args.labels
    model = ImageClassifier(labels).to(torch.device(args.device)).eval()

    print(model)
    # 关键：直接用最后一层 Transformer Block 的 mlp 子层（或 self_attn 也可尝试）
    target_layers = [model.model.vision_model.encoder.layers[-3].layer_norm2]

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (512, 512))
    rgb_img = rgb_img.astype(np.uint8)
    input_tensor = model.processor(
        images=[rgb_img],
        return_tensors="pt"
    )["pixel_values"].to(args.device)

    # 明确指定 target
    target_index = args.target_index
    targets = [ClassifierOutputTarget(target_index)]

    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)[0]

    cam_image = show_cam_on_image(rgb_img / 255.0, grayscale_cam, use_rgb=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # 输出文件名包含 target label
    target_label = labels[target_index].replace(" ", "_")
    output_path = os.path.join(args.output_dir,
                               f"{args.method}_cam_{target_label}_layers-3_layer_norm2.jpg")
    cv2.imwrite(output_path, cam_image)
    print(f"Saved Grad-CAM visualization to: {output_path}")