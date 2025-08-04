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

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

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
            # default: Load the model on the available device(s)
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
        # probs = torch.softmax(logits/0.7, dim=-1)
        probs = logits
        return probs.float()


H, W = None, None
def reshape_transform(tensor):
    """Reshape transform for Vision Transformer features"""
    global H, W
    p_h, p_w = (H//28), (W//28)
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0) # model.model.model.visual.blocks[-1]
    if tensor.shape[1] > p_h * p_w: # 没有 merge
        p_h, p_w = p_h*2, p_w*2

    print(tensor.shape)
    result = tensor.reshape(tensor.size(0), p_h, p_w, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """
    Example usage of CAM methods on Visualized BGE model.
    python visualbge_example.py --image-path <path_to_image> --text-queries "a cat" "a dog"
    """

    args = get_args()
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    # Initialize model
    model = ImageClassifier()

    # 仅限 visual 相关层，该层输出必须为 Tensor 类型
    # target_layers = [model.model.model.visual.blocks[-1]] # merge 前
    target_layers = [model.model.model.visual.merger] # merge 后

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    # 设置文本查询
    model.set_text_query(args.image_path, args.text_query)
    print(f"文本查询已设置: {args.text_query}")

    # 加载和预处理图片
    img_tensor = load_and_preprocess_images([args.image_path])
    H, W = img_tensor.shape[2:]
    print(f"图片尺寸 H×W: {H}×{W}")

    # Initialize CAM
    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform,
                                   ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    # Set targets (None for highest scoring category)
    targets = None

    # Set batch size for efficiency
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=img_tensor,
                        targets=targets,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Extract single image from batch
    grayscale_cam = grayscale_cam[0, :]

    # Create visualization
    cam_image = show_cam_on_image(img_tensor[0].permute(1, 2, 0).cpu().numpy(), grayscale_cam)

    # Save result
    output_path = f'qwen2_5vl_{args.method}_cam.jpg'
    cv2.imwrite(output_path, cam_image)
    print(f"CAM visualization saved to {output_path}")
