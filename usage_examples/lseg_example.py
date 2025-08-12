import argparse
import os
import sys

import logging
from typing import Optional, Tuple
from typing import Dict, Tuple, Any, Optional, List
import os
os.environ['HOME'] = '/data25/wuqin'

import numpy as np
import torch

import clip


logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
import numpy as np

# 延后导入与可选依赖
import torch
import cv2
import clip  # 本仓库的 CLIP_Surgery 扩展模块
import torchvision.transforms as T
from vlmaps.lseg.modules.models.lseg_net import LSegEncNet
from vlmaps.utils.lseg_utils import get_lseg_feat

import cv2
import numpy as np
import torch

from utils import overlay_similarity_heatmap


def get_args():
    parser = argparse.ArgumentParser(description="LSeg text-to-pixel similarity heatmap demo")
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (e.g., cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--image-path", type=str, default="./examples/both.png",
        help="Input image path",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=["plant"],
        help="Query texts (one or more)",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default='/data25/wuqin/projects/3d_rec/demo_e200.ckpt',
        help="Path to LSeg checkpoint (as used in lseg.ipynb)",
    )
    parser.add_argument(
        "--clip-model-name", type=str, default="ViT-B/32",
        help="ClipTextEncoder model name",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./output/lseg",
        help="Directory to save heatmaps",
    )
    return parser.parse_args()

class ClipTextEncoder:
    def __init__(self, device: Optional[str] = None, model_name: str = 'CS-ViT-B/16') -> None:
        if device in (None, 'auto'):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self._model = None
        self._model_name = model_name
        if 'Meta' in model_name:
            def get_tokenizer(tokenizer: str):
                from transformers import AutoTokenizer
                os.environ['TOKENIZERS_PARALLELISM'] = 'false'
                t = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
                def tokenize_fn(txt_list):
                    return t(txt_list, return_tensors='pt', padding='max_length', truncation=True, max_length=77)['input_ids']
                return tokenize_fn
            self.tokenize = get_tokenizer('/data25/wuqin/.cache/huggingface/hub/models--facebook--xlm-v-base/snapshots/68c75dd7733d2640b3a98114e3e94196dc543fe1')
        else:
            self.tokenize = clip.tokenize

    @property
    def model(self):
        if self._model is None:
            logger.info('Loading CLIP_Surgery model: %s', self._model_name)
            m, _ = clip.load(self._model_name, device=self.device)
            self._model = m.eval()
        return self._model

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        feat = clip.encode_text_with_prompt_ensemble(self.model, [text], self.device, tokenize=self.tokenize, prompt_templates=['{}'])[0]
        return feat / feat.norm(dim=-1, keepdim=True)

class LSegExtractor:
    """LSeg 像素级特征提取器。

    - 输出形状与图像一致的像素特征 (1, C, H, W)，本封装在 from_rgb 中返回 dict 以与上游对齐。
    - 文本特征仍使用 CLIP 文本编码器（查询阶段处理）。
    - crop_size/base_size 默认按输入图像长边设置，避免裁剪，只做必要的 padding。
    """

    def __init__(self, ckpt_path: str, device: Optional[str] = None,
                 norm_mean: list[float] | None = None,
                 norm_std: list[float] | None = None) -> None:
        if device is None or device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.ckpt_path = ckpt_path
        self.target_long_edge = 512  # 推理前将图像长边缩放到该尺寸
        self.norm_mean = norm_mean or [0.5, 0.5, 0.5]
        self.norm_std = norm_std or [0.5, 0.5, 0.5]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.norm_mean, self.norm_std),
        ])

        # 直接以固定 512 初始化模型（无需按输入尺寸重建）
        long_edge = int(self.target_long_edge)
        logger.info('Initializing LSeg model (crop_size=%d)', long_edge)
        model = LSegEncNet('', arch_option=0, block_depth=0, activation='lrelu', crop_size=long_edge)
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        # 兼容带前缀的 key
        state_dict = {k.lstrip('net.'): v for k, v in state.get('state_dict', state).items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.model = model.to(self.device)
        self.crop_size = long_edge
        self.base_size = long_edge
        self._feat_dim_cache: int | None = None

    @torch.no_grad()
    def extract_patch_features_from_rgb(self, rgb: np.ndarray) -> Dict[str, Any]:
        """从 RGB 数组提取像素特征。

        Args:
            rgb: (H, W, 3) RGB uint8

        Returns:
            - features: np.ndarray, (1, C, H, W)
            - grid_hw: Tuple[int, int] = (H, W) 用于与上游接口对齐
        """
        assert rgb.ndim == 3 and rgb.shape[2] == 3, 'rgb must be HxWx3'
        h, w = rgb.shape[:2]

        # 1) 将输入图像按长边等比缩放到 target_long_edge
        long_edge = max(h, w)
        scale = float(self.target_long_edge) / float(long_edge)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        if new_h != h or new_w != w:
            resized_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized_rgb = rgb

        outputs = get_lseg_feat(
            self.model,
            resized_rgb,
            labels=['example'],
            transform=self.transform,
            device=self.device,
            crop_size=self.crop_size,
            base_size=self.base_size,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            vis=False,
        )  # (1, C, H, W) numpy float32

        # 2) 将特征双线性插值回原图大小，保证与原像素坐标对齐
        if outputs.shape[2] != h or outputs.shape[3] != w:
            feat_t = torch.from_numpy(outputs)  # (1, C, Hr, Wr)
            feat_t = torch.nn.functional.interpolate(
                feat_t, size=(h, w), mode='bilinear', align_corners=False
            )
            outputs = feat_t.numpy()
        outputs = torch.from_numpy(outputs).to(self.device)  # (1, C, H, W)

        if self._feat_dim_cache is None:
            self._feat_dim_cache = int(outputs.shape[1])
            logger.info('LSeg feature dim: %d', self._feat_dim_cache)

        return {
            'features': outputs/outputs.norm(dim=1, keepdim=True),       # (1, C, H, W) numpy
            'grid_hw': outputs.shape[2:],
        }

@torch.no_grad()
def run_lseg(
    device: str,
    image_path: str,
    labels: List[str],
    out_dir: str,
    ckpt_path: str,
    clip_model_name: str,
):
    if isinstance(labels, str):
        labels = [labels]
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"Failed to read image: {image_path}")
        sys.exit(1)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if not ckpt_path:
        print("Warning: --ckpt-path is empty. LSegExtractor may fail without a valid checkpoint.")

    lseg = LSegExtractor(ckpt_path=ckpt_path, device=str(device))
    txt_enc = ClipTextEncoder(model_name=clip_model_name, device=str(device))

    feat_dict = lseg.extract_patch_features_from_rgb(rgb)
    if not isinstance(feat_dict, dict) or "features" not in feat_dict:
        print("Unexpected LSeg feature dict. Expected key 'features'.")
        sys.exit(1)

    pix_feats = feat_dict["features"]
    if isinstance(pix_feats, np.ndarray):
        pix_feats = torch.from_numpy(pix_feats)
    pix_feats = pix_feats.to(device).float()  # (1, C, H, W)

    # Encode texts -> (T, D)
    text_embs = []
    for t in labels:
        emb = txt_enc.encode_text(t)
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        text_embs.append(emb.float().to(device))
    text_feats = torch.stack(text_embs, dim=0)

    # Validate shapes
    if pix_feats.dim() != 4:
        print(f"Unexpected feature shape from LSeg: {tuple(pix_feats.shape)} (expected 4D)")
        sys.exit(1)
    T, D = text_feats.shape
    _, C, H, W = pix_feats.shape
    if D != C:
        print(f"Dimension mismatch: text D={D} vs pixel C={C}. Ensure encoders are aligned.")
        sys.exit(1)

    # Cosine similarity per pixel
    pix_unit = pix_feats / (torch.linalg.vector_norm(pix_feats, dim=1, keepdim=True).clamp_min(1e-6))
    txt_unit = text_feats / (torch.linalg.vector_norm(text_feats, dim=1, keepdim=True).clamp_min(1e-6))
    sims = torch.einsum("tc,bchw->thw", txt_unit, pix_unit)  # (T, H, W)
    sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-6)

    base = os.path.splitext(os.path.basename(image_path))[0]
    for i, label in enumerate(labels):
        sim_i = sims[i].detach().cpu().numpy()
        blended = overlay_similarity_heatmap(sim_i, rgb)
        out_path = os.path.join(out_dir, f"{base}_lseg_{labels[0]}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")


def main():
    args = get_args()
    # align with clip_surgery style main wrapper
    run_lseg(
        device=args.device,
        image_path=args.image_path,
        labels=args.labels,
        out_dir=args.out_dir,
        ckpt_path=args.ckpt_path,
        clip_model_name=args.clip_model_name,
    )


if __name__ == "__main__":
    main()
