import argparse
import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

import clip


def get_args():
    parser = argparse.ArgumentParser(description="CLIP Surgery heatmap demo")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use (e.g., cuda, cuda:0, cpu)"
    )
    parser.add_argument(
        "--image-path", type=str, default="./examples/both.png",
        help="Input image path"
    )
    parser.add_argument(
        "--labels", type=str, nargs='+', default=["table"],
        help="Query texts (one or more)"
    )
    parser.add_argument(
        "--model-name", type=str, default="CS-ViT-B/16",
        help="Model name from clip.available_models()"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./output/clip_surgery",
        help="Directory to save heatmaps"
    )
    parser.add_argument(
        "--meta-tokenizer-path", type=str, default="/data25/wuqin/.cache/huggingface/hub/models--facebook--xlm-v-base/snapshots/68c75dd7733d2640b3a98114e3e94196dc543fe1",
        help="Local path to a HF tokenizer (used when model name contains 'Meta')"
    )
    parser.add_argument(
        "--no-empty-redundant", action="store_true",
        help="If set, do not subtract redundant features encoded from empty string"
    )
    return parser.parse_args()


def get_tokenizer_from_hf(path: str):
    from transformers import AutoTokenizer
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(path or "xlm-roberta-base", use_fast=True)

    def tokenize_fn(txt_list: List[str]):
        input_ids = tokenizer(
            txt_list, return_tensors='pt', padding='max_length', truncation=True, max_length=77
        )['input_ids']
        return input_ids

    return tokenize_fn


@torch.no_grad()
def run_clip_surgery(
    model_name: str,
    device: str,
    image_path: str,
    labels: List[str],
    out_dir: str,
    meta_tokenizer_path: Optional[str] = None,
    use_empty_redundant: bool = True,
):
    os.makedirs(out_dir, exist_ok=True)

    model, preprocess = clip.load(model_name, device=device, download_root='/data25/wuqin/ckpt/metaclip')
    model.eval()

    is_meta_clip = 'Meta' in model_name
    prompt_templates = ['{}'] if is_meta_clip else None
    tokenize = get_tokenizer_from_hf(meta_tokenizer_path) if is_meta_clip else clip.tokenize

    pil_img = Image.open(image_path).convert('RGB')
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image = preprocess(pil_img).unsqueeze(0).to(device)

    # Encode image tokens and normalize per-token
    image_features = model.encode_image(image)
    image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-6)

    # Text features with prompt ensemble
    text_features = clip.encode_text_with_prompt_ensemble(
        model, labels, device, tokenize=tokenize, prompt_templates=prompt_templates
    )

    redundant_features = None
    if use_empty_redundant:
        redundant_features = clip.encode_text_with_prompt_ensemble(
            model, [""], device, tokenize=tokenize, prompt_templates=prompt_templates
        )

    # Feature surgery
    similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
    # Drop CLS token (index 0), keep patch tokens only
    similarity_map = clip.get_similarity_map(similarity[:, 1:, :], cv2_img.shape[:2])

    # Save heatmaps using cv2.imwrite
    base = os.path.splitext(os.path.basename(image_path))[0]
    for b in range(similarity_map.shape[0]):
        for n, label in enumerate(labels):
            sm = (similarity_map[b, :, :, n].detach().cpu().numpy() * 255).astype('uint8')
            heat = cv2.applyColorMap(sm, cv2.COLORMAP_JET)
            vis = cv2.addWeighted(cv2_img, 0.4, heat, 0.6, 0)
            out_path = os.path.join(out_dir, f"{base}_clip_surgery_{n}_{label}.jpg")
            cv2.imwrite(out_path, vis)


def main():
    args = get_args()
    run_clip_surgery(
        model_name=args.model_name,
        device=args.device,
        image_path=args.image_path,
        labels=args.labels,
        out_dir=args.out_dir,
        meta_tokenizer_path=args.meta_tokenizer_path,
        use_empty_redundant=not args.no_empty_redundant,
    )


if __name__ == "__main__":
    main()
