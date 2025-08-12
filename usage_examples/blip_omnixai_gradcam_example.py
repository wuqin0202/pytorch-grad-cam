# transformers==4.26.1
import argparse
import os
import torch
import numpy as np
from PIL import Image as PilImage
from omnixai.data.text import Text
from omnixai.data.image import Image
from omnixai.data.multi_inputs import MultiInputs
from omnixai.preprocessing.image import Resize
from omnixai.explainers.vision_language.specific.gradcam import GradCAM

from lavis.models import BlipITM
from lavis.processors import load_processor
from lavis.models import load_model
from lavis.models import load_model_and_preprocess

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from omnixai.explainers.vision_language.specific.gradcam.pytorch.gradcam import Base

device = "cuda"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device to use')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/girl_dog_1.jpg',
        help='Input image path')
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        default=["A girl playing with her dog on the beach"],
        help='need recognition labels'
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
    parser.add_argument('--output-dir', type=str, default='output/blip_gradcam',
                        help='Output directory to save the images')
    args = parser.parse_args()
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')
    return args

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load image and text
    image = Resize(size=480).transform(
        Image(PilImage.open(args.image_path).convert("RGB")))
    text = Text(args.labels[0])
    inputs = MultiInputs(image=image, text=text)

    # Load model
    model = load_model(
        name="blip_image_text_matching",
        model_type="base",
        is_eval=True,
        device=device,
        checkpoint="/data25/wuqin/Downloads/pytorch-grad-cam/model_base_retrieval_coco.pth",
    )
    print(model)
    print("======" * 20)

    # Load processors
    image_processor = load_processor("blip_image_eval").build(image_size=384)
    text_processor = load_processor("blip_caption")
    tokenizer = BlipITM.init_tokenizer()

    # Print model structure to find target layer
    for name, module in model.named_modules():
        if 'crossattention' in name.lower():
            print(f"{name}: {type(module)}")
            for attr_name in dir(module):
                if not attr_name.startswith('_'):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, 'weight') or hasattr(attr, 'bias'):
                        print(f"  - {attr_name}: {type(attr)}")

    # Define preprocess function
    def preprocess(x: MultiInputs):
        images = torch.stack([image_processor(z.to_pil()) for z in x.image])
        images = images.to(device, non_blocking=True)
        texts = [text_processor(z) for z in x.text.values]
        return {"image": images, "text_input": texts}

    # Select target layer
    target_layer = model.text_encoder.encoder.layer[6].crossattention.self.dropout

    # Initialize GradCAM explainer
    explainer = GradCAM(
        model=model,
        target_layer=target_layer,
        preprocess_function=preprocess,
        tokenizer=tokenizer,
        loss_function=lambda outputs: outputs[:, 1].sum(),
        patch_shape=(24, 24)
    )

    # Generate explanations
    explanations = explainer.explain(inputs)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    # Create a descriptive filename based on labels and method
    labels_str = "_".join([label.replace(" ", "_") for label in args.labels])
    save_path = os.path.join(args.output_dir, f"{args.method}_blip_gradcam_{labels_str}.png")

    from utils import overlay_similarity_heatmap
    import cv2
    heatmap = overlay_similarity_heatmap(explanations.explanations[0]['scores'][0], explanations.explanations[0]['image'])
    cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    figures = explanations.plot()
    for i, fig in enumerate(figures):
        fig_path = save_path.replace('.png', f'_{i}.png')
        fig.savefig(fig_path)
        print(f"图形 {i} 已保存到: {fig_path}")

if __name__ == "__main__":
    main()