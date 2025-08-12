import argparse, os, cv2, numpy as np, torch
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from src.mini_clip.factory import create_model_and_transforms, get_tokenizer   # 官方实现
os.environ['HOME'] = '/data25/wuqin'

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ---------- 命令行 ----------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--image-path', default='./examples/both.png')
    p.add_argument('--labels', nargs='+', default=["a cat", "a dog", "a car", "a person", "a shoe", "green plant"])
    p.add_argument('--method', default='gradcam')
    p.add_argument('--output-dir', default='./output/metaclip2')
    return p.parse_args()

# ---------- 包装模型 ----------
class MetaCLIPModelWrapper(torch.nn.Module):
    def __init__(self, model, labels, tokenizer, device):
        super().__init__()
        self.model = model.eval()
        self.labels = labels
        self.tokenizer = tokenizer
        self.device = device
        with torch.no_grad():
            text = tokenizer(labels).to(device)
            text_features = model.encode_text(text)
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # ← 非 in-place
        logits = 100.0 * image_features @ self.text_features.T
        print("Logits shape:", logits.shape)  # 打印 logits 的形状
        return logits.softmax(dim=-1)

# ---------- reshape_transform ----------
def reshape_transform(tensor, height=16, width=16):
    # 如果是 tuple，取第一个元素
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    # 检查是否是 3 维张量
    if tensor.dim() == 3:
        # 假设 tensor 的形状是 [seq_length, 1, channels]
        seq_length, _, channels = tensor.shape

        # 计算高度和宽度
        if height * width != seq_length - 1:
            raise ValueError(f"Sequence length {seq_length} does not match height*width {height}*{width}")

        # 去掉第一个 token（例如 CLS token）
        tensor = tensor[1:, :, :]

        # 转换为 [1, channels, height, width]
        tensor = tensor.reshape(1, height, width, channels).permute(0, 3, 1, 2)

    return tensor

# ---------- 主程序 ----------
if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device)

    # 1. 创建官方模型
    model, _, preprocess = create_model_and_transforms(
        'ViT-H-14-quickgelu-worldwide@WorldWideCLIP',
        pretrained='metaclip2_worldwide'
    )
    tokenizer = get_tokenizer("facebook/xlm-v-base")
    model = model.to(device)
    print(model)
    print("=====" * 20)

    # 2. Grad-CAM 包装
    wrapped = MetaCLIPModelWrapper(model, args.labels, tokenizer, device)

    # 3. 准备图像
    rgb_img = cv2.imread(args.image_path)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224)).astype(np.float32) / 255
    tensor = preprocess(Image.fromarray((rgb_img * 255).astype(np.uint8))).unsqueeze(0).to(device)
    print("Input tensor shape:", tensor.shape)

    # 4. CAM
    # 推荐 hook transformer block 输出
    target_layer = model.visual.transformer.resblocks[-1].ln_1
    cam_methods = {
        "gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "scorecam": ScoreCAM,
        "ablationcam": AblationCAM, "xgradcam": XGradCAM, "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM, "layercam": LayerCAM, "fullgrad": FullGrad
    }
    cam = cam_methods[args.method](
        model=wrapped,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)[0]
    print("Grayscale CAM shape:", grayscale_cam.shape)

    # 5. 保存
    os.makedirs(args.output_dir, exist_ok=True)
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 创建一个描述性的文件名，包含标签信息
    labels_str = "_".join([label.replace(" ", "_") for label in args.labels])
    save_path = os.path.join(args.output_dir, f"{args.method}_metaclip2_{labels_str}.jpg")
    cv2.imwrite(save_path, cam_img[:, :, ::-1])
    print("Saved:", save_path)
