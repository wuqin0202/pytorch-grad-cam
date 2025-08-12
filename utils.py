from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2


def overlay_similarity_heatmap(sim_norm: np.ndarray,
                               rgb: np.ndarray,
                               alpha_img: float = 0.4,
                               alpha_map: float = 0.6,
                               colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    参数:
        sim_norm: (H, W) 的相似度归一化图, 建议已在 [0, 1]，若越界会被裁剪
        rgb: (H, W, 3) 的 RGB 图 (np.uint8 或 float)
        alpha_img: 原图权重
        alpha_map: 热图权重
        colormap: OpenCV 伪彩色映射（默认 JET）
    返回:
        叠加后的 RGB 图 (uint8)
    """
    assert sim_norm.ndim == 2, "sim_norm 必须是二维(H, W)"
    assert rgb.ndim == 3 and rgb.shape[2] == 3, "rgb 必须是(H, W, 3)"

    H, W = rgb.shape[:2]

    # 处理 NaN/Inf 并裁剪到 [0,1]
    sim = np.nan_to_num(sim_norm, nan=0.0, posinf=1.0, neginf=0.0)
    sim = np.clip(sim, 0.0, 1.0)

    # 尺寸对齐（双线性插值）
    if sim.shape != (H, W):
        sim_resized = cv2.resize(sim, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        sim_resized = sim

    # 转为 0-255 的 uint8 单通道
    sim_u8 = (sim_resized * 255.0).astype(np.uint8)

    # 生成 BGR 伪彩色热图
    heatmap_bgr = cv2.applyColorMap(sim_u8, colormap)

    # 规范化 rgb 到 uint8，并从 RGB->BGR
    if np.issubdtype(rgb.dtype, np.floating):
        # 自动判断浮点范围：若最大值<=1，则按[0,1]处理，否则假定已是[0,255]
        scale = 255.0 if rgb.max() <= 1.0 else 1.0
        rgb_u8 = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb.astype(np.uint8)

    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    # 融合（等价于 cv2_img * 0.4 + vis * 0.6）
    blended_bgr = cv2.addWeighted(bgr, alpha_img, heatmap_bgr, alpha_map, 0.0)

    # 转回 RGB 输出
    blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
    return blended_rgb


def show_tensor_image(image_tensor):
    """
    Display a single image tensor using matplotlib.

    Args:
        image_tensor (torch.Tensor): Image tensor of shape (C, H, W) or (1, C, H, W)
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove batch dimension if present
    image_np = image_tensor.permute(1, 2, 0).numpy()  # Convert to HWC format
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()


def split_to_grid(tensor, grid_size=8):
    """
    将 CHW 张量分割成 grid_size 的网格
    如果不能整除grid_size，则中心裁剪能整除grid_size的最大部分

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (C, H, W)
        grid_size (int, tuple): 网格大小，默认为8，若为元组则表示 (h, w)

    返回:
        list: 包含64个块的列表，每个块为 (C, h, w)
    """
    C, H, W = tensor.shape
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    # 计算能整除grid_size的最大尺寸
    new_H = (H // grid_size[0]) * grid_size[0]
    new_W = (W // grid_size[1]) * grid_size[1]

    # 中心裁剪
    if new_H != H or new_W != W:
        start_H = (H - new_H) // 2
        start_W = (W - new_W) // 2
        tensor = tensor[:, start_H:start_H+new_H, start_W:start_W+new_W]

    # 分割成grid_size网格
    h, w = new_H // grid_size[0], new_W // grid_size[1]
    grid = tensor.unfold(1, h, h).unfold(2, w, w)
    grid = grid.reshape(C, grid_size[0], grid_size[1], h, w).permute(1, 2, 3, 4, 0)

    return grid


def show_grid(grid, sims=None, fig_size=8, mask=None):
    """
    显示网格的小块

    参数:
        grid (torch.Tensor): 包含64个小块的列表，每个小块形状为 (H//p, W//p, p, p, C)
        sims (torch.Tensor, optional): 相似度矩阵，形状为 (H//p, W//p)，用于显示相似度值
        fig_size (int): plt fig大小，默认为8
        mask (torch.Tensor, optional): 掩码，形状为 (H//p, W//p)，用于选择性显示块
    """
    h, w, p_h, p_w = grid.shape[:-1]
    print(f"Grid shape: {len(grid)} blocks of size {p_h}x{p_w}")
    fig, axes = plt.subplots(h, w, figsize=(fig_size, fig_size/(w * p_w)*p_h*h))
    for i in range(h):
        for j in range(w):
            axes[i, j].axis('off')
            if mask is not None and not mask[i, j]:
                continue
            axes[i, j].imshow(grid[i, j].numpy())
            if sims is not None:
                sim_value = sims[i, j].item()
                axes[i, j].text(
                    p_w // 2, p_h // 2,  # 中心坐标
                    f"{sim_value:.2f}",  # 显示2位小数
                    color='white',  # 文字颜色
                    ha='center',    # 水平居中
                    va='center',    # 垂直居中
                    fontsize=8,     # 字体大小
                    bbox=dict(facecolor='black', alpha=0.5, boxstyle='round')  # 背景框
                )
    plt.subplots_adjust(
        left=0.,    # 左边距
        right=1.,   # 右边距
        bottom=0.,  # 底部边距
        top=1.,     # 顶部边距
        wspace=0.05,  # 水平间距（子图之间的宽度间隔）
        hspace=0.05   # 垂直间距（子图之间的高度间隔）
    )
    plt.show()


def tensor2pil(tensor):
    """
    将张量转换为PIL图像

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (C, H, W)

    返回:
        PIL.Image: 转换后的PIL图像
    """
    # 确保张量在CPU上并转换为numpy数组
    image_np = tensor.cpu().numpy().transpose(1, 2, 0)  # 从 CHW 转换为 HWC
    return Image.fromarray((image_np * 255).astype('uint8'))  # 假设张量值在[0, 1]范围内

