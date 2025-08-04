from PIL import Image
import matplotlib.pyplot as plt


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

