import os
import cv2
import numpy as np
import torch
from PIL import Image

# 检查必要的依赖
try:
    import folder_paths
except ImportError:
    raise ImportError("未找到ComfyUI的folder_paths模块，请确保在ComfyUI环境中运行此插件")

# 定义节点类
class IsYellowishNode:
    # 定义节点在UI中的显示名称
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI的标准图像输入
                "threshold": ("FLOAT", {
                    "default": 140.0,
                    "min": 128.0,
                    "max": 160.0,
                    "step": 0.5
                }),  # 黄色判定阈值，可调节
            },
        }

    # 定义输出类型
    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("is_yellowish", "yellow_percentage", "b_channel_mean", "result_text")
    FUNCTION = "detect_yellow"
    CATEGORY = "image/analysis"  # 节点分类

    # 检测黄色的主要函数
    def detect_yellow(self, image, threshold):
        try:
            # 1. 验证输入张量
            if image is None:
                print("错误：输入图像为None")
                return (False, 0.0, 0.0, "错误：输入图像为None")

            print(f"输入图像张量形状: {image.shape}, 类型: {image.dtype}")
            if image.numel() == 0:
                print("错误：输入图像张量为空（没有元素）")
                return (False, 0.0, 0.0, "错误：输入图像张量为空")
            # 转换ComfyUI的tensor图像为OpenCV格式
            # ComfyUI中图像是RGB格式的tensor，范围0-1
            i = 255. * image.cpu().numpy().squeeze()
            img = np.clip(i, 0, 255).astype(np.uint8)

            # 如果是RGBA，转换为RGB
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]

            # RGB转BGR (OpenCV使用BGR)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 转换到LAB色彩空间
            lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

            # 提取b通道（黄-蓝轴）
            b_channel = lab_img[:, :, 2]

            # 计算b通道平均值
            b_mean = float(np.mean(b_channel))

            # 计算大于阈值的像素占比
            yellow_pixels = np.sum(b_channel > threshold)
            total_pixels = img.shape[0] * img.shape[1]
            yellow_percentage = float(yellow_pixels / total_pixels)

            # 如果b通道平均值大于阈值，判定为偏黄
            is_yellow = b_mean > threshold

            # 创建结果文本
            result_text = f"黄色分析结果:\n是否偏黄: {'是' if is_yellow else '否'}\n黄色区域占比: {yellow_percentage*100:.2f}%\nLAB b通道均值: {b_mean:.2f}"

            return (is_yellow, yellow_percentage, b_mean, result_text)
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"处理图像时发生错误:\n{traceback_str}")
            return (False, 0.0, 0.0, f"错误: {str(e)}")

# 创建可视化节点 - 生成黄色热力图
class YellowHeatmapNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {
                    "default": 140.0,
                    "min": 128.0,
                    "max": 160.0,
                    "step": 0.5
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_heatmap"
    CATEGORY = "image/analysis"

    def generate_heatmap(self, image, threshold):
        # 转换图像格式
        i = 255. * image.cpu().numpy().squeeze()
        img = np.clip(i, 0, 255).astype(np.uint8)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 转换到LAB色彩空间
        lab_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        b_channel = lab_img[:, :, 2]

        # 创建热力图
        # 归一化b通道值到0-1
        b_norm = (b_channel - 128) / (255 - 128)
        b_norm = np.clip(b_norm, 0, 1)

        # 应用颜色映射
        heatmap = cv2.applyColorMap((b_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # 创建掩码，只显示大于阈值的区域
        mask = np.zeros_like(b_channel)
        mask[b_channel > threshold] = 255

        # 将掩码应用到热力图
        masked_heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

        # 将热力图与原图混合
        alpha = 0.7
        beta = 0.3
        gamma = 0
        blended = cv2.addWeighted(img, alpha, masked_heatmap, beta, gamma)

        # 转回ComfyUI格式 (RGB tensor)
        tensor_image = torch.from_numpy(blended.astype(np.float32) / 255.0)[None, ...]

        return (tensor_image,)

# 在插件中注册节点
NODE_CLASS_MAPPINGS = {
    "IsYellowish": IsYellowishNode,
    "YellowHeatmap": YellowHeatmapNode
}

# 提供节点显示名称的映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "IsYellowish": "Detect Yellowish Image",
    "YellowHeatmap": "Generate Yellow Heatmap"
}

# 这两个变量是ComfyUI识别插件的必要内容
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']