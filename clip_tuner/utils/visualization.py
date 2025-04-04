import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


def visualize_dataset_samples(dataset, num_samples=5):
    # 创建一个数据加载器
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    # 获取一批数据

    example_batch = next(iter(dataloader))
    images = example_batch["pixel_values"]
    labels = example_batch["label"]
    # 可视化图像
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        image = images[i].numpy().transpose((1, 2, 0))  # 转换为 HWC 格式
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean  # 反归一化
        image = np.clip(image, 0, 1)  # 限制像素值在 [0, 1]
        plt.imshow(image)

        plt.title(
            f"{labels[i].item()}: {dataset.features["label"].int2str(labels[i].item())}"
        )
        plt.tight_layout()
        plt.axis("off")
    # plt.savefig("./logs/figs/plt.png")
    plt.show()
