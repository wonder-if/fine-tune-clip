import os
import shutil
import yaml


def add_train_dir(cfg):

    datasets_root = cfg["datasets_root"]
    i = 1

    for dataset in cfg["datasets_meta"]:
        dataset_dir = os.path.join(datasets_root, dataset["dir"])

        for domain in dataset["domains"]:
            # 重命名原始目录为 train（先移出）
            raw_domain_dir = os.path.join(
                os.path.join(dataset_dir, domain["domain_dir"])
            )
            print(f"[开始] {raw_domain_dir}", os.path.exists(raw_domain_dir))
            if os.listdir(raw_domain_dir) == ["train"]:
                continue

            temp_train_path = os.path.join(dataset_dir, "train")

            os.rename(raw_domain_dir, temp_train_path)

            # 构造新的 domain 层级文件夹

            os.makedirs(raw_domain_dir, exist_ok=True)

            shutil.move(temp_train_path, raw_domain_dir)
            print(f"[完成] {raw_domain_dir} ")
            print(f"[完成] {i} 个数据集处理")
            i += 1
