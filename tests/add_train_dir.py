import hydra
from omegaconf import DictConfig

from clip_tuner.data import add_train_dir


@hydra.main(
    version_base=None, config_path="pkg://clip_tuner/configs", config_name="config"
)
def main(cfg: DictConfig):
    add_train_dir(cfg.datasets_info)


if __name__ == "__main__":
    main()
