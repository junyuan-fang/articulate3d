"""Run the PL validation loop so AP metrics actually get computed.

main_instance_segmentation_articulation.py only exposes fit() and test().
runner.test() ends in test_epoch_end(mode='test'), and eval_instance_epoch_end
gates the whole AP block behind `if mode != 'test'` -- so the test path prints
losses and never computes AP, no matter what data.test_mode is set to.
validation_epoch_end is the one that calls test_epoch_end(mode='validation'),
so we need runner.validate().

Usage: same hydra overrides as scripts/infer_mov_local.sh.
"""
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from main_instance_segmentation_articulation import get_parameters


@hydra.main(
    config_path="conf", config_name="config_articulation_instance_segmentation.yaml"
)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())
    cfg, model, loggers = get_parameters(cfg)
    runner = Trainer(
        gpus=cfg.general.gpus,
        logger=loggers,
        weights_save_path=str(cfg.general.save_dir),
        **cfg.trainer,
    )
    runner.validate(model)


if __name__ == "__main__":
    main()
