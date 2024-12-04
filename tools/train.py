import argparse
import torch.backends.cudnn as cudnn
from configs.cfg import CFG as cfg
from mitigators import method_to_trainer

cudnn.benchmark = True


def main(cfg):

    # train
    trainer = method_to_trainer[cfg.MITIGATOR.TYPE](cfg)
    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("training")
    parser.add_argument("--cfg", type=str, default="")

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    cfg.freeze()
    main(cfg)
