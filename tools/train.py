import argparse
import torch.backends.cudnn as cudnn
from configs.cfg import CFG as cfg
from mitigators import method_to_trainer

cudnn.benchmark = True


def main(cfg):

    # train
    trainer = method_to_trainer[cfg.MITIGATOR.TYPE](cfg)
    if cfg.EXPERIMENT.EVAL:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("training")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    if args.seed is not None:
        cfg.EXPERIMENT.SEED = args.seed

    cfg.freeze()
    main(cfg)
