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
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode")
    parser.add_argument("--epoch_steps", type=int, default=None)
    parser.add_argument("--placeholder_steps", type=int, default=None)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)

    if args.seed is not None:
        cfg.EXPERIMENT.SEED = args.seed

    cfg.EXPERIMENT.EVAL = args.eval

    if args.epoch_steps is not None:
        cfg.EXPERIMENT.EPOCH_STEPS = args.epoch_steps

    if args.placeholder_steps is not None:
        cfg.EXPERIMENT.PLACEHOLDER_STEPS = args.placeholder_steps

    cfg.freeze()
    main(cfg)
