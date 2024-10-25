from __future__ import print_function, absolute_import

import argparse
import torch

torch.backends.cudnn.benchmark = True

from scripts.utils.misc import save_checkpoint, adjust_learning_rate

import scripts.datasets as datasets
import scripts.machines as machines
from options import Options


def main(args):
    args.arch="vvv4n"
    args.schedule = 10000
    args.style_loss = 0.025
    args.ssim_loss = 0.15
    args.loss_type = "hybrid"
    args.machine = "vx"
    args.base_dir = "data/manhua_ds"
    
    val_loader = torch.utils.data.DataLoader(
        datasets.COMIC("val", args),
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    data_loaders = (None, val_loader)

    Machine = machines.__dict__[args.machine](datasets=data_loaders, args=args)

    Machine.test()


if __name__ == "__main__":
    parser = Options().init(argparse.ArgumentParser(description="WaterMark Removal"))
    main(parser.parse_args())
