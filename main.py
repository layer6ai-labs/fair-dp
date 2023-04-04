import argparse
import pprint
import random
import sys

import numpy as np
import torch
from opacus import PrivacyEngine

from config import get_config, parse_config_arg
from datasets import get_loaders_from_config
from evaluators import create_evaluator
from models import create_model
from privacy_engines.dpsgd_f_engine import DPSGDF_PrivacyEngine
from privacy_engines.dpsgd_global_adaptive_engine import DPSGDGlobalAdaptivePrivacyEngine
from privacy_engines.dpsgd_global_engine import DPSGDGlobalPrivacyEngine
from trainers import create_trainer
from utils import privacy_checker
from writer import Writer


def main():
    parser = argparse.ArgumentParser(description="Fairness for DP-SGD")

    parser.add_argument("--dataset", type=str, default="mnist",
                        help="Dataset to train on.")
    parser.add_argument("--method", type=str, default="regular",
                        choices=["regular", "dpsgd", "dpsgd-f", "fairness-lens", "dpsgd-global", "dpsgd-global-adapt"],
                        help="Method for training and clipping.")

    parser.add_argument("--config", default=[], action="append",
                        help="Override config entries. Specify as `key=value`.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = get_config(
        dataset=args.dataset,
        method=args.method,
    )
    cfg = {**cfg, **dict(parse_config_arg(kv) for kv in args.config)}

    # Checks group_ratios is specified correctly
    if len(cfg["group_ratios"]) != cfg["num_groups"]:
        raise ValueError(
            "Number of group ratios, {}, not equal to number of groups of {}, {}"
                .format(len(cfg["group_ratios"]), cfg["protected_group"], cfg["num_groups"])
        )

    if any(x > 1 or (x < 0 and x != -1) for x in cfg["group_ratios"]):
        raise ValueError("All elements of group_ratios must be in [0,1]. Indicate no sampling with -1.")

    pprint.sorted = lambda x, key=None: x
    pp = pprint.PrettyPrinter(indent=4)
    print(10 * "-" + "-cfg--" + 10 * "-")
    pp.pprint(cfg)

    # Set random seeds based on config
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    train_loader, valid_loader, test_loader = get_loaders_from_config(
        cfg,
        device
    )

    writer = Writer(
        logdir=cfg.get("logdir_root", "runs"),
        make_subdir=True,
        tag_group=args.dataset,
        dir_name=cfg.get("logdir", "")
    )
    writer.write_json(tag="config", data=cfg)

    model, optimizer = create_model(cfg, device)

    if cfg["method"] != "regular":
        sample_rate = 1 / len(train_loader)
        privacy_checker(sample_rate, cfg)

    if cfg["method"] == "dpsgd":
        privacy_engine = PrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],
            max_grad_norm=cfg["l2_norm_clip"]  # C
        )
    elif cfg["method"] == "dpsgd-global":
        privacy_engine = DPSGDGlobalPrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
        )
    elif cfg["method"] == "dpsgd-f":
        privacy_engine = DPSGDF_PrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],
            max_grad_norm=0  # this parameter is not applicable for DPSGD-F
        )

    elif cfg["method"] == "dpsgd-global-adapt":
        privacy_engine = DPSGDGlobalAdaptivePrivacyEngine(accountant=cfg["accountant"])
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=cfg["noise_multiplier"],  # sigma in sigma * C
            max_grad_norm=cfg["l2_norm_clip"],  # C
        )
    else:
        # doing regular training
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0,
            max_grad_norm=sys.float_info.max,
            poisson_sampling=False
        )

    evaluator = create_evaluator(
        model,
        valid_loader=valid_loader, test_loader=test_loader,
        valid_metrics=cfg["valid_metrics"],
        test_metrics=cfg["test_metrics"],
        num_classes=cfg["output_dim"],
        num_groups=cfg["num_groups"],
    )

    trainer = create_trainer(
        train_loader,
        valid_loader,
        test_loader,
        model,
        optimizer,
        privacy_engine,
        evaluator,
        writer,
        device,
        cfg
    )

    trainer.train()


if __name__ == "__main__":
    main()
