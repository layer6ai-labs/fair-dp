def get_base_config(dataset):
    if dataset in ["mnist", "fashion-mnist", "svhn", "cifar10"]:
        delta = 1e-6
        output_dim = 10
        num_groups = 10
        protected_group = "labels"
        selected_groups = [2, 8]
    elif dataset in ["celeba"]:
        delta = 1e-6
        output_dim = 2
        num_groups = 2
        protected_group = "eyeglasses"
        selected_groups = [0, 1]
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    net_configs = {
        "net": "cnn",
        "activation": "tanh",
        "hidden_channels": [32, 16],
        "kernel_size": [3, 3, 3, 3],
        "stride": [1, 1, 1, 1],
        "output_dim": output_dim,
    }

    return {
        "protected_group": protected_group,
        "num_groups": num_groups,
        "selected_groups": selected_groups,

        "seed": 0,

        "optimizer": "sgd",
        "lr": 0.01,
        "use_lr_scheduler": False,
        "max_epochs": 60,
        "accountant": "rdp",
        "delta": delta,
        "noise_multiplier": 0.8,
        "l2_norm_clip": 1.0,

        "make_valid_loader": False,
        "train_batch_size": 256,
        "valid_batch_size": 256,
        "test_batch_size": 256,
        "group_ratios": [-1] * num_groups,

        "valid_metrics": ["accuracy", "accuracy_per_group"],
        "test_metrics": ["accuracy", "accuracy_per_group", "macro_accuracy"],
        "evaluate_angles": False,
        "evaluate_hessian": False,
        "angle_comp_step": 200,
        "num_hutchinson_estimates": 100,
        "sampled_expected_loss": False,

        **net_configs
    }


def get_non_private_config(dataset):
    return {}


def get_dpsgd_config(dataset):
    return {
        "activation": "tanh",
    }


def get_dpsgd_f_config(dataset):
    return {
        "activation": "tanh",
        "base_max_grad_norm": 1.0,  # C0
        "counts_noise_multiplier": 10.0  # noise scale applied on mk and ok
    }


def get_fairness_lens_config(dataset):
    return {
        "activation": "tanh",
        "gradient_regularizer": 1.0,
        "boundary_regularizer": 1.0
    }


def get_dpsgd_global_config(dataset):
    return {
        "activation": "tanh",
        "strict_max_grad_norm": 100,  # Z
    }


# TODO: change defaults
def get_dpsgd_global_adapt_config(dataset):
    return {
        "activation": "tanh",
        "strict_max_grad_norm": 100,  # Z
        "bits_noise_multiplier": 10.0,  # noise scale applied on average of bits
        "lr_Z": 0.1,  # learning rate with which Z^t is tuned
        "threshold": 1  # threshold in how we compare gradient norms to Z
    }


CFG_MAP_IMG = {
    "base": get_base_config,
    "regular": get_non_private_config,
    "dpsgd": get_dpsgd_config,
    "dpsgd-f": get_dpsgd_f_config,
    "fairness-lens": get_fairness_lens_config,
    "dpsgd-global": get_dpsgd_global_config,
    "dpsgd-global-adapt": get_dpsgd_global_adapt_config
}
