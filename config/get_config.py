from .images import CFG_MAP_IMG
from .tabular import CFG_MAP_TAB

_IMAGE_DATASETS = ["mnist", "fashion-mnist", "svhn", "cifar10", "celeba"]
_TABULAR_DATASETS = ["adult", "dutch"]


def get_config(dataset, method):
    if dataset in _IMAGE_DATASETS:
        cfg_map = CFG_MAP_IMG
        print("Note: protected group set to labels")
    elif dataset in _TABULAR_DATASETS:
        cfg_map = CFG_MAP_TAB
    else:
        raise ValueError(
            f"Invalid dataset {dataset}. "
            + f"Valid choices are {_IMAGE_DATASETS + _TABULAR_DATASETS}."
        )

    base_config = cfg_map["base"](dataset)

    try:
        method_config_function = cfg_map[method]
    except KeyError:
        cfg_map.pop("base")
        raise ValueError(
            f"Invalid method {method}. "
            + f"Valid choices are {cfg_map.keys()}."
        )

    return {
        **base_config,

        "dataset": dataset,
        "method": method,

        **method_config_function(dataset)
    }
