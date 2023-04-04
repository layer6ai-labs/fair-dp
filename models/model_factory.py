import torch
import torch.nn as nn

from .neural_networks import (CNN, MLP, LogisticRegression)

activation_map = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "swish": nn.SiLU
}


def create_model(config, device):
    if config["net"] == "mlp":
        model = MLP(
            n_units_list=[config["data_dim"], *config["hidden_dims"], config["output_dim"]],
            activation=activation_map[config.get("activation", "relu")],
        )

    elif config["net"] == "cnn":
        model = CNN(
            input_channels=config["data_shape"][0],
            hidden_channels_list=config["hidden_channels"],
            output_dim=config["output_dim"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            image_height=config["data_shape"][1],
            activation=activation_map[config.get("activation", "relu")],
        )

    elif config["net"] == "logistic":
        model = LogisticRegression(
            input_dim=config["data_shape"][0],
            output_dim=config["output_dim"],
        )

    else:
        raise ValueError(f"Unknown network type {config['net']}")

    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Unknown optimizer")

    model.set_device(device)

    return model, optimizer
