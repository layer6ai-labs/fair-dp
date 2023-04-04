import math

import matplotlib.pyplot as plt
import torch
from opacus.accountants.analysis.gdp import compute_eps_poisson
from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent


def privacy_checker(sample_rate, config):
    assert sample_rate <= 1.0
    steps = config["max_epochs"] * math.ceil(1 / sample_rate)

    if config["accountant"] == 'rdp':
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        rdp = compute_rdp(
            q=sample_rate,
            noise_multiplier=config["noise_multiplier"],
            steps=steps,
            orders=orders)
        epsilon, alpha = get_privacy_spent(
            orders=orders,
            rdp=rdp,
            delta=config["delta"])
        print(
            "-----------privacy------------"
            f"\nDP-SGD (RDP) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {config['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {epsilon:.3g},"
            f"\n\tdelta = {config['delta']}."
            f"\nThe optimal alpha is {alpha}."
        )
    elif config["accountant"] == 'gdp':
        eps = compute_eps_poisson(
            steps=steps,
            noise_multiplier=config["noise_multiplier"],
            sample_rate=sample_rate,
            delta=config["delta"],
        )
        print(
            "-----------privacy------------"
            f"\nDP-SGD (GDP) with\n\tsampling rate = {100 * sample_rate:.3g}%,"
            f"\n\tnoise_multiplier = {config['noise_multiplier']},"
            f"\n\titerated over {steps} steps,\nsatisfies "
            f"differential privacy with\n\tepsilon = {eps:.3g},"
            f"\n\tdelta = {config['delta']}."
        )
    else:
        raise ValueError(f"Unknown accountant {config['accountant']}. Try 'rdp' or 'gdp'.")


def get_grads(named_parameters, group, num_groups):
    ave_grads = [[] for _ in range(num_groups)]
    max_grads = [[] for _ in range(num_groups)]
    name_grads = list(named_parameters)
    for batch_idx in range(group.shape[0]):
        grads_per_sample = []
        for layer_idx in range(len(name_grads)):
            if name_grads[layer_idx][1].requires_grad:
                grads_per_sample.append(name_grads[layer_idx][1].grad_sample[batch_idx].abs().reshape(-1))
        ave_grads[group[batch_idx]].append(torch.mean(torch.cat(grads_per_sample, 0)))
        max_grads[group[batch_idx]].append(torch.max(torch.cat(grads_per_sample, 0)))
    return ave_grads, max_grads


def get_grad_norms_clip(per_sample_grads, group, num_groups, clipping_scale_fn, **kwargs):
    grad_norms = [[] for _ in range(num_groups)]
    clip_grad_norms = [[] for _ in range(num_groups)]
    sum_grad_vec = []
    sum_clip_grad_vec = []
    for sample_idx in range(group.shape[0]):
        grad_vec = per_sample_grads[sample_idx]
        grad_norm = torch.linalg.norm(grad_vec).item()
        clipping_scale = clipping_scale_fn(grad_norm, sample_idx, **kwargs)
        clip_grad_vec = clipping_scale * grad_vec
        clip_grad_norm = torch.linalg.norm(clip_grad_vec).item()
        grad_norms[group[sample_idx]].append(grad_norm)
        clip_grad_norms[group[sample_idx]].append(clip_grad_norm)
        if sample_idx == 0:
            for _ in range(num_groups):
                sum_grad_vec.append(torch.zeros(grad_vec.shape[0], device=grad_vec.device, requires_grad=False))
                sum_clip_grad_vec.append(torch.zeros(grad_vec.shape[0], device=grad_vec.device, requires_grad=False))
        sum_grad_vec[group[sample_idx]] += grad_vec
        sum_clip_grad_vec[group[sample_idx]] += clip_grad_vec
    return grad_norms, clip_grad_norms, sum_grad_vec, sum_clip_grad_vec


def get_grad_norms(per_sample_grads, group, num_groups):
    grad_norms = [[] for _ in range(num_groups)]
    sum_grad_vec = []
    for sample_idx in range(group.shape[0]):
        grad_vec = per_sample_grads[sample_idx]
        grad_norms[group[sample_idx]].append(torch.linalg.norm(grad_vec).item())
        if sample_idx == 0:
            for _ in range(num_groups):
                sum_grad_vec.append(torch.zeros(grad_vec.shape[0], device=grad_vec.device, requires_grad=False))
        sum_grad_vec[group[sample_idx]] += grad_vec
    return grad_norms, sum_grad_vec


def get_num_layers(model):
    num_layers = 0
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            num_layers += 1

    return num_layers


# splits data, labels according to group of data point
# returns tensor of size num_groups, each element is (subset of data, subset of labels)  
# corresponding to specific group given by index
def split_by_group(data, labels, group, num_groups, return_counts=False):
    sorter = torch.argsort(group)
    unique, counts = torch.unique(group, return_counts=True)
    unique = unique.tolist()
    counts = counts.tolist()

    complete_unique = [0] * num_groups
    complete_counts = [0] * num_groups
    for i in range(num_groups):
        complete_unique[i] = i
        if i in unique:
            j = unique.index(i)
            complete_counts[i] = counts[j]

    sorted_data = torch.split(data[sorter], complete_counts)
    sorted_labels = torch.split(labels[sorter], complete_counts)

    if not return_counts:
        return list(zip(sorted_data, sorted_labels))
    return list(zip(sorted_data, sorted_labels)), complete_counts


def plot_by_group(data_by_group, writer, data_title=None, scale_to_01=False):
    fig = plt.figure()
    plt.bar(range(len(data_by_group)), data_by_group, width=0.9)
    plt.xlabel("Groups")
    if data_title is not None:
        plt.ylabel(data_title)
    plt.title(data_title)
    plt.xticks(range(len(data_by_group)))
    if scale_to_01:
        plt.ylim(0, 1)
    writer.write_figure(data_title, fig)


def unflatten_grads(params, grad_vec):
    grad_size = []
    for layer in params:
        grad_size.append(layer.reshape(-1).shape[0])
    grad_list = list(torch.split(grad_vec, grad_size))
    for layer_idx in range(len(grad_list)):
        grad_list[layer_idx] = grad_list[layer_idx].reshape(params[layer_idx].shape)
    return tuple(grad_list)


def rademacher(tens):
    """Draws a random tensor of size [tens.shape] from the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty_like(tens)
    x.random_(0, 2)
    x[x == 0] = -1
    return x
