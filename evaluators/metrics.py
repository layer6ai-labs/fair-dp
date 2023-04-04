import torch
from opacus.grad_sample.grad_sample_module import GradSampleModule

from utils import split_by_group


def accuracy(model, dataloader, **kwargs):
    correct = 0
    total = 0
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    return (correct / total).item()


def accuracy_per_group(model, dataloader, num_groups=None, **kwargs):
    correct_per_group = [0] * num_groups
    total_per_group = [0] * num_groups
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)

            per_group = split_by_group(data, labels, group, num_groups)
            for i, group in enumerate(per_group):
                data_group, labels_group = group
                outputs = model(data_group)
                _, predicted = torch.max(outputs, 1)
                total_per_group[i] += labels_group.size(0)
                correct_per_group[i] += (predicted == labels_group).sum()
    return [float(correct_per_group[i] / total_per_group[i]) for i in range(num_groups)]


def macro_accuracy(model, dataloader, num_classes=None, **kwargs):
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        device = model._module.device if isinstance(model, GradSampleModule) else model.device
        for _batch_idx, (data, labels, group) in enumerate(dataloader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            for true_p, all_p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[true_p.long(), all_p.long()] += 1

    accs = confusion_matrix.diag() / confusion_matrix.sum(1)
    return accs.mean().item()
