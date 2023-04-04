from typing import Any, Tuple

import torch


class GroupLabelDataset(torch.utils.data.Dataset):
    ''' 
    Implementation of torch Dataset that returns features 'x', classification labels 'y', and protected group labels 'z'
    '''

    def __init__(self, role, x, y=None, z=None):
        if y is None:
            y = torch.zeros(x.shape[0]).long()

        if z is None:
            z = torch.zeros(x.shape[0]).long()

        assert x.shape[0] == y.shape[0] and x.shape[0] == z.shape[0]
        assert role in ["train", "valid", "test"]

        self.role = role

        self.x = x
        self.y = y
        self.z = z

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        return self.x[index], self.y[index], self.z[index]

    def to(self, device):
        return GroupLabelDataset(
            self.role,
            self.x.to(device),
            self.y.to(device),
            self.z.to(device),
        )
