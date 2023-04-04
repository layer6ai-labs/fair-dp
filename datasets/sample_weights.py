from numpy import argmax

'''
Suppose we have data partitioned by groups, want to sample data by group_ratios such that
after sampling,
number of samples in group i / total number of samples ~= group_ratios[i] / sum(group_ratios)

Transform group_ratios -> sample_weights and then sample each group i with probability
sample_weights[i] to achieve this

all group ratios should be <= 1, -1 indicates group should be sampled with probability 1
'''


def find_restricted(group_ratios, num_samples, sample_idx):
    """
    group_ratios: -1, -1, 0.09, -1, ...
    num_samples: a list of sample counts that falls into each group
    sample_idx: a list of index that is not -1 in group_ratios
    """
    candidates = []
    for i in sample_idx:
        if all(group_ratios[j] * num_samples[i] <= num_samples[j] for j in sample_idx):
            candidates.append(i)
    restricted_index = argmax([group_ratios[i] * num_samples[i] for i in candidates])
    restricted = candidates[restricted_index]
    return restricted


def find_sample_weights(group_ratios, num_samples):
    to_sample_idx = [i for i, item in enumerate(group_ratios) if item != -1]
    if to_sample_idx == []:
        return {j: 1 for j in range(len(group_ratios))}
    restricted = find_restricted(group_ratios, num_samples, to_sample_idx)
    sample_weights = {j: group_ratios[j] * num_samples[restricted] / num_samples[j] if j in to_sample_idx else 1 for j
                      in range(len(group_ratios))}
    return sample_weights
