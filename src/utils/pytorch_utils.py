import torch


def dict_iter2tensor(dict_of_iterable):
    return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in dict_of_iterable.items()}
