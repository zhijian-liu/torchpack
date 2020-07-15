import torch

__all__ = ['batched_index_select']


def batched_index_select(inputs: torch.Tensor, indices: torch.Tensor,
                         dim: int) -> torch.Tensor:
    vsizes, esizes = [], []
    for k, size in enumerate(inputs.shape):
        if k == 0:
            vsizes.append(size)
            esizes.append(-1)
        elif k == dim:
            vsizes.append(-1)
            esizes.append(-1)
        else:
            vsizes.append(1)
            esizes.append(size)

    indices = indices.view(vsizes).expand(esizes)
    outputs = torch.gather(inputs, dim, indices)
    return outputs
