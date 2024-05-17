import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import Dataset, DataLoader, default_collate


class SubsequenceDataset(Dataset):
    r"""A dataset returning sub-sequences extracted from longer sequences.
    Args:
        *tensors (Tensor): tensors that have the same size on the first dimension.
    Examples:
        >>> u = torch.randn(1000, 2) # 2 inputs
        >>> y = torch.randn(1000, 3) # 3 outputs
        >>> train_dataset = SubsequenceDataset(u, y, subseq_len=100)
    """

    def __init__(self, *tensors, subseq_len, stride=1):
        self.tensors = tensors

        self.subseq_len = subseq_len
        self.length = self.tensors[0].shape[0]
        self.stride = stride

    def __len__(self):
        return int((self.length - self.subseq_len + 1)//self.stride)

    def __getitem__(self, idx):
        subsequences = [tensor[self.stride*idx:self.stride*idx+self.subseq_len] for tensor in self.tensors]
        return subsequences


def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

class NumpyLoader(DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)