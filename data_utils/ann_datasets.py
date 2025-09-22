import json

import accelerate
import numpy as np
import scipy
import torch

from .tokenization import GeneTokenizer
from polygene.configs import BaseConfig
from anndata import read_h5ad


class IterableAnnDataset(torch.utils.data.IterableDataset):
    """
    IterableDataset for storing a collection of AnnData objects.
    We only store the filenames to the h5ad files at initialization,
    and the files are only loaded into AnnData objects during __iter__().

    Supports multiple workers, but only up to the number of data shards. This is because
    the shards are currently partitioned across workers, which avoids multiple workers loading the same shard.
    However, the partition gives each worker an integral number of shards, so additional workers will have 0 shards.

    This is an IterableDataset, so it cannot be shuffled by Trainer. The __iter__() function currently
    does not allow shuffling, but shuffling within a shard could be implemented.
    """
    def __init__(self, filenames: list[str], config: BaseConfig, tokenizer: GeneTokenizer):
        """
        Args:
            filenames: paths to h5ad files
            config: any Config class, simplifies the function signature compared to only passing relevant args
        """
        super(IterableAnnDataset).__init__()
        # np.string_ is important because objects get copy-on-access for forked processes.
        # https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.byte_filenames = np.fromiter(filenames, object).astype(np.string_)

        # cumulative_shard_sizes[i + 1] will store the sum of sizes of shards 0 to i.
        self.cumulative_shard_sizes = np.arange(len(self.byte_filenames) + 1) * config.shard_size

        self.config = config
        self.tokenizer = tokenizer

    def __iter__(self):
        """ Generates a dictionary for each example. """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single worker, this case might be included in the multi-worker code already?
            for i, byte_filename in enumerate(self.byte_filenames):
                yield from self._single_shard_generator(byte_filename)
        elif worker_info.id >= len(self.byte_filenames):
            print(f"Warning: More workers than shards, worker {worker_info.id} is idle.")  # TODO: logger
        else:
            # Divide the shards across workers. Might not be evenly balanced right now.
            worker_shard_inds = []
            ideal_shard_size = self.__len__() / worker_info.num_workers
            ideal_start, ideal_end = ideal_shard_size * worker_info.id, ideal_shard_size * (worker_info.id + 1)
            for shard_ind, cumulative_shard_size in enumerate(self.cumulative_shard_sizes):
                # assigning shards based on whether the lower index is above the ideal index
                if cumulative_shard_size >= ideal_start:
                    worker_shard_inds.append(shard_ind)
                    
                # stop whenever the next shard's lower index becomes ideal for the next worker
                if cumulative_shard_size == self.__len__() or self.cumulative_shard_sizes[shard_ind + 1] >= ideal_end:
                    break

            for i in worker_shard_inds:
                yield from self._single_shard_generator(self.byte_filenames[i])

    def __len__(self):
        return self.cumulative_shard_sizes[-1]

    def _single_shard_generator(self, byte_filename):
        """ Yields all the data in a single shard. Shuffling not implemented yet. """
        adata = read_h5ad(
            str(byte_filename, encoding="utf-8"),
        )
        # not sure if this is that helpful since X is still sparse
        # also might use too much memory if there are many workers?
        adata = adata.to_memory()
        if not scipy.sparse.issparse(adata.X):
            raise NotImplementedError

        # could iterate on shuffled permutation instead, but remember https://github.com/huggingface/transformers/blob/19e5ed736611227b004c6f55679ce3536db3c28d/src/transformers/trainer_pt_utils.py#L705
        for cell in adata:
            input_ids, token_type_ids, labels = self.tokenizer(cell)
            cell_data = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            }
            yield cell_data

