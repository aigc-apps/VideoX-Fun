"""DataLoader-side building blocks shared by the training scripts.

`SeededDataLoader` keys sample and collate randomness to (seed, epoch, index), so it survives
worker restarts and skipped batches.
"""

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, default_collate


def sample_seed(*values):
    payload = ":".join(str(int(value)) for value in values).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


@contextmanager
def seeded_data_rng(seed):
    """Isolate CPU augmentation RNG, including when loading in the training process."""
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # torch.manual_seed also touches CUDA; data workers must only seed CPU.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class _EpochSeededDataset(Dataset):
    """Derive augmentation seeds from (run seed, epoch, sample index).

    The shared epoch scalar also reaches persistent workers. The sampler still addresses the
    original dataset; `SeededDataLoader` wraps only the DataLoader's dataset argument. Pair this
    wrapper with `_SeededCollate` so seed envelopes do not reach training.
    """

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    @property
    def checkpoint_dataset(self):
        return self.dataset

    def set_data_epoch(self, epoch):
        # Accelerate's fresh skip_first_batches wrapper starts its own iteration
        # counter at zero. Keep the saved data epoch independent of that counter.
        self.epoch.fill_(epoch)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        seed = sample_seed(self.seed, int(self.epoch), index)
        with seeded_data_rng(seed):
            return seed, self.dataset[index]


class _SeededCollate:
    """Key the batch-level draws (bucket size, downsample ratio) on the batch contents."""

    def __init__(self, collate_fn):
        self.collate_fn = collate_fn

    def __call__(self, examples):
        seeds, samples = zip(*examples)
        with seeded_data_rng(sample_seed(*seeds)):
            return self.collate_fn(list(samples))


class SeededDataLoader(DataLoader):
    """`DataLoader` drop-in whose data-side randomness is keyed by (seed, epoch, index).

    Pass the original dataset and collate function plus the run `seed` and `process_index`; the
    sample and batch wrappers, the worker generator and `persistent_workers` are derived here.
    Call `set_data_epoch(epoch)` once per epoch, before that epoch's first iteration: it advances
    every data-side stream that depends on the epoch — the scalar the workers read, the worker
    base seeds and the sampler's permutation generator — so none of them can drift out of step.

    The method is deliberately not named `set_epoch`: Accelerate's `DataLoaderShard.__iter__`
    calls `set_epoch(self.iteration)`, and that counter restarts at zero after a resume or a
    `skip_first_batches` wrapper, so an automatic epoch would silently reseed the wrong one.
    """

    def __init__(self, dataset, *, seed, process_index=0, collate_fn=None, num_workers=0,
                 persistent_workers=None, **kwargs):
        self.data_seed = int(seed) + int(process_index)
        self.sampler_seed = int(seed)
        self._seeded_dataset = _EpochSeededDataset(dataset, self.data_seed)
        self._worker_generator = torch.Generator().manual_seed(self.data_seed)
        super().__init__(
            self._seeded_dataset,
            generator=self._worker_generator,
            collate_fn=_SeededCollate(collate_fn if collate_fn is not None else default_collate),
            num_workers=num_workers,
            persistent_workers=bool(num_workers) if persistent_workers is None else persistent_workers,
            **kwargs,
        )

    def set_data_epoch(self, epoch):
        self._seeded_dataset.set_data_epoch(epoch)
        self._worker_generator.manual_seed(self.data_seed + epoch)
        sampler = getattr(self.batch_sampler, "sampler", None)
        if isinstance(getattr(sampler, "generator", None), torch.Generator):
            sampler.generator = torch.Generator().manual_seed(self.sampler_seed + epoch)
