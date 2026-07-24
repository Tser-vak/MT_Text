"""Global seed for reproducibility (Day-1 utility). ONE import point
(`seed.seed_everything`) so the baseline, training, and eval all seed
identically -- a run you can't reproduce is a run you can't defend.

Seeds stdlib `random`, numpy, and torch (CPU + CUDA) when present, plus
PYTHONHASHSEED. numpy/torch are imported lazily so the call still works on the
data-only local box; on the VM they're installed and all get seeded.

NOT set here: `torch.use_deterministic_algorithms(True)` / cudnn-deterministic.
Those force bitwise-identical CUDA kernels at a real speed cost and some ops
lack deterministic impls (they raise). For an SFT run we want reproducible
seeding (data order, init, dropout), not bitwise CUDA determinism -- add it
only if a specific result needs to be bit-reproduced.
"""
import os
import random

SEED = 42


def seed_everything(seed: int = SEED) -> int:
    """Seed every RNG in the stack; returns the seed so callers can log it."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # no-op if no CUDA; safe to call
    except ImportError:
        pass
    return seed


if __name__ == "__main__":
    seed_everything()
    a = [random.random() for _ in range(5)]
    seed_everything()
    b = [random.random() for _ in range(5)]
    assert a == b, (a, b)
    print("seed.seed_everything OK")