# POPCLL for PyTorch

## Installation

### To build package yourself

1. `cd` into `popcll_torch`
2. Run `python setup.py install`

### To use pre-built package

1. `cd` into `popcll_torch/dist`
2. Run `pip install popcll_torch-1.0-cp39-cp39-linux_x86_64.whl`


## Usage

```python
import torch as ch
from popcll_torch import popcll
z = ch.tensor([0,1,2,3,4,5,6,7,8], dtype=ch.long).cuda()
counts = popcll(z)
```

Currently only works with int/long 1-D tensors on CUDA.