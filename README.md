# Hadamard Row-Wise Generation Algorithm
[![arXiv](https://img.shields.io/badge/arXiv-2409.02406-b31b1b.svg)](https://arxiv.org/abs/2409.02406) 
## Abstract
In this paper, we introduce an efficient algorithm for generating specific Hadamard rows, addressing the memory demands of pre-computing the entire matrix. 
Leveraging Sylvester's recursive construction, our method generates the required $i$-th row on demand, significantly reducing computational resources. 
The algorithm uses the Kronecker product to construct the desired row from the binary representation of the index, without creating the full matrix. 
This approach is particularly useful for single-pixel imaging systems that need only one row at a time.

## How to cite
If this code is useful for your and you use it in an academic work, please consider citing this paper as


```bib
@misc{monroy2024hadamard,
      title={Hadamard Row-Wise Generation Algorithm}, 
      author={Brayan Monroy and Jorge Bacca},
      year={2024},
      eprint={2409.02406},
      archivePrefix={arXiv},
      primaryClass={cs.DS},
      url={https://arxiv.org/abs/2409.02406}, 
}
```
