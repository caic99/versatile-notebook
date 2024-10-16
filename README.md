# `versatile_notebook`: Make C++, CUDA, and MPI works in Jupyter Notebook

`versatile_notebook` is a Jupyter Notebook extension that allows you to write and run C++, CUDA, and MPI code in Jupyter Notebook, without installing a new jupyter kernel.
It leverages [customized](https://ipython.readthedocs.io/en/stable/config/custommagics.html) [cell magics](https://ipython.readthedocs.io/en/stable/interactive/magics.html) of the default iPython kernel, allowing an drop-in enhancement without installing a new kernel.

The magic saves the code cell to a temporary file, and runs the corresponding compiler to build and run the code.

| Language | Magic Commands | Build | Run |
| --- | --- | --- | --- |
| C++ | `%%cpp` | `g++` | `./a.out` |
| CUDA | `%%cu` | `nvcc` | `./a.out` |
| MPI | `%%mpi` | `mpic++` | `mpirun -n x ./a.out` |

> `a.out` is the default output executable name.

Make sure you have the corresponding compiler installed on your system.

Please see [./demo.ipynb](./demo.ipynb) for examples.

## Installation

First, install and load the extension:
```bash
pip install git+https://github.com/caic99/versatile_notebook
%load_ext versatile_notebook
```

Build and run a C++ code cell:
```cpp
%%cpp
#include <iostream>
int main() {
	std::cout << "Hello, world!" << std::endl;
	return 0;
}
```

For CUDA, use `%%cuda`:
```cpp
%%cu
#include <iostream>
#include <cuda_runtime.h>
__global__ void kernel() {
    printf("Hello from CUDA kernel!\n");
}
int main() {
    int device = 0;
    cudaGetDevice(&device);
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

For MPI, use `%%mpi -n 4 cpp` to build a C++ source file, and run it with 4 processes:
```cpp
%%mpi -n 4 cpp
#include <mpi.h>
#include <iostream>
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cout << "Hello from process " << rank << std::endl;
    MPI_Finalize();
    return 0;
}
```
