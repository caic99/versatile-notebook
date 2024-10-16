import logging
import os
import shutil
import subprocess
import sys
import tempfile
from IPython.core.magic import register_cell_magic

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_nvcc_compiler():
    if shutil.which("nvcc"):
        return "nvcc"
    nvcc_default_path = "/usr/local/cuda/bin/nvcc"
    if os.path.exists(nvcc_default_path):
        return nvcc_default_path
    raise FileNotFoundError(
        "nvcc compiler not found. Please ensure CUDA is installed and nvcc is available in the PATH. See "
        "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup"
    )


def compile_and_run(code, compiler="g++", compile_flags="", mpi_num_processes=None):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".cu" if "nvcc" in compiler else ".cpp"
    ) as temp_file:
        temp_file.write(code.encode("utf-8"))
        temp_file_path = temp_file.name
    logger.debug(f"Temporary file created at {temp_file_path}")

    # Compile the code
    compile_command = f"{compiler} {temp_file_path} {compile_flags}"
    logger.info(f"Compiling: {compile_command}")
    compile_result = subprocess.run(
        compile_command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if compile_result.returncode != 0:
        # Compilation failed
        logger.error(f"Compilation failed: {compile_result.stderr}")
        raise RuntimeError(f"Compilation Error:\n{compile_result.stderr}")

    # Run the compiled executable
    if mpi_num_processes:
        # Set env var to bypass OpenMPI rules
        # since MPICH does not take '--allow-run-as-root'
        os.environ['OMPI_ALLOW_RUN_AS_ROOT']='1'
        os.environ['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM']='1'
        run_command = f"mpirun -n {mpi_num_processes} ./a.out"
    else:
        run_command = "./a.out"
    logger.info(f"Running: {run_command}")
    process = subprocess.Popen(
        run_command.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # Stream the output
    # process.stdout:TextIOWrapper
    while process.poll() is None:
        sys.stdout.write(process.stdout.buffer.read1().decode())

    if process.returncode != 0:
        logger.error(f"Runtime error with exit code {process.returncode}")
        raise RuntimeError(f"Runtime error with exit code {process.returncode}")


@register_cell_magic("cpp")
def cpp_cell_magic(line, cell):
    """
    Jupyter cell magic to handle C++ code compilation and execution.

    Usage:
    %%cpp
    #include <iostream>
    int main() {
        std::cout << "Hello, C++ from Jupyter!" << std::endl;
        return 0;
    }

    You can also provide custom compile flags using the line argument, e.g.:
    %%cpp -fopenmp
    """
    compile_flags = line.strip()
    compile_and_run(cell, compiler="g++", compile_flags=compile_flags)


@register_cell_magic("mpi")
def mpi_cell_magic(line, cell):
    """
    Jupyter cell magic to handle MPI C++ and CUDA source file compilation and execution.

    Usage:
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

    The -n option specifies the number of processes, and you can specify the source type (cpp or cu).
    """
    line_parts = line.split()

    # Should strictly follow the pattern of `%%mpi -n <number of processes> <source type>`
    assert line_parts[0] == "-n", "Number of processes should be specified with -n option"
    assert line_parts[1].isdigit(), "Number of processes should be an integer"
    mpi_num_processes = int(line_parts[1])

    if line_parts[2] == "cpp":
        compiler = "mpic++"
    else:
        raise ValueError("Only C++ source files are supported for MPI execution")

    compile_flags = " ".join(line_parts[3:])

    compile_and_run(
        cell,
        compiler=compiler,
        compile_flags=compile_flags,
        mpi_num_processes=mpi_num_processes,
    )


@register_cell_magic("cu")
def cu_cell_magic(line, cell):
    """
    Jupyter cell magic to handle CUDA C++ code compilation and execution.

    Usage:
    %%cu
    #include <iostream>
    #include <cuda_runtime.h>
    __global__ void kernel() {
        printf("Hello from CUDA kernel!");
    }
    int main() {
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
    }

    You can also provide custom compile flags using the line argument, e.g.:
    %%cu -arch=sm_50
    """
    compiler = get_nvcc_compiler()
    compile_flags = line.strip()
    compile_and_run(cell, compiler=compiler, compile_flags=compile_flags)


def load_ipython_extension(ipython):
    """Register the C++, MPI, and CUDA cell magics when the extension is loaded."""
    ipython.register_magic_function(cpp_cell_magic, "cell")
    ipython.register_magic_function(mpi_cell_magic, "cell")
    ipython.register_magic_function(cu_cell_magic, "cell")


def unload_ipython_extension(ipython):
    """Unregister the C++, MPI, and CUDA cell magics when the extension is unloaded."""
    ipython.unregister_magic_function("cpp")
    ipython.unregister_magic_function("mpi")
    ipython.unregister_magic_function("cu")

"""
Usage: %run test.py

Please ensure `g++`, `nvcc` and MPI components is available in PATH.
"""

if __name__ == "__main__":
    from IPython import get_ipython

    ip = get_ipython()
    if ip:
        load_ipython_extension(ip)
        print("Use `%%cpp` to write and run C++ code in cells.")
        print("Use `%%mpi -n 4 cpp` to write and run MPI C++ code in cells.")
        print("Use `%%cu` to write and run CUDA C++ code in cells.")
