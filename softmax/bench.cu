#include <nvbench/nvbench.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda_utils.cuh"
#include "vectorized_4.cuh"

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

void softmax_vectorized_bench(nvbench::state& state) {
    const auto N = static_cast<int>(state.get_int64("N"));
    const int M = 1024;
    const int matsize = M * N;

    // Allocate and initialize host data with clamped normal distribution
    thrust::host_vector<float> h_mat(matsize);
    for (int i = 0; i < matsize; i++) {
        h_mat[i] = random_normal_clamped(-10.0f, 10.0f);
    }

    // Transfer to device
    thrust::device_vector<float> d_mat = h_mat;
    thrust::device_vector<float> d_res(matsize);

    float* matd = thrust::raw_pointer_cast(d_mat.data());
    float* resd = thrust::raw_pointer_cast(d_res.data());

    // Report throughput metrics
    state.add_element_count(matsize, "Elements");
    state.add_global_memory_reads<float>(matsize);
    state.add_global_memory_writes<float>(matsize);

    // Kernel launch parameters (same as run_kernel_4)
    dim3 block_size(1024);
    dim3 grid_size(M);
    int warp_size = 32;
    size_t smem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    state.exec([&](nvbench::launch& launch) {
        softmax_kernel_4<<<grid_size, block_size, smem_size, launch.get_stream()>>>(
            matd, resd, M, N);
    });
}

NVBENCH_BENCH(softmax_vectorized_bench)
    .add_int64_power_of_two_axis("N", nvbench::range(11, 17, 1))  // 2048 to 131072
    .set_timeout(1);