#include <iostream>
#include <fftw3.h>
#include <vector>
#include <sys/resource.h>
#include <unistd.h>

// Function to measure current memory usage
long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

// 1D FFT Convolution with pre-allocated buffers and FFTW_MEASURE
void fft_convolution_1d(int n1, fftw_complex* fft1, fftw_complex* fft2, fftw_complex* result) {
    std::vector<double> signal1(n1, 1.0);
    std::vector<double> signal2(n1, 1.0);

    fftw_plan p1 = fftw_plan_dft_r2c_1d(n1, signal1.data(), fft1, FFTW_MEASURE);
    fftw_plan p2 = fftw_plan_dft_r2c_1d(n1, signal2.data(), fft2, FFTW_MEASURE);

    fftw_execute(p1);
    fftw_execute(p2);

    for (int i = 0; i < n1; ++i) {
        result[i][0] = fft1[i][0] * fft2[i][0] - fft1[i][1] * fft2[i][1];
        result[i][1] = fft1[i][0] * fft2[i][1] + fft1[i][1] * fft2[i][0];
    }

    fftw_plan p3 = fftw_plan_dft_c2r_1d(n1, result, signal1.data(), FFTW_MEASURE);
    fftw_execute(p3);

    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);
}

// 2D FFT Convolution with pre-allocated buffers and FFTW_MEASURE
void fft_convolution_2d(int n1, int n2, fftw_complex* fft1, fftw_complex* fft2, fftw_complex* result) {
    std::vector<double> signal1(n1 * n2, 1.0);
    std::vector<double> signal2(n1 * n2, 1.0);

    fftw_plan p1 = fftw_plan_dft_r2c_2d(n1, n2, signal1.data(), fft1, FFTW_MEASURE);
    fftw_plan p2 = fftw_plan_dft_r2c_2d(n1, n2, signal2.data(), fft2, FFTW_MEASURE);

    fftw_execute(p1);
    fftw_execute(p2);

    for (int i = 0; i < n1 * n2; ++i) {
        result[i][0] = fft1[i][0] * fft2[i][0] - fft1[i][1] * fft2[i][1];
        result[i][1] = fft1[i][0] * fft2[i][1] + fft1[i][1] * fft2[i][0];
    }

    fftw_plan p3 = fftw_plan_dft_c2r_2d(n1, n2, result, signal1.data(), FFTW_MEASURE);
    fftw_execute(p3);

    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);
}

// 3D FFT Convolution with pre-allocated buffers and FFTW_MEASURE
void fft_convolution_3d(int n1, int n2, int n3, fftw_complex* fft1, fftw_complex* fft2, fftw_complex* result) {
    std::vector<double> signal1(n1 * n2 * n3, 1.0);
    std::vector<double> signal2(n1 * n2 * n3, 1.0);

    fftw_plan p1 = fftw_plan_dft_r2c_3d(n1, n2, n3, signal1.data(), fft1, FFTW_MEASURE);
    fftw_plan p2 = fftw_plan_dft_r2c_3d(n1, n2, n3, signal2.data(), fft2, FFTW_MEASURE);

    fftw_execute(p1);
    fftw_execute(p2);

    for (int i = 0; i < n1 * n2 * n3; ++i) {
        result[i][0] = fft1[i][0] * fft2[i][0] - fft1[i][1] * fft2[i][1];
        result[i][1] = fft1[i][0] * fft2[i][1] + fft1[i][1] * fft2[i][0];
    }

    fftw_plan p3 = fftw_plan_dft_c2r_3d(n1, n2, n3, result, signal1.data(), FFTW_MEASURE);
    fftw_execute(p3);

    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);
}

// Main function
int main() {
    // 1D Convolution Test
    int n1 = 32768;  // Larger tensor size
    std::cout << "1D Convolution" << std::endl;

    fftw_complex* fft1_1d = fftw_alloc_complex(n1);
    fftw_complex* fft2_1d = fftw_alloc_complex(n1);
    fftw_complex* result_1d = fftw_alloc_complex(n1);

    long memoryBefore = getMemoryUsage();
    fft_convolution_1d(n1, fft1_1d, fft2_1d, result_1d);
    long memoryAfter = getMemoryUsage();

    std::cout << "Memory used: " << (memoryAfter - memoryBefore) << " KB" << std::endl;

    fftw_free(fft1_1d);
    fftw_free(fft2_1d);
    fftw_free(result_1d);

    // 2D Convolution Test
    int n2 = 32;  // Larger tensor size
    std::cout << "2D Convolution" << std::endl;

    fftw_complex* fft1_2d = fftw_alloc_complex(n1 * n2);
    fftw_complex* fft2_2d = fftw_alloc_complex(n1 * n2);
    fftw_complex* result_2d = fftw_alloc_complex(n1 * n2);

    memoryBefore = getMemoryUsage();
    fft_convolution_2d(n1, n2, fft1_2d, fft2_2d, result_2d);
    memoryAfter = getMemoryUsage();

    std::cout << "Memory used: " << (memoryAfter - memoryBefore) << " KB" << std::endl;

    fftw_free(fft1_2d);
    fftw_free(fft2_2d);
    fftw_free(result_2d);

    // 3D Convolution Test
    int n3 = 32;  // Reduced tensor size for feasibility
    std::cout << "3D Convolution" << std::endl;

    fftw_complex* fft1_3d = fftw_alloc_complex(n1 * n2 * n3);
    fftw_complex* fft2_3d = fftw_alloc_complex(n1 * n2 * n3);
    fftw_complex* result_3d = fftw_alloc_complex(n1 * n2 * n3);

    memoryBefore = getMemoryUsage();
    fft_convolution_3d(n1, n2, n3, fft1_3d, fft2_3d, result_3d);
    memoryAfter = getMemoryUsage();

    std::cout << "Memory used: " << (memoryAfter - memoryBefore) << " KB" << std::endl;

    fftw_free(fft1_3d);
    fftw_free(fft2_3d);
    fftw_free(result_3d);

    return 0;
}
