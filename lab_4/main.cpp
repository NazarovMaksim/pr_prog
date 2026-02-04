#include <chrono>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <omp.h>
#include <random>
#include <thread>
#include <vector>

using namespace std::numbers;

constexpr size_t kMaxTaskDepth = 100;

void print_vector(const std::vector<std::complex<double>>& v) {
    for (std::size_t i = 0; i < v.size(); i++) {
        std::cout << "[" << i << "] " << std::fixed << v[i] << "\n";
    }
}

void randomize_vector(std::vector<std::complex<double>>& v) {
    std::uniform_real_distribution<double> dist(0, 100000);
    static std::random_device rd;
    std::default_random_engine rng(rd());

    for (auto& x : v) {
        x = dist(rng);
    }
}

bool approx_equal(const std::vector<std::complex<double>>& v,
                  const std::vector<std::complex<double>>& u) {
    for (std::size_t i = 0; i < v.size(); i++) {
        if (std::abs(v[i] - u[i]) > 0x1P-10) {
            std::cout << "Mismatch at index " << i
                      << ": " << v[i] << " != " << u[i]
                      << " (diff = " << std::abs(v[i] - u[i]) << ")"
                      << std::endl;
            return false;
        }
    }
    return true;
}

void dft_generic(const std::complex<double>* input,
                 std::complex<double>* output,
                 size_t n,
                 std::complex<double> w,
                 int inverse) {
    for (size_t k = 0; k < n; k++) {
        std::complex<double> sum(0.0, 0.0);
        std::complex<double> wStep = std::pow(w, static_cast<double>(k));
        std::complex<double> wCur = 1.0;

        for (size_t m = 0; m < n; m++) {
            sum += input[m] * wCur;
            wCur *= wStep;
        }

        output[k] = (inverse == -1) ? sum / static_cast<double>(n) : sum;
    }
}

void dft(const std::complex<double>* time,
         std::complex<double>* spectrum,
         size_t n,
         std::complex<double> w) {
    dft_generic(time, spectrum, n, w, 1);
}

void idft(const std::complex<double>* spectrum,
          std::complex<double>* restored,
          size_t n,
          std::complex<double> w) {
    dft_generic(spectrum, restored, n, w, -1);
}

void test_dft_correctness(size_t n) {
    std::cout << "======= Test DFT ==========" << std::endl;

    std::vector<std::complex<double>> original(n);
    if (n < 40)
        randomize_vector(original);

    std::cout << std::endl << "====== Original signal =======" << std::endl;
    if (n < 40)
        print_vector(original);

    std::vector<std::complex<double>> spectrum(n);
    std::complex<double> wForward = std::polar(1.0, -2.0 * pi_v<double> / n);

    auto dftStart = std::chrono::high_resolution_clock::now();
    dft(original.data(), spectrum.data(), n, wForward);
    auto dftEnd = std::chrono::high_resolution_clock::now();
    auto dftTime = std::chrono::duration_cast<std::chrono::milliseconds>(dftEnd - dftStart);

    std::cout << std::endl << "===== Spectrum =======" << std::endl;
    if (n < 40)
        print_vector(spectrum);

    std::cout << std::endl << "====== DFT TIME =======" << std::endl;
    std::cout << "DFT time: " << dftTime.count() << " ms" << std::endl;

    // 3. Выполняем обратное DFT
    std::vector<std::complex<double>> restored(n);
    std::complex<double> wInverse = std::polar(1.0, 2.0 * pi_v<double> / n);

    auto idftStart = std::chrono::high_resolution_clock::now();
    idft(spectrum.data(), restored.data(), n, wInverse);
    auto idftEnd = std::chrono::high_resolution_clock::now();
    auto idftTime = std::chrono::duration_cast<std::chrono::milliseconds>(idftEnd - idftStart);

    std::cout << std::endl << "====== Restored signal =========" << std::endl;
    if (n < 40)
        print_vector(restored);

    std::cout << std::endl << "====== iDFT TIME =======" << std::endl;
    std::cout << "iDFT time: " << idftTime.count() << " ms" << std::endl;

    std::cout << std::endl << "====== Check =========" << std::endl;
    if (!approx_equal(original, restored)) {
        std::cout << std::endl << "====== Error =========" << std::endl;
    } else {
        std::cout << std::endl << "====== OK =========" << std::endl;
    }
}

void fft_openmp_core(std::complex<double>* buf, size_t n, int inverse, size_t depth = 0) {
    if (n <= 1) return;

    std::vector<std::complex<double>> evenPart(n / 2), oddPart(n / 2);

#pragma omp parallel for if(depth == 0 && n > 1000)
    for (size_t i = 0; i < n / 2; i++) {
        evenPart[i] = buf[2 * i];
        oddPart[i] = buf[2 * i + 1];
    }

#pragma omp task shared(evenPart) if(depth < kMaxTaskDepth && n > 1000)
    {
        fft_openmp_core(evenPart.data(), n / 2, inverse, depth + 1);
    }

#pragma omp task shared(oddPart) if(depth < kMaxTaskDepth && n > 1000)
    {
        fft_openmp_core(oddPart.data(), n / 2, inverse, depth + 1);
    }

#pragma omp taskwait

#pragma omp parallel for if(depth == 0 && n > 1000)
    for (size_t i = 0; i < n / 2; i++) {
        double phi = (inverse == 1) ? -2.0 * pi_v<double> * i / n : 2.0 * pi_v<double> * i / n;
        std::complex<double> w = std::polar(1.0, phi);

        std::complex<double> tw = w * oddPart[i];
        buf[i] = evenPart[i] + tw;
        buf[i + n / 2] = evenPart[i] - tw;
    }
}

void fft_recursive(std::complex<double>* data, size_t n, int inverse) {
#pragma omp parallel
#pragma omp single nowait
    {
        fft_openmp_core(data, n, inverse, 0);
    }
}

void fft_openmp(std::complex<double>* data, size_t n) {
    fft_recursive(data, n, 1);
}

void ifft_openmp(std::complex<double>* data, size_t n) {
    fft_recursive(data, n, -1);

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        data[i] /= static_cast<double>(n);
    }
}

void test_fft_openmp_correctness(size_t n) {
    std::cout << "======= Test OPEN MP ==========" << std::endl;

    std::vector<std::complex<double>> original(n);
    if (n < 40)
        randomize_vector(original);

    std::cout << std::endl << "====== Original signal =======" << std::endl;
    if (n < 40)
        print_vector(original);

    std::vector<std::complex<double>> baseline = original;
    std::vector<std::complex<double>> spectrum = original;

    auto fft_start = std::chrono::high_resolution_clock::now();
    fft_openmp(spectrum.data(), n);
    auto fft_end = std::chrono::high_resolution_clock::now();
    auto fft_time = std::chrono::duration_cast<std::chrono::milliseconds>(fft_end - fft_start);

    std::cout << std::endl << "====== Spectrum =======" << std::endl;
    if (n < 40)
        print_vector(spectrum);

    std::cout << std::endl << "====== OPEN MP TIME =======" << std::endl;
    std::cout << "FFT time: " << fft_time.count() << " ms" << std::endl;

    auto ifft_start = std::chrono::high_resolution_clock::now();
    ifft_openmp(spectrum.data(), n);
    auto ifft_end = std::chrono::high_resolution_clock::now();
    auto ifft_time = std::chrono::duration_cast<std::chrono::milliseconds>(ifft_end - ifft_start);

    std::cout << std::endl << "====== Restored signal =========" << std::endl;
    if (n < 40)
        print_vector(spectrum);

    std::cout << std::endl << "====== INVERSE OPEN MP TIME =======" << std::endl;
    std::cout << "Total time: " << (fft_time + ifft_time).count() << " ms" << std::endl;

    if (!approx_equal(baseline, spectrum)) {
        std::cout << std::endl << "====== Error =========" << std::endl;
    } else {
        std::cout << std::endl << "====== OK =========" << std::endl;
    }
}

void speedtest_fft_openmp(size_t n, size_t exp_count) {
    std::cout << "======= SPEEDTEST OPEN MP FFT ==========" << std::endl;

    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output.csv"));
    if (!output.is_open()) {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    output << "T,Time,Avg,Acceleration\n";

    std::vector<std::complex<double>> signal(n);
    randomize_vector(signal);

    const std::vector<std::complex<double>> signalRef = signal;

    double timeSumSingle = 0.0;
    const auto hw = std::thread::hardware_concurrency();

    for (int threads = 1; threads <= hw; threads++) {
        omp_set_num_threads(threads);

        double time_sum = 0;
        auto t0 = std::chrono::steady_clock::now();

        signal = signalRef;

        for (size_t exp = 0; exp < exp_count; exp++) {
            std::vector<std::complex<double>> spectrum = signal;
            std::vector<std::complex<double>> restored(n);

            auto t1 = std::chrono::high_resolution_clock::now();
            fft_openmp(spectrum.data(), n);
            ifft_openmp(spectrum.data(), n);
            auto t2 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            if (!approx_equal(signalRef, spectrum)) {
                std::cout << "Warning: FFT/IFFT mismatch in experiment " << exp + 1
                          << " with " << threads << " thread_num\n";
            }
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0);

        if (threads == 1) {
            timeSumSingle = time_sum;
        }

        std::cout << "FFT: T = " << threads << "\t| total experiment time: " << total_time.count()
                  << "\t| avg fft time = " << time_sum / exp_count
                  << "\tacceleration = " << (timeSumSingle / exp_count) / (time_sum / exp_count) << "\n";

        output << threads << "," << total_time.count() << "," << time_sum / exp_count << ","
               << (timeSumSingle / exp_count) / (time_sum / exp_count) << std::endl;
    }

    output.close();
}

int main() {
    //    test_dft_correctness(1 << 4);

    //    std::cout << std::endl << std::endl;

    //    test_fft_openmp_correctness(1 << 4);

    speedtest_fft_openmp(1 << 14, 5);
    return 0;
}
