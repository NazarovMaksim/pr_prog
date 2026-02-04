#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

#include <omp.h>

double av_omp_3(const double* data, size_t size) {
    unsigned proc_count = omp_get_num_procs();
    unsigned thread_count;
    double* partial = static_cast<double*>(calloc(proc_count, sizeof(double)));

#pragma omp parallel shared(thread_count)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            thread_count = omp_get_num_threads();
        }

        double local_sum = 0.0;
        for (size_t i = t; i < size; i += thread_count)
            local_sum += data[i];
        partial[t] = local_sum;
    }

    double sum = 0.0;
    for (size_t i = 0; i < proc_count; i++)
        sum += partial[i];

    free(partial);

    return sum / size;
}

struct sum_t {
    double v;
    char padding[64 - sizeof(double)];
};

double av_omp_4(const double* V, size_t n) {
    unsigned proc_count = omp_get_num_procs();
    unsigned thread_count;
    struct sum_t* partial = static_cast<sum_t*>(calloc(proc_count, sizeof(sum_t)));

#pragma omp parallel shared(thread_count)
    {
        unsigned t = omp_get_thread_num();
#pragma omp single
        {
            thread_count = omp_get_num_threads();
        }

        double local_sum = 0.0;
        for (size_t i = t; i < n; i += thread_count)
            local_sum += V[i];
        partial[t].v = local_sum;
    }

    double sum = 0.0;
    for (size_t i = 0; i < proc_count; i++)
        sum += partial[i].v;

    free(partial);

    return sum / n;
}

void speedtest_avg_openmp(size_t n, size_t exp_count) {
    std::cout << "======= SPEEDTEST OPEN MP AVG ==========" << std::endl;

    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output.csv"));
    if (!output.is_open())
    {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    output << "T,Time,Avg,Acceleration\n";

    double baseline_time_sum = 0.0;
    for (int threads = 1; threads <= static_cast<int>(std::thread::hardware_concurrency()); threads++) {
        omp_set_num_threads(threads);

        double time_sum = 0;
        auto t0 = std::chrono::steady_clock::now();


        for (size_t exp = 0; exp < exp_count; exp++) {

            double* buffer = static_cast<double*>(malloc(n * sizeof(double)));

            for (size_t i = 0; i < n; i++) {
                buffer[i] = (double)i;
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            av_omp_4(buffer, n);
            auto t2 = std::chrono::high_resolution_clock::now();
            time_sum += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

            free(buffer);
        }

        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);

        if (threads == 1) {
            baseline_time_sum = time_sum;
        }

        std::cout << "AVG: T = " << threads << "\t| total experiment time: " << total_time.count() << "\t| avg time = "
                  << time_sum / exp_count << "\tacceleration = "
                  << (baseline_time_sum / exp_count) / (time_sum / exp_count) << "\n";

        output << threads << "," << total_time.count() << "," << time_sum / exp_count << ","
               << (baseline_time_sum / exp_count) / (time_sum / exp_count) << std::endl;

    }

    output.close();
}

int main()
{
    speedtest_avg_openmp(1 << 25, 5);
    return 0;
}