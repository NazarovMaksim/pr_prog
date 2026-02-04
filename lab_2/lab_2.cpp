#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <vector>

using namespace std;

void transposeAVX2(double* dst, const double* src, size_t rows, size_t cols)
{
    constexpr size_t blockSize = 4; // 4 double = 256 bits

    size_t r = 0, c = 0;
    for (r = 0; r + blockSize <= rows; r += blockSize) {
        for (c = 0; c + blockSize <= cols; c += blockSize) {

            // Load 4x4 block
            __m256d row0 = _mm256_loadu_pd(&src[(r + 0) * cols + c]);
            __m256d row1 = _mm256_loadu_pd(&src[(r + 1) * cols + c]);
            __m256d row2 = _mm256_loadu_pd(&src[(r + 2) * cols + c]);
            __m256d row3 = _mm256_loadu_pd(&src[(r + 3) * cols + c]);

            // Unpack pairs
            __m256d t0 = _mm256_unpacklo_pd(row0, row1);
            __m256d t1 = _mm256_unpackhi_pd(row0, row1);
            __m256d t2 = _mm256_unpacklo_pd(row2, row3);
            __m256d t3 = _mm256_unpackhi_pd(row2, row3);

            // Permute into columns
            __m256d col0 = _mm256_permute2f128_pd(t0, t2, 0x20);
            __m256d col2 = _mm256_permute2f128_pd(t1, t3, 0x20);
            __m256d col1 = _mm256_permute2f128_pd(t0, t2, 0x31);
            __m256d col3 = _mm256_permute2f128_pd(t1, t3, 0x31);

            // Store transposed block
            _mm256_storeu_pd(&dst[(c + 0) * rows + r], col0);
            _mm256_storeu_pd(&dst[(c + 1) * rows + r], col2);
            _mm256_storeu_pd(&dst[(c + 2) * rows + r], col1);
            _mm256_storeu_pd(&dst[(c + 3) * rows + r], col3);
        }

        // tail columns
        for (; c < cols; ++c) {
            for (size_t rr = 0; rr < blockSize && (r + rr) < rows; ++rr) {
                dst[c * rows + (r + rr)] = src[(r + rr) * cols + c];
            }
        }
    }

    // tail rows
    for (; r < rows; ++r) {
        for (c = 0; c < cols; ++c) {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
}

void matrixPrint(double* mat, size_t rows, size_t cols)
{
    cout << "Matrix:\n";
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            cout << mat[r * cols + c] << ' ';
        }
        cout << endl;
    }
}

void matrixMul(double* out, const double* lhs, const double* rhs, size_t sharedDim, size_t rowsA, size_t colsB)
{
    for (size_t r = 0; r < rowsA; r++) {
        for (size_t c = 0; c < colsB; c++) {
            double sum = 0;
            for (size_t k = 0; k < sharedDim; k++) {
                sum += lhs[r * sharedDim + k] * rhs[k * colsB + c];
            }
            out[r * colsB + c] = sum;
        }
    }
}

void matrixMulAVX2(double* out, const double* lhs, const double* rhs, size_t sharedDim, size_t rowsA, size_t colsB)
{
    std::vector<double> rhsT(colsB * sharedDim, 0.0);

    if (sharedDim % 4 == 0 && colsB % 4 == 0) {
        transposeAVX2(rhsT.data(), rhs, colsB, sharedDim);
    } else {
        for (size_t r = 0; r < sharedDim; ++r) {
            for (size_t c = 0; c < colsB; ++c) {
                rhsT[c * sharedDim + r] = rhs[r * colsB + c];
            }
        }
    }

    const double* rhsTData = rhsT.data();
    double lane[4];

    for (size_t r = 0; r < rowsA; r++) {
        for (size_t c = 0; c < colsB; c++) {
            __m256d acc = _mm256_setzero_pd();
            size_t k = 0;

            // blocks of 4 doubles
            for (; k + 3 < sharedDim; k += 4) {
                __m256d x = _mm256_loadu_pd(&lhs[r * sharedDim + k]);
                __m256d y = _mm256_loadu_pd(&rhsTData[c * sharedDim + k]);
                acc = _mm256_add_pd(acc, _mm256_mul_pd(x, y));
            }

            // horizontal sum
            _mm256_storeu_pd(lane, acc);
            double dot = lane[0] + lane[1] + lane[2] + lane[3];

            // tail
            for (; k < sharedDim; ++k) {
                dot += lhs[r * sharedDim + k] * rhsTData[c * sharedDim + k];
            }

            out[r * colsB + c] = dot;
        }
    }
}

void matrixMulAVX2_gather(double* out, const double* lhs, const double* rhs, size_t sharedDim, size_t rowsA, size_t colsB)
{
    double lane[4];

    for (size_t r = 0; r < rowsA; r++) {
        for (size_t c = 0; c < colsB; c++) {
            __m256d acc = _mm256_setzero_pd();
            size_t k = 0;

            // blocks of 4 doubles
            for (; k + 3 < sharedDim; k += 4) {
                __m256d a_vec = _mm256_loadu_pd(&lhs[r * sharedDim + k]);

                __m256i idx = _mm256_setr_epi64x(
                    k * colsB + c,
                    (k + 1) * colsB + c,
                    (k + 2) * colsB + c,
                    (k + 3) * colsB + c);

                __m256d b_vec = _mm256_i64gather_pd(rhs, idx, sizeof(double));
                acc = _mm256_add_pd(acc, _mm256_mul_pd(a_vec, b_vec));
            }

            // horizontal sum
            _mm256_storeu_pd(lane, acc);
            double dot = lane[0] + lane[1] + lane[2] + lane[3];

            // tail
            for (; k < sharedDim; ++k) {
                dot += lhs[r * sharedDim + k] * rhs[k * colsB + c];
            }

            out[r * colsB + c] = dot;
        }
    }
}

//void matrixMul_clear(double *result, const double *matA, const double *matB, size_t sharedDim, size_t rowsA, size_t colsB){
//    for (size_t r1 = 0; r1 < rowsA / sizeof(__m256d); r1++)
//        for (size_t c2 = 0; c2 < colsB; c2++) {
//            __m256d accum = _mm256_setzero_pd();
//            for (size_t i = 0; i < sharedDim; i++)
//                __m256d x = _mm256_loadu_pd(&matA[r1 * sharedDim + i]);
//                accum += matA[r1 * sharedDim + i] * matB[i * colsB + c2];
//            result[r1 * colsB + c2] = accum;
//        }
//}

void speedtest(size_t rowsA, size_t sharedDim, size_t colsB)
{
    std::vector<double> matA(rowsA * sharedDim, 1.0);
    std::vector<double> matB(colsB * sharedDim);
    std::vector<double> outSync(rowsA * colsB, 0.0);
    std::vector<double> outAvx(rowsA * colsB, 0.0);
    std::vector<double> outGather(rowsA * colsB, 0.0);

    for (size_t i = 0; i < colsB * sharedDim; ++i) {
        matB[i] = (double)i;
    }

    auto t1 = std::chrono::steady_clock::now();
    matrixMul(outSync.data(), matA.data(), matB.data(), sharedDim, rowsA, colsB);
    auto t2 = std::chrono::steady_clock::now();

    auto t3 = std::chrono::steady_clock::now();
    matrixMulAVX2(outAvx.data(), matA.data(), matB.data(), sharedDim, rowsA, colsB);
    auto t4 = std::chrono::steady_clock::now();

    auto t5 = std::chrono::steady_clock::now();
    matrixMulAVX2_gather(outGather.data(), matA.data(), matB.data(), sharedDim, rowsA, colsB);
    auto t6 = std::chrono::steady_clock::now();

    cout << "Duration of synchronous calc: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << endl;
    cout << "Duration of AVX2 calc: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << endl;
    cout << "Duration of AVX2_Gather calc: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count() << endl;
}

void speedtest_avg_openmp(size_t n)
{
    std::cout << "======= SPEEDTEST MATRIX MUL ==========" << std::endl;

    std::cout << "Current dir: " << std::filesystem::current_path() << std::endl;
    auto base_dir = std::filesystem::current_path().parent_path();
    std::ofstream output(base_dir.append("output.csv"));
    if (!output.is_open()) {
        std::cout << "Error while opening file" << std::endl;
        return;
    }

    output << "Sync,AVX2,AVX2Acceleration,Gather,GatherAcceleration\n";

    std::vector<double> matA(n * n, 1.0);
    std::vector<double> matB(n * n);
    std::vector<double> outSync(n * n, 0.0);
    std::vector<double> outAvx(n * n, 0.0);
    std::vector<double> outGather(n * n, 0.0);

    for (size_t i = 0; i < n * n; ++i) {
        matB[i] = (double)i;
    }

    auto t1 = std::chrono::steady_clock::now();
    matrixMul(outSync.data(), matA.data(), matB.data(), n, n, n);
    auto t2 = std::chrono::steady_clock::now();

    auto t3 = std::chrono::steady_clock::now();
    matrixMulAVX2(outAvx.data(), matA.data(), matB.data(), n, n, n);
    auto t4 = std::chrono::steady_clock::now();

    auto t5 = std::chrono::steady_clock::now();
    matrixMulAVX2_gather(outGather.data(), matA.data(), matB.data(), n, n, n);
    auto t6 = std::chrono::steady_clock::now();

    const auto syncMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    const auto avxMs = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count();
    const auto gatherMs = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5).count();

    std::cout << "Sync time: " << syncMs
              << "\t| AVX2 time: " << avxMs
              << "\t| AVX2 Acceleration: " << (double)syncMs / avxMs
              << "\t| Gather time: " << gatherMs
              << "\t| Gather Acceleration: " << (double)syncMs / gatherMs << endl;

    output << syncMs << ','
           << avxMs << ','
           << (double)syncMs / avxMs << ','
           << gatherMs << ','
           << (double)syncMs / gatherMs << endl;

    output.close();
}

int main()
{
    cout << "Test 1:" << endl;

    std::size_t aRows = 2, kDim = 3, bCols = 2;
    std::vector<double> A(aRows * kDim, 1.0);
    std::vector<double> B(kDim * bCols, 2.0);
    std::vector<double> R(aRows * bCols, 0.0);

    matrixMul(R.data(), A.data(), B.data(), kDim, aRows, bCols);
    matrixPrint(R.data(), aRows, bCols);

    cout << "Test 2:" << endl;
    aRows = 2;
    kDim = 3;
    bCols = 2;

    A = std::vector<double>({1.0, 2.0, -3.0, -2.0, 13.0, -2.0});
    B = std::vector<double>({3.0, 4.0, 5.0, -1.0, 4.0, 4.0});
    R = std::vector<double>(aRows * bCols, 0.0);

    matrixMul(R.data(), A.data(), B.data(), kDim, aRows, bCols);
    matrixPrint(R.data(), aRows, bCols);

    speedtest_avg_openmp(1000);
    return 0;
}
