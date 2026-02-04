#include <iomanip>
#include <iostream>

#include "num_threads.h"
#include "performance.h"
#include "test.h"
#include "vector_mod.h"

int main(int argc, char** argv)
{
    std::cout << "==Correctness tests. ";
    for (std::size_t testIdx = 0; testIdx < test_data_count; ++testIdx)
    {
        const auto& tc = test_data[testIdx];
        if (tc.result != vector_mod(tc.dividend, tc.dividend_size, tc.divisor))
        {
            std::cout << "FAILURE==\n";
            return -1;
        }
    }
    std::cout << "ok.==\n";
    std::cout << "==Performance tests. ";
    const auto perfStats = run_experiments();
    std::cout << "Done==\n";

    const std::size_t wordHexWidth = 2 * sizeof(IntegerWord);
    const std::size_t valueColWidth = 3 + wordHexWidth;

    std::cout << std::setfill(' ') << std::setw(2) << "T:" << " |" << std::setw(valueColWidth) << "Value:" << " | "
              << std::setw(14) << "Duration, ms:" << " | Acceleration:\n";
    for (std::size_t threadsUsed = 1; threadsUsed <= perfStats.size(); ++threadsUsed)
    {
        const auto& row = perfStats[threadsUsed - 1];

        std::cout << std::setw(2) << threadsUsed << " | 0x" << std::setw(wordHexWidth) << std::setfill('0') << std::hex << row.result;
        std::cout << " | " << std::setfill(' ') << std::setw(14) << std::dec << row.time.count();
        std::cout << " | " << (static_cast<double>(perfStats[0].time.count()) / row.time.count()) << "\n";
    }

    return 0;
}