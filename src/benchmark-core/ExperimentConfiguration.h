#pragma once

#include <filesystem>

struct ExperimentConfiguration {
    unsigned long long randomSeed;
    std::filesystem::path outputFile;
};