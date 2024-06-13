#pragma once

#include <filesystem>
#include "json.hpp"
#include "dataset/Dataset.h"

namespace ShapeBench {
    struct BenchmarkConfiguration {
        std::filesystem::path configurationFilePath;
        std::filesystem::path replicationFilePath;
        bool enableReplicationMode = false;
        nlohmann::json configuration;
        nlohmann::json computedConfiguration;
        ShapeBench::Dataset dataset;
    };
}