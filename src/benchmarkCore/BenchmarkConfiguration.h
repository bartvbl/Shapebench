#pragma once

#include <filesystem>
#include "nlohmann/json.hpp"
#include "dataset/Dataset.h"

namespace ShapeBench {
    struct ReplicationSettings {
        bool enabled = false;
        std::string methodName = "UNKNOWN_METHOD";
        uint32_t experimentIndex = 0;
        nlohmann::json experimentResults;
    };

    struct BenchmarkConfiguration {
        std::filesystem::path configurationFilePath;
        std::filesystem::path replicationFilePath;
        nlohmann::json configuration;
        nlohmann::json computedConfiguration;
        ShapeBench::Dataset dataset;
        ReplicationSettings replicationSettings;
    };
}