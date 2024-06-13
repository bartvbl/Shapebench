#pragma once

#include <filesystem>
#include "json.hpp"
#include "dataset/Dataset.h"

namespace ShapeBench {
    struct ReplicationSettings {
        bool enabled = false;
        std::string methodName = "UNKNOWN_METHOD";
        uint32_t experimentIndex = 0;
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