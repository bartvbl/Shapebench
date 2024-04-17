#pragma once

#include "nlohmann/json.hpp"
#include "Dataset.h"

namespace ShapeBench {
    void downloadFile(const nlohmann::json& config, const DatasetEntry &datasetEntry);
}