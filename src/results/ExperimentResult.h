#pragma once

#include <cstdint>
#include <vector>
#include "benchmarkCore/Dataset.h"
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"

namespace ShapeBench {
    struct ExperimentResultsEntry {
        ShapeBench::VertexInDataset sourceVertex;
        uint32_t filteredDescriptorRank = 0;
    };

    struct ExperimentResult {
        std::string methodName = "NOT SPECIFIED";
        nlohmann::json usedConfiguration;
        ShapeBench::ComputedConfig usedComputedConfiguration;
        uint64_t experimentRandomSeed = 0;
        float fractionAddedNoise = 0;
        float fractionSurfacePartiality = 0;

        std::vector<ExperimentResultsEntry> vertexResults;
    };
}
