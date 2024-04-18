#pragma once

#include <cstdint>
#include <vector>
#include "dataset/Dataset.h"
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"

namespace ShapeBench {
    struct ExperimentResultsEntry {
        bool included = false;
        ShapeBench::VertexInDataset sourceVertex;
        uint32_t filteredDescriptorRank = 0;
        ShapeDescriptor::OrientedPoint originalVertexLocation;
        ShapeDescriptor::OrientedPoint filteredVertexLocation;
        float fractionAddedNoise = 0;
        float fractionSurfacePartiality = 0;
        nlohmann::json filterOutput;
    };

    struct ExperimentResult {
        std::string methodName = "NOT SPECIFIED";
        nlohmann::json usedConfiguration;
        nlohmann::json methodMetadata;
        ShapeBench::ComputedConfig usedComputedConfiguration;
        uint64_t experimentRandomSeed = 0;
        uint32_t experimentIndex = 0;

        std::vector<ExperimentResultsEntry> vertexResults;
    };
}
