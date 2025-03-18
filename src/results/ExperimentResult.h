#pragma once

#include <cstdint>
#include <vector>
#include "dataset/Dataset.h"
#include "nlohmann/json.hpp"
#include "benchmarkCore/ComputedConfig.h"

namespace ShapeBench {
    struct PRCMeasure {
        // Used for computing the Tao ratio
        double distanceToNearestNeighbour = 0;
        double distanceToSecondNearestNeighbour = 0;

        // Used for determining if the model matches
        uint32_t scenePointMeshID = 0;
        uint32_t modelPointMeshID = 0;

        // Used for determining whether the euclidean distance is within half the support radius
        ShapeDescriptor::cpu::float3 nearestNeighbourVertexScene = {0, 0, 0};
        ShapeDescriptor::cpu::float3 nearestNeighbourVertexModel = {0, 0, 0};
    };

    struct MatchingPerformanceMeasure {
        PRCMeasure prcMeasure;
        uint32_t descriptorDistanceIndex = 0;
    };

    struct ObjectSpecificResults {
        ShapeDescriptor::OrientedPoint originalVertexLocation;
        ShapeDescriptor::OrientedPoint filteredVertexLocation;
        float fractionAddedNoise = 0;
        float fractionSurfacePartiality = 0;
        nlohmann::json filterOutput;
    };

    struct ExperimentResultsEntry {
        bool included = false;
        ShapeBench::VertexInDataset sourceVertex;
        uint32_t filteredDescriptorRank = 0;
        ShapeBench::ObjectSpecificResults modelObject;
        ShapeBench::ObjectSpecificResults sceneObject;

        // All necessary information to later compute PRC curve and its corresponding AUC metric
        PRCMeasure prcMetadata;
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
