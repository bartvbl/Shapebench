#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <mutex>
#include "nlohmann/json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "dataset/Dataset.h"
#include "utils/gl/Shader.h"
#include "utils/gl/GeometryBuffer.h"
#include "filters/FilteredMeshPair.h"
#include "filters/Filter.h"
#include "utils/filterUtils/OccludedSceneGenerator.h"

namespace ShapeBench {
    class MultiViewOcclusionFilter : public ShapeBench::Filter {
        OccludedSceneGenerator sceneGenerator;

    public:
        virtual void init(const nlohmann::json& config, bool invalidateCaches);
        virtual void destroy();
        virtual void saveCaches(const nlohmann::json& config);

        FilterOutput
        apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
              ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed,
              const nlohmann::json &filterOutputPreviousSequence) override;

    };
}
