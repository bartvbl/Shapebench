#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <mutex>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "dataset/Dataset.h"
#include "GLFW/glfw3.h"
#include "utils/gl/Shader.h"
#include "utils/gl/GeometryBuffer.h"
#include "filters/FilteredMeshPair.h"
#include "filters/Filter.h"
#include "utils/filterUtils/OccludedSceneGenerator.h"

namespace ShapeBench {
    class OcclusionFilter : public ShapeBench::Filter {
        OccludedSceneGenerator sceneGenerator;

    public:
        virtual void init(const nlohmann::json& config);
        virtual void destroy();
        virtual void saveCaches(const nlohmann::json& config);

        virtual FilterOutput apply(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed);

    };
}
