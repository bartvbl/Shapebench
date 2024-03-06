#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <mutex>
#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "benchmarkCore/Dataset.h"
#include "GLFW/glfw3.h"
#include "utils/gl/Shader.h"
#include "utils/gl/GeometryBuffer.h"
#include "filters/FilteredMeshPair.h"

namespace ShapeBench {
    class OccludedSceneGenerator {
        GLFWwindow* window = nullptr;
        bool isDestroyed = false;
        Shader objectIDShader;
        Shader fullscreenQuadShader;
        uint32_t frameBufferID = 0;
        uint32_t renderBufferID = 0;
        uint32_t renderTextureID = 0;
        GeometryBuffer screenQuadVAO;
        uint32_t offscreenTextureWidth = 0;
        uint32_t offscreenTextureHeight = 0;
        std::mutex occlusionFilterLock;

    public:
        explicit OccludedSceneGenerator();
        ~OccludedSceneGenerator();
        void computeOccludedMesh(const nlohmann::json& config, ShapeBench::FilteredMeshPair &scene, uint64_t seed);
        void init(uint32_t visibilityImageWidth, uint32_t visibilityImageHeight);
        void destroy();
    };

    static OccludedSceneGenerator occlusionSceneGeneratorInstance;



    inline void applyOcclusionFilter(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, uint64_t seed) {
        occlusionSceneGeneratorInstance.computeOccludedMesh(config, scene, seed);
    }
}
