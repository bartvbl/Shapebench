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
#include "filters/Filter.h"

namespace ShapeBench {
    struct OcclusionRendererSettings {
        float nearPlaneDistance = 0.001;
        float farPlaneDistance = 100.0;
        float fovy = 1.552;
        float objectDistanceFromCamera = 20;
    };

    class OccludedSceneGenerator {
        GLFWwindow* window = nullptr;
        bool isDestroyed = false;
	    bool isCreated = false;
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
        ShapeBench::FilterOutput computeOccludedMesh(const nlohmann::json& config, ShapeBench::FilteredMeshPair &scene, uint64_t seed);
        void init(uint32_t visibilityImageWidth, uint32_t visibilityImageHeight);
        void destroy();
    };

    class OcclusionFilter : public ShapeBench::Filter {
        OccludedSceneGenerator sceneGenerator;

    public:
        virtual void init(const nlohmann::json& config);
        virtual void destroy();
        virtual void saveCaches(const nlohmann::json& config);

        virtual FilterOutput apply(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed);

    };
}
