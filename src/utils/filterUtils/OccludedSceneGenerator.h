#pragma once

#include <mutex>
#include "GLFW/glfw3.h"
#include "utils/gl/Shader.h"
#include "utils/gl/GeometryBuffer.h"
#include "filters/Filter.h"

namespace ShapeBench {
    struct OcclusionRendererSettings {
        float nearPlaneDistance = 0.001;
        float farPlaneDistance = 100.0;
        float fovy = 1.552;
        float objectDistanceFromCamera = 20;
        float roll = 0;
        float yaw = 0;
        float pitch = 0;
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
        void computeOccludedMesh(ShapeBench::OcclusionRendererSettings settings, ShapeBench::FilteredMeshPair &scene);
        void init(uint32_t visibilityImageWidth, uint32_t visibilityImageHeight);
        void destroy();
    };
}
