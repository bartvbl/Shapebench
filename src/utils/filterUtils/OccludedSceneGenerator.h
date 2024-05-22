#pragma once

#include <mutex>
#include "portablegl.h"
#include "filters/FilteredMeshPair.h"
#include "utils/gl/GeometryBuffer.h"

namespace ShapeBench {
    struct OcclusionRendererSettings {
        float nearPlaneDistance = 0.001;
        float farPlaneDistance = 100.0;
        float fovy = 1.552;
        float objectDistanceFromCamera = 20;
        float roll = 0;
        float yaw = 0;
        float pitch = 0;
        float rgbdDepthCutoffFactor = 0.2;
    };

    struct SoftwareContext {
        glContext* context;
        uint32_t* frameBuffer;
    };

    class OccludedSceneGenerator {
        ShapeBench::SoftwareContext window;
        unsigned int GeometryIDShader = 0;

        bool isDestroyed = false;
        bool isCreated = false;
        uint32_t offscreenTextureWidth = 0;
        uint32_t offscreenTextureHeight = 0;
        std::mutex occlusionFilterLock;

        void renderSceneToOffscreenBuffer(ShapeBench::FilteredMeshPair& scene, OcclusionRendererSettings settings, ShapeDescriptor::cpu::float3* vertexColours, uint8_t* outFrameBuffer, float* outDepthBuffer = nullptr);

        public:
        explicit OccludedSceneGenerator();
        ~OccludedSceneGenerator();
        void computeOccludedMesh(ShapeBench::OcclusionRendererSettings settings, ShapeBench::FilteredMeshPair &scene);
        ShapeDescriptor::cpu::Mesh computeRGBDMesh(ShapeBench::OcclusionRendererSettings settings, ShapeBench::FilteredMeshPair& scene, float& out_distanceBetweenPixels);
        static glm::mat4 computeTransformationMatrix(float pitch, float roll, float yaw, float objectDistanceFromCamera);
        void init(uint32_t visibilityImageWidth, uint32_t visibilityImageHeight);
        void destroy();
    };
}
