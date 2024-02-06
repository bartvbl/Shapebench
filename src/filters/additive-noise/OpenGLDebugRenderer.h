#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "Jolt/Jolt.h"
#include "Jolt/Renderer/DebugRenderer.h"
#include "utils/gl/Shader.h"
#include "GLFW/glfw3.h"

namespace ShapeBench {
    class OpenGLDebugRenderer : public JPH::DebugRenderer {
        Shader shader;
        GLFWwindow* window;
        ShapeDescriptor::cpu::float3 cameraPosition = {-4.74, -5.8, 0.34};
        ShapeDescriptor::cpu::float3 cameraOrientation = {0.62, 4.9, 0};
        int windowWidth;
        int windowHeight;
        std::mutex drawLock;

    public:
        OpenGLDebugRenderer();

        virtual void DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) override;
        virtual void DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor, ECastShadow inCastShadow) override;
        virtual Batch CreateTriangleBatch(const Triangle *inTriangles, int inTriangleCount) override;
        virtual Batch CreateTriangleBatch(const Vertex *inVertices, int inVertexCount, const JPH::uint32 *inIndices, int inIndexCount) override;
        virtual void DrawGeometry(JPH::RMat44Arg inModelMatrix, const JPH::AABox &inWorldSpaceBounds, float inLODScaleSq, JPH::ColorArg inModelColor, const GeometryRef &inGeometry, ECullMode inCullMode, ECastShadow inCastShadow, EDrawMode inDrawMode) override;
        virtual void DrawText3D(JPH::RVec3Arg inPosition, const JPH::string_view &inString, JPH::ColorArg inColor, float inHeight) override;

        void nextFrame();
        bool windowShouldClose();


        void destroy();
    };
}
