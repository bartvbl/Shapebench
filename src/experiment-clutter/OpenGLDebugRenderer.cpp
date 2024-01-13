
#include "OpenGLDebugRenderer.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>

OpenGLDebugRenderer::OpenGLDebugRenderer() {
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "Physics Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        printf("Failed to initialize OpenGL context\n");
        return;
    }

    // Successfully loaded OpenGL
    printf("Loaded OpenGL %d.%d\n", GLAD_VERSION_MAJOR(version), GLAD_VERSION_MINOR(version));
}

void OpenGLDebugRenderer::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) {

}

void
OpenGLDebugRenderer::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor,
                                  JPH::DebugRenderer::ECastShadow inCastShadow) {

}

JPH::DebugRenderer::Batch
OpenGLDebugRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Triangle *inTriangles, int inTriangleCount) {
    return JPH::DebugRenderer::Batch();
}

JPH::DebugRenderer::Batch
OpenGLDebugRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Vertex *inVertices, int inVertexCount,
                                         const JPH::uint32 *inIndices, int inIndexCount) {
    JPH::DebugRenderer::Batch batch;

    return batch;
}

void OpenGLDebugRenderer::DrawGeometry(JPH::RMat44Arg inModelMatrix, const JPH::AABox &inWorldSpaceBounds,
                                       float inLODScaleSq, JPH::ColorArg inModelColor,
                                       const JPH::DebugRenderer::GeometryRef &inGeometry,
                                       JPH::DebugRenderer::ECullMode inCullMode,
                                       JPH::DebugRenderer::ECastShadow inCastShadow,
                                       JPH::DebugRenderer::EDrawMode inDrawMode) {

}

void OpenGLDebugRenderer::DrawText3D(JPH::RVec3Arg inPosition, const std::string_view &inString, JPH::ColorArg inColor,
                                     float inHeight) {

}

void OpenGLDebugRenderer::nextFrame() {

}
