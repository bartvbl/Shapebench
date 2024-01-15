
#include "OpenGLDebugRenderer.h"
#include "utils/gl/GeometryBuffer.h"
#include "utils/gl/ShaderLoader.h"
#include "utils/gl/GLUtils.h"
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <shapeDescriptor/types/float3.h>

class OpenGLBatchImplementation : public JPH::RefTargetVirtual
{
public:
    explicit OpenGLBatchImplementation(GeometryBuffer _buffer) : buffer(_buffer) {  }

    virtual void AddRef() override {
        ++mRefCount;
    }
    virtual void Release() override {
        if (--mRefCount == 0) {
            delete this;
        }
    }
    JPH::atomic<uint32_t> mRefCount = 0;
    GeometryBuffer buffer;
};

GeometryBuffer generateGeometryBuffer(const JPH::DebugRenderer::Vertex* vertices, uint32_t vertexCount, const uint32_t* indices, uint32_t indexCount) {
    GeometryBuffer buffer;

    glGenVertexArrays(1, &buffer.vaoID);
    glBindVertexArray(buffer.vaoID);

    glGenBuffers(1, &buffer.vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, buffer.vertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(ShapeDescriptor::cpu::float3), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &buffer.indexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.indexBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW);

    buffer.indexCount = indexCount;

    return buffer;
}


OpenGLDebugRenderer::OpenGLDebugRenderer() {
    window = GLinitialise();

    shader = loadShader("res/shaders", "phong");
    shader.use();
}

void OpenGLDebugRenderer::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) {

}

void OpenGLDebugRenderer::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor,
                                  JPH::DebugRenderer::ECastShadow inCastShadow) {

}

JPH::DebugRenderer::Batch
OpenGLDebugRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Triangle *inTriangles, int inTriangleCount) {
    std::vector<uint32_t> indices(inTriangleCount * 3);
    for(uint32_t i = 0; i < indices.size(); i++) {
        indices.at(i) = i;
    }
    std::vector<JPH::DebugRenderer::Vertex> vertices(3 * inTriangleCount);
    for(uint32_t i = 0; i < inTriangleCount; i++) {
        vertices.at(3 * i + 0) = inTriangles[i].mV[0];
        vertices.at(3 * i + 1) = inTriangles[i].mV[1];
        vertices.at(3 * i + 2) = inTriangles[i].mV[2];
    }
    return CreateTriangleBatch(vertices.data(), vertices.size(), indices.data(), indices.size());
}

JPH::DebugRenderer::Batch
OpenGLDebugRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Vertex *inVertices, int inVertexCount,
                                         const JPH::uint32 *inIndices, int inIndexCount) {
    GeometryBuffer buffer = generateGeometryBuffer(inVertices, inVertexCount, inIndices, inIndexCount);
    return new OpenGLBatchImplementation(buffer);
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
    while (!glfwWindowShouldClose(window))
    {
        // Clear colour and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int windowWidth;
        int windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        glfwSwapBuffers(window);
    }
}
