
#include "glad/gl.h"
#include "OpenGLDebugRenderer.h"
#include "utils/gl/GeometryBuffer.h"
#include "utils/gl/GLUtils.h"
#include "GLFW/glfw3.h"
#include "glm/gtc/matrix_inverse.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <shapeDescriptor/shapeDescriptor.h>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "utils/gl/ShaderLoader.h"
#include <mutex>

class OpenGLBatchImplementation : public JPH::RefTargetVirtual
{
public:
    explicit OpenGLBatchImplementation(ShapeBench::GeometryBuffer _buffer) : buffer(_buffer) {  }

    virtual void AddRef() override {
        ++mRefCount;
    }
    virtual void Release() override {
        if (--mRefCount == 0) {
            delete this;
        }
    }
    JPH::atomic<uint32_t> mRefCount = 0;
    ShapeBench::GeometryBuffer buffer;
};

ShapeBench::GeometryBuffer generateGeometryBuffer(const JPH::DebugRenderer::Vertex* vertices, uint32_t vertexCount, const uint32_t* indices, uint32_t indexCount) {
    ShapeBench::GeometryBuffer buffer;

    glGenVertexArrays(1, &buffer.vaoID);
    glBindVertexArray(buffer.vaoID);


    /*std::vector<ShapeDescriptor::cpu::float3> vertexList(indexCount);
    std::vector<ShapeDescriptor::cpu::float3> normals(indexCount);
    std::vector<uint32_t> newIndices(indexCount);

    for(uint32_t i = 0; i < indexCount; i += 3) {
        uint32_t index0 = indices[i + 0];
        uint32_t index1 = indices[i + 1];
        uint32_t index2 = indices[i + 2];

        ShapeDescriptor::cpu::float3 vertex0 = vertices[index0].;
    }*/


    glGenBuffers(1, &buffer.vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, buffer.vertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(JPH::DebugRenderer::Vertex), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(JPH::DebugRenderer::Vertex), nullptr);
    glEnableVertexAttribArray(0);

    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(JPH::DebugRenderer::Vertex), (void*)sizeof(JPH::Float3));
    glEnableVertexAttribArray(1);


    glGenBuffers(1, &buffer.indexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.indexBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(unsigned int), indices, GL_STATIC_DRAW);

    buffer.indexCount = indexCount;

    return buffer;
}

inline void printGLError() {
    int errorID = glGetError();

    if(errorID != GL_NO_ERROR) {
        std::string errorString;

        switch(errorID) {
            case GL_INVALID_ENUM:
                errorString = "GL_INVALID_ENUM";
                break;
            case GL_INVALID_OPERATION:
                errorString = "GL_INVALID_OPERATION";
                break;
            case GL_INVALID_FRAMEBUFFER_OPERATION:
                errorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
                break;
            case GL_OUT_OF_MEMORY:
                errorString = "GL_OUT_OF_MEMORY";
                break;
            case GL_STACK_UNDERFLOW:
                errorString = "GL_STACK_UNDERFLOW";
                break;
            case GL_STACK_OVERFLOW:
                errorString = "GL_STACK_OVERFLOW";
                break;
            default:
                errorString = "[Unknown error ID]";
                break;
        }

        fprintf(stderr, "An OpenGL error occurred (%i): %s.\n",
                errorID, errorString.c_str());
        throw std::runtime_error(errorString);
    }
}


ShapeBench::OpenGLDebugRenderer::OpenGLDebugRenderer() {
    window = GLinitialise(1920, 1080);
    printGLError(__FILE__, __LINE__);

    shader = ShapeBench::loadShader("../res/shaders", "phong");
    printGLError(__FILE__, __LINE__);
    shader.use();
    printGLError(__FILE__, __LINE__);

    Initialize();
    printGLError(__FILE__, __LINE__);
}

void ShapeBench::OpenGLDebugRenderer::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo, JPH::ColorArg inColor) {
    std::unique_lock<std::mutex> ensureUnique(drawLock);
    glfwMakeContextCurrent(window);
}

void ShapeBench::OpenGLDebugRenderer::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3, JPH::ColorArg inColor,
                                  JPH::DebugRenderer::ECastShadow inCastShadow) {
    std::unique_lock<std::mutex> ensureUnique(drawLock);
    glfwMakeContextCurrent(window);
}

glm::mat4 toGLMMatrix(const JPH::Mat44 &mat44) {
    glm::mat4 outMatrix(0.0);
    for(int column = 0; column < 4; column++) {
        JPH::Vec4 col = mat44.GetColumn4(column);
        outMatrix[column] = {col.GetX(), col.GetY(), col.GetZ(), col.GetW()};
    }
    return outMatrix;
}

JPH::DebugRenderer::Batch
ShapeBench::OpenGLDebugRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Triangle *inTriangles, int inTriangleCount) {
    std::unique_lock<std::mutex> ensureUnique(drawLock);
    glfwMakeContextCurrent(window);
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
ShapeBench::OpenGLDebugRenderer::CreateTriangleBatch(const JPH::DebugRenderer::Vertex *inVertices, int inVertexCount,
                                         const JPH::uint32 *inIndices, int inIndexCount) {
    GeometryBuffer buffer = generateGeometryBuffer(inVertices, inVertexCount, inIndices, inIndexCount);
    return new OpenGLBatchImplementation(buffer);
}

void ShapeBench::OpenGLDebugRenderer::DrawGeometry(JPH::RMat44Arg inModelMatrix, const JPH::AABox &inWorldSpaceBounds,
                                       float inLODScaleSq, JPH::ColorArg inModelColor,
                                       const JPH::DebugRenderer::GeometryRef &inGeometry,
                                       JPH::DebugRenderer::ECullMode inCullMode,
                                       JPH::DebugRenderer::ECastShadow inCastShadow,
                                       JPH::DebugRenderer::EDrawMode inDrawMode) {
    std::unique_lock<std::mutex> ensureUnique(drawLock);
    glfwMakeContextCurrent(window);
    shader.use();
    GeometryBuffer& buffer = reinterpret_cast<OpenGLBatchImplementation*>(inGeometry->mLODs.at(0).mTriangleBatch.GetPtr())->buffer;
    glBindVertexArray(buffer.vaoID);
    printGLError(nullptr, 0);

    float aspectRatio = float(windowWidth) / float(windowHeight);

    glm::mat4 projectionMatrix = glm::perspective<float>(glm::radians(90.0), aspectRatio, 0.01, 100);
    glm::mat4 viewMatrix =
          glm::rotate<float>(glm::mat4(1.0), cameraOrientation.x, glm::vec3(1, 0, 0))
        * glm::rotate<float>(glm::mat4(1.0), cameraOrientation.y, glm::vec3(0, 1, 0))
        * glm::rotate<float>(glm::mat4(1.0), cameraOrientation.z, glm::vec3(0, 0, 1))
        * glm::translate(glm::mat4(1.0), glm::vec3(cameraPosition.x, cameraPosition.y, cameraPosition.z));
    glm::mat4 modelMatrix = toGLMMatrix(inModelMatrix);

    glm::mat4 MV = viewMatrix * modelMatrix;
    glm::mat4 MVP = projectionMatrix * MV;
    glm::mat4 inMatrix = toGLMMatrix(inModelMatrix);

    glm::mat4 invTraMatrix = glm::inverseTranspose(inMatrix);
    glm::mat3 normalMatrix(1.0);
    normalMatrix[0] = {invTraMatrix[0].x, invTraMatrix[0].y, invTraMatrix[0].z};
    normalMatrix[1] = {invTraMatrix[1].x, invTraMatrix[1].y, invTraMatrix[1].z};
    normalMatrix[2] = {invTraMatrix[2].x, invTraMatrix[2].y, invTraMatrix[2].z};

    glm::vec4 lightPosition = modelMatrix * glm::vec4(0, 3, 0, 1);

    shader.setUniform(30, glm::value_ptr(MVP));
    shader.setUniform(31, glm::value_ptr(MV));
    shader.setUniformMat3(32, glm::value_ptr(normalMatrix));

    shader.setUniform(50, lightPosition.x, lightPosition.y, lightPosition.z);
    shader.setUniform(20, 1, 1, 1, 1);

    glDrawElements(GL_TRIANGLES, buffer.indexCount, GL_UNSIGNED_INT, nullptr);
    printGLError(__FILE__, __LINE__);
}

void ShapeBench::OpenGLDebugRenderer::DrawText3D(JPH::RVec3Arg inPosition, const std::string_view &inString, JPH::ColorArg inColor,
                                     float inHeight) {
    std::unique_lock<std::mutex> ensureUnique(drawLock);
    glfwMakeContextCurrent(window);
}

void ShapeBench::OpenGLDebugRenderer::nextFrame() {
    glEnable(GL_DEPTH_TEST);

    const float rotationSpeed = 0.01;
    const float movementSpeed = 0.1;
    if(glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        cameraOrientation.y += rotationSpeed;
    }
    if(glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        cameraOrientation.y -= rotationSpeed;
    }
    if(glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        cameraOrientation.x -= rotationSpeed;
    }
    if(glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        cameraOrientation.x += rotationSpeed;
    }

    float dx = 0;
    float dy = 0;

    if(glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        dx += 1;
    }
    if(glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        dx -= 1;
    }
    if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
        dy -= 1;
    }
    if(glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        dy += 1;
    }
    if(glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
        cameraPosition.y -= movementSpeed;
    }
    if(glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
        cameraPosition.y += movementSpeed;
    }

    if(glfwGetKey(window, GLFW_KEY_ESCAPE)) {
        glfwSetWindowShouldClose(window, true);
    }

    float angleYRadiansForward = cameraOrientation.y;
    float angleYRadiansSideways = cameraOrientation.y + float(M_PI / 2.0);

    cameraPosition.x += dy * std::sin(angleYRadiansForward) * movementSpeed;
    cameraPosition.z -= dy * std::cos(angleYRadiansForward) * movementSpeed;

    cameraPosition.x -= dx * std::sin(angleYRadiansSideways) * movementSpeed;;
    cameraPosition.z += dx * std::cos(angleYRadiansSideways) * movementSpeed;

    std::unique_lock<std::mutex> ensureUnique(drawLock);
    glfwMakeContextCurrent(window);
    glfwSwapBuffers(window);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    glViewport(0, 0, windowWidth, windowHeight);
    glfwPollEvents();
}

bool ShapeBench::OpenGLDebugRenderer::windowShouldClose() {
    return glfwWindowShouldClose(window);
}

void ShapeBench::OpenGLDebugRenderer::destroy() {
    glfwDestroyWindow(window);
}
