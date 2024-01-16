#include "OcclusionGenerator.h"
#include "glad/gl.h"
#include "utils/gl/GLUtils.h"
#include "utils/gl/VAOGenerator.h"
#include "utils/gl/Shader.h"
#include "utils/gl/ShaderLoader.h"
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include "GLFW/glfw3.h"
#include <random>

OccludedSceneGenerator::OccludedSceneGenerator() {
    init();
}

OccludedSceneGenerator::~OccludedSceneGenerator() {
    destroy();
}

ShapeDescriptor::cpu::Mesh
OccludedSceneGenerator::computeOccludedMesh(const ShapeDescriptor::cpu::Mesh mesh, uint64_t seed) {
    // Handle other events
    glfwPollEvents();



    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    ShapeDescriptor::cpu::Mesh loadedMesh = ShapeDescriptor::loadMesh(haystackFiles.at(i));

    ShapeDescriptor::cpu::float3 averageSum = {0, 0, 0};
    for(unsigned int vertex = 0; vertex < loadedMesh.vertexCount; vertex++) {
        averageSum += loadedMesh.vertices[vertex];
    }
    averageSum.x /= float(loadedMesh.vertexCount);
    averageSum.y /= float(loadedMesh.vertexCount);
    averageSum.z /= float(loadedMesh.vertexCount);

    std::vector<ShapeDescriptor::cpu::float3> vertexColours(loadedMesh.vertexCount);
    for(unsigned int triangle = 0; triangle < loadedMesh.vertexCount / 3; triangle++) {
        float red = float((triangle & 0x00FF0000U) >> 16U) / 255.0f;
        float green = float((triangle & 0x0000FF00U) >> 8U) / 255.0f;
        float blue = float((triangle & 0x000000FFU) >> 0U) / 255.0f;

        vertexColours.at(3 * triangle + 0) = {red, green, blue};
        vertexColours.at(3 * triangle + 1) = {red, green, blue};
        vertexColours.at(3 * triangle + 2) = {red, green, blue};
    }

    GeometryBuffer buffers = generateVertexArray(loadedMesh.vertices, loadedMesh.normals, vertexColours.data(), loadedMesh.vertexCount);

    objectIDShader.use();

    float yaw = float(distribution(generator) * 2.0 * M_PI);
    float pitch = float((distribution(generator) - 0.5) * M_PI);
    float roll = float(distribution(generator) * 2.0 * M_PI);

    glm::mat4 objectProjection = glm::perspective(1.57f, (float) windowWidth / (float) windowHeight, 1.0f, 10000.0f);
    glm::mat4 positionTransformation = glm::translate(glm::mat4(1.0), glm::vec3(0, 0, -200.0f));
    positionTransformation *= glm::rotate(glm::mat4(1.0), roll, glm::vec3(0, 0, 1));
    positionTransformation *= glm::rotate(glm::mat4(1.0), yaw, glm::vec3(1, 0, 0));
    positionTransformation *= glm::rotate(glm::mat4(1.0), pitch, glm::vec3(0, 1, 0));
    positionTransformation *= glm::translate(glm::mat4(1.0), -glm::vec3(averageSum.x, averageSum.y, averageSum.z));
    glm::mat4 objectTransformation = objectProjection * positionTransformation;
    glUniformMatrix4fv(16, 1, false, glm::value_ptr(objectTransformation));

    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    glViewport(0, 0, offscreenTextureWidth, offscreenTextureHeight);
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBindVertexArray(buffers.VAOID);
    glEnable(GL_DEPTH_TEST);
    glDrawElements(GL_TRIANGLES, loadedMesh.vertexCount, GL_UNSIGNED_INT, nullptr);

    // Do visibility testing

    glBindTexture(GL_TEXTURE_2D, renderTextureID);
    glReadPixels(0, 0, offscreenTextureWidth, offscreenTextureHeight, GL_RGB, GL_UNSIGNED_BYTE, localFramebufferCopy.data());


    std::vector<bool> triangleAppearsInImage(loadedMesh.vertexCount / 3);

    for(size_t pixel = 0; pixel < offscreenTextureWidth * offscreenTextureHeight; pixel++) {
        unsigned int triangleIndex =
                (((unsigned int) localFramebufferCopy.at(3 * pixel + 0)) << 16U) |
                (((unsigned int) localFramebufferCopy.at(3 * pixel + 1)) << 8U) |
                (((unsigned int) localFramebufferCopy.at(3 * pixel + 2)) << 0U);

        // Test if pixel is background
        if(triangleIndex == 0x00FFFFFF) {
            continue;
        }

        triangleAppearsInImage.at(triangleIndex) = true;
    }

    ShapeDescriptor::cpu::Mesh outMesh(loadedMesh.vertexCount);

    unsigned int visibleVertexCount = 0;
    for(unsigned int triangle = 0; triangle < triangleAppearsInImage.size(); triangle++) {
        if(triangleAppearsInImage.at(triangle)) {
            outMesh.vertices[visibleVertexCount + 0] = loadedMesh.vertices[3 * triangle + 0];
            outMesh.vertices[visibleVertexCount + 1] = loadedMesh.vertices[3 * triangle + 1];
            outMesh.vertices[visibleVertexCount + 2] = loadedMesh.vertices[3 * triangle + 2];

            outMesh.normals[visibleVertexCount + 0] = loadedMesh.normals[3 * triangle + 0];
            outMesh.normals[visibleVertexCount + 1] = loadedMesh.normals[3 * triangle + 1];
            outMesh.normals[visibleVertexCount + 2] = loadedMesh.normals[3 * triangle + 2];

            visibleVertexCount += 3;
        }
    }

    outMesh.vertexCount = visibleVertexCount;

    // Draw visible version

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    fullscreenQuadShader.use();

    glBindVertexArray(screenQuadVAO.vaoID);
    glDisable(GL_DEPTH_TEST);
    glBindTextureUnit(0, renderTextureID);

    glm::mat4 fullscreenProjection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

    glUniformMatrix4fv(16, 1, false, glm::value_ptr(fullscreenProjection));

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

    buffers.destroy();

    // Flip buffers
    glfwSwapBuffers(window);
}

void OccludedSceneGenerator::destroy() {
    screenQuadVAO.destroy();
    glDeleteFramebuffers(1, &frameBufferID);
    glDeleteRenderbuffers(1, &renderBufferID);
    glDeleteTextures(1, &renderTextureID);
    glfwDestroyWindow(window);
}

void OccludedSceneGenerator::init() {
    window = GLinitialise();
    objectIDShader = loadShader("res/shaders/", "objectIDShader");
    fullscreenQuadShader = loadShader("res/shaders/", "fullscreenquad");

    std::vector<ShapeDescriptor::cpu::float3> screenQuadVertices = {{0, 0, 0},
                                                                    {1, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 1, 0}};
    std::vector<ShapeDescriptor::cpu::float3> screenQuadTexCoords ={{0, 0, 0},
                                                                    {1, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 0, 0},
                                                                    {1, 1, 0},
                                                                    {0, 1, 0}};
    std::vector<ShapeDescriptor::cpu::float3> screenQuadColours =  {{1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1},
                                                                    {1, 1, 1}};
    screenQuadVAO = generateVertexArray(screenQuadVertices.data(), screenQuadTexCoords.data(), screenQuadColours.data(), 6);

    // Create offscreen renderer

    const unsigned int offscreenTextureWidth = 4 * 7680;
    const unsigned int offscreenTextureHeight = 4 * 4230;

    glGenFramebuffers(1, &frameBufferID);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    glEnable(GL_DEPTH_TEST);

    glGenTextures(1, &renderTextureID);
    glBindTexture(GL_TEXTURE_2D, renderTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, offscreenTextureWidth, offscreenTextureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, renderTextureID, 0);

    glGenRenderbuffers(1, &renderBufferID);
    glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, offscreenTextureWidth, offscreenTextureHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID);

    localFramebufferCopy.resize(3 * offscreenTextureWidth * offscreenTextureHeight);
}
