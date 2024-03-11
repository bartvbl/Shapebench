#include "glad/gl.h"
#include "OcclusionFilter.h"
#include "utils/gl/GLUtils.h"
#include "utils/gl/VAOGenerator.h"
#include "utils/gl/Shader.h"
#include "utils/gl/ShaderLoader.h"
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include "GLFW/glfw3.h"
#include "benchmarkCore/randomEngine.h"
#include <random>

ShapeBench::OccludedSceneGenerator::OccludedSceneGenerator() {

}

ShapeBench::OccludedSceneGenerator::~OccludedSceneGenerator() {
    destroy();
}

// Mesh is assumed to be fit inside unit sphere
ShapeBench::SubtractiveNoiseOutput ShapeBench::OccludedSceneGenerator::computeOccludedMesh(const nlohmann::json& config, ShapeBench::FilteredMeshPair &scene, uint64_t seed) {
    ShapeBench::randomEngine randomEngine(seed);
    ShapeBench::SubtractiveNoiseOutput output;

    std::vector<unsigned char> localFramebufferCopy(3 * offscreenTextureWidth * offscreenTextureHeight);
    const uint32_t totalVertexCount = scene.filteredSampleMesh.vertexCount + scene.filteredAdditiveNoise.vertexCount;
    std::vector<ShapeDescriptor::cpu::float3> vertexColours(totalVertexCount);
    for (unsigned int triangle = 0; triangle < totalVertexCount / 3; triangle++) {
        float red = float((triangle & 0x00FF0000U) >> 16U) / 255.0f;
        float green = float((triangle & 0x0000FF00U) >> 8U) / 255.0f;
        float blue = float((triangle & 0x000000FFU) >> 0U) / 255.0f;

        vertexColours.at(3 * triangle + 0) = {red, green, blue};
        vertexColours.at(3 * triangle + 1) = {red, green, blue};
        vertexColours.at(3 * triangle + 2) = {red, green, blue};
    }

    std::uniform_real_distribution<float> distribution(0, 1);
    float yaw = float(distribution(randomEngine) * 2.0 * M_PI);
    float pitch = float((distribution(randomEngine) - 0.5) * M_PI);
    float roll = float(distribution(randomEngine) * 2.0 * M_PI);

    nlohmann::json entry;
    entry["subtractive-noise-pitch"] = pitch;
    entry["subtractive-noise-yaw"] = yaw;
    entry["subtractive-noise-roll"] = roll;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        output.metadata.push_back(entry);
    }

    float nearPlaneDistance = config.at("filterSettings").at("subtractiveNoise").at("nearPlaneDistance");
    float farPlaneDistance = config.at("filterSettings").at("subtractiveNoise").at("farPlaneDistance");
    float fovy = config.at("filterSettings").at("subtractiveNoise").at("fovYAngleRadians");
    float objectDistanceFromCamera = config.at("filterSettings").at("subtractiveNoise").at("objectDistanceFromCamera");

    glm::mat4 objectProjection = glm::perspective(fovy, (float) offscreenTextureWidth / (float) offscreenTextureHeight, nearPlaneDistance, farPlaneDistance);
    glm::mat4 positionTransformation = glm::translate(glm::mat4(1.0), glm::vec3(0, 0, -objectDistanceFromCamera));
    positionTransformation *= glm::rotate(glm::mat4(1.0), roll, glm::vec3(0, 0, 1));
    positionTransformation *= glm::rotate(glm::mat4(1.0), yaw, glm::vec3(1, 0, 0));
    positionTransformation *= glm::rotate(glm::mat4(1.0), pitch, glm::vec3(0, 1, 0));
    glm::mat4 objectTransformation = objectProjection * positionTransformation;

    {
        std::unique_lock<std::mutex> filterLock{occlusionFilterLock};
        glfwMakeContextCurrent(window);

        // Handle other events
        glfwPollEvents();
        GeometryBuffer sampleMeshBuffers = generateVertexArray(scene.filteredSampleMesh.vertices,
                                                               scene.filteredSampleMesh.normals,
                                                               vertexColours.data(),
                                                               scene.filteredSampleMesh.vertexCount);
        GeometryBuffer additiveNoiseBuffers = generateVertexArray(scene.filteredAdditiveNoise.vertices,
                                                                  scene.filteredAdditiveNoise.normals,
                                                                  vertexColours.data() +
                                                                  scene.filteredSampleMesh.vertexCount,
                                                                  scene.filteredAdditiveNoise.vertexCount);
        objectIDShader.use();
        glUniformMatrix4fv(16, 1, false, glm::value_ptr(objectTransformation));

        glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
        glViewport(0, 0, offscreenTextureWidth, offscreenTextureHeight);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glEnable(GL_DEPTH_TEST);
        glBindVertexArray(sampleMeshBuffers.vaoID);
        glDrawElements(GL_TRIANGLES, scene.filteredSampleMesh.vertexCount, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(additiveNoiseBuffers.vaoID);
        glDrawElements(GL_TRIANGLES, scene.filteredAdditiveNoise.vertexCount, GL_UNSIGNED_INT, nullptr);

        // Do visibility testing

        glBindTexture(GL_TEXTURE_2D, renderTextureID);
        glReadPixels(0, 0, offscreenTextureWidth, offscreenTextureHeight, GL_RGB, GL_UNSIGNED_BYTE,
                     localFramebufferCopy.data());

        // Draw visible version

        int windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);

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

        sampleMeshBuffers.destroy();
        additiveNoiseBuffers.destroy();

        // Flip buffers
        glfwSwapBuffers(window);

        // Allow other threads to take control of the context
        glfwMakeContextCurrent(nullptr);
    }

    // We are now done with the OpenGL part, so we can allow other threads to run that part

    std::vector<bool> triangleAppearsInImage(totalVertexCount / 3);

    for (size_t pixel = 0; pixel < offscreenTextureWidth * offscreenTextureHeight; pixel++) {
        unsigned int triangleIndex =
                (((unsigned int) localFramebufferCopy.at(3 * pixel + 0)) << 16U) |
                (((unsigned int) localFramebufferCopy.at(3 * pixel + 1)) << 8U) |
                (((unsigned int) localFramebufferCopy.at(3 * pixel + 2)) << 0U);

        // Test if pixel is background
        if (triangleIndex == 0x00FFFFFF) {
            continue;
        }

        triangleAppearsInImage.at(triangleIndex) = true;
    }

    uint32_t visibleSampleMeshVertexCount = 0;
    uint32_t visibleAdditiveNoiseVertexCount = 0;
    for (unsigned int triangle = 0; triangle < triangleAppearsInImage.size(); triangle++) {
        if (triangleAppearsInImage.at(triangle)) {
            if (3 * triangle < scene.filteredSampleMesh.vertexCount) {
                visibleSampleMeshVertexCount += 3;
            } else {
                visibleAdditiveNoiseVertexCount += 3;
            }
        }
    }

    //std::cout << "Visible sample: " << visibleSampleMeshVertexCount << ", additive: " << visibleAdditiveNoiseVertexCount << ", total: " << totalVertexCount << std::endl;
    ShapeDescriptor::cpu::Mesh occludedSampleMesh(visibleSampleMeshVertexCount);
    ShapeDescriptor::cpu::Mesh occludedAdditiveNoiseMesh(visibleAdditiveNoiseVertexCount);

    uint32_t nextSampleMeshTargetIndex = 0;
    uint32_t nextAdditiveNoiseMeshTargetIndex = 0;
    for(uint32_t i = 0; i < scene.mappedVertexIncluded.size(); i++) {
        scene.mappedVertexIncluded.at(i) = false;
    }

    for (unsigned int triangle = 0; triangle < triangleAppearsInImage.size(); triangle++) {
        if (triangleAppearsInImage.at(triangle)) {
            if (3 * triangle < scene.filteredSampleMesh.vertexCount) {
                uint32_t targetIndex = nextSampleMeshTargetIndex;
                nextSampleMeshTargetIndex += 3;

                occludedSampleMesh.vertices[targetIndex + 0] = scene.filteredSampleMesh.vertices[3 * triangle + 0];
                occludedSampleMesh.vertices[targetIndex + 1] = scene.filteredSampleMesh.vertices[3 * triangle + 1];
                occludedSampleMesh.vertices[targetIndex + 2] = scene.filteredSampleMesh.vertices[3 * triangle + 2];

                occludedSampleMesh.normals[targetIndex + 0] = scene.filteredSampleMesh.normals[3 * triangle + 0];
                occludedSampleMesh.normals[targetIndex + 1] = scene.filteredSampleMesh.normals[3 * triangle + 1];
                occludedSampleMesh.normals[targetIndex + 2] = scene.filteredSampleMesh.normals[3 * triangle + 2];

                // We just need to make sure that the vertex position itself is on a triangle that exists
                // There is no need to also compare the normal
                for(uint32_t i = 0; i < scene.mappedVertexIncluded.size(); i++) {
                    ShapeDescriptor::cpu::float3 mappedVertex = scene.mappedReferenceVertices.at(i).vertex;
                    if(mappedVertex == scene.filteredSampleMesh.vertices[3 * triangle + 0]
                    || mappedVertex == scene.filteredSampleMesh.vertices[3 * triangle + 1]
                    || mappedVertex == scene.filteredSampleMesh.vertices[3 * triangle + 2]) {
                        scene.mappedVertexIncluded.at(i) = true;
                    }
                }
            } else {
                uint32_t baseIndex = 3 * triangle - scene.filteredSampleMesh.vertexCount;

                uint32_t targetIndex = nextAdditiveNoiseMeshTargetIndex;
                nextAdditiveNoiseMeshTargetIndex += 3;

                occludedAdditiveNoiseMesh.vertices[targetIndex + 0] = scene.filteredAdditiveNoise.vertices[baseIndex + 0];
                occludedAdditiveNoiseMesh.vertices[targetIndex + 1] = scene.filteredAdditiveNoise.vertices[baseIndex + 1];
                occludedAdditiveNoiseMesh.vertices[targetIndex + 2] = scene.filteredAdditiveNoise.vertices[baseIndex + 2];

                occludedAdditiveNoiseMesh.normals[targetIndex + 0] = scene.filteredAdditiveNoise.normals[baseIndex + 0];
                occludedAdditiveNoiseMesh.normals[targetIndex + 1] = scene.filteredAdditiveNoise.normals[baseIndex + 1];
                occludedAdditiveNoiseMesh.normals[targetIndex + 2] = scene.filteredAdditiveNoise.normals[baseIndex + 2];
            }
        }
    }

    //std::cout << "Added to mesh sample: " << nextSampleMeshTargetIndex << ", additive: " << nextAdditiveNoiseMeshTargetIndex << ", total: " << totalVertexCount << std::endl;

    ShapeDescriptor::free(scene.filteredSampleMesh);
    ShapeDescriptor::free(scene.filteredAdditiveNoise);
    scene.filteredSampleMesh = occludedSampleMesh;
    scene.filteredAdditiveNoise = occludedAdditiveNoiseMesh;

    //std::cout << "Output count: " << scene.filteredSampleMesh.vertexCount << ", " << scene.filteredAdditiveNoise.vertexCount << std::endl;

    return output;
}

void ShapeBench::OccludedSceneGenerator::destroy() {
    if(isDestroyed) {
        return;
    }
    if(!isCreated) {
        return;
    }
    screenQuadVAO.destroy();
    glDeleteFramebuffers(1, &frameBufferID);
    glDeleteRenderbuffers(1, &renderBufferID);
    glDeleteTextures(1, &renderTextureID);
    glfwDestroyWindow(window);
    isDestroyed = true;
}

void ShapeBench::OccludedSceneGenerator::init(uint32_t visibilityImageWidth, uint32_t visibilityImageHeight) {
    offscreenTextureWidth = visibilityImageWidth;
    offscreenTextureHeight = visibilityImageHeight;

    window = GLinitialise(700, 700);
    objectIDShader = loadShader("../res/shaders/", "objectIDShader");
    fullscreenQuadShader = loadShader("../res/shaders/", "fullscreenquad");

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
    glGenFramebuffers(1, &frameBufferID);
    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);

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


    glfwMakeContextCurrent(nullptr);
    isCreated = true;
}
