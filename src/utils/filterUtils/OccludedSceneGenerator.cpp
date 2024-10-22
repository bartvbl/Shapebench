//
// Created by bart on 03/04/24.
//

#include "glad/gl.h"
#include "OccludedSceneGenerator.h"
#include "benchmarkCore/randomEngine.h"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "utils/gl/VAOGenerator.h"
#include "glm/gtc/type_ptr.hpp"
#include "utils/gl/GLUtils.h"
#include "utils/gl/ShaderLoader.h"
#include <glm/glm.hpp>


ShapeBench::OccludedSceneGenerator::OccludedSceneGenerator() {

}

ShapeBench::OccludedSceneGenerator::~OccludedSceneGenerator() {
    destroy();
}

void ShapeBench::OccludedSceneGenerator::renderSceneToOffscreenBuffer(ShapeBench::FilteredMeshPair& scene, OcclusionRendererSettings settings, ShapeDescriptor::cpu::float3* vertexColours, uint8_t* outFrameBuffer, float* outDepthBuffer) {
    glm::mat4 objectProjection = glm::perspective(settings.fovy, (float) offscreenTextureWidth / (float) offscreenTextureHeight, settings.nearPlaneDistance, settings.farPlaneDistance);
    glm::mat4 positionTransformation = computeTransformationMatrix(settings.pitch, settings.roll, settings.yaw, settings.objectDistanceFromCamera);
    glm::mat4 objectTransformation = objectProjection * positionTransformation;

    std::unique_lock<std::mutex> filterLock{occlusionFilterLock};
    glfwMakeContextCurrent(window);

    // Handle other events
    glfwPollEvents();
    GeometryBuffer sampleMeshBuffers = generateVertexArray(scene.filteredSampleMesh.vertices,
                                                           scene.filteredSampleMesh.normals,
                                                           vertexColours,
                                                           scene.filteredSampleMesh.vertexCount);
    GeometryBuffer additiveNoiseBuffers = generateVertexArray(scene.filteredAdditiveNoise.vertices,
                                                              scene.filteredAdditiveNoise.normals,
                                                              vertexColours == nullptr ? nullptr : vertexColours + scene.filteredSampleMesh.vertexCount,
                                                              scene.filteredAdditiveNoise.vertexCount);
    objectIDShader.use();
    glUniformMatrix4fv(16, 1, false, glm::value_ptr(objectTransformation));

    glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
    glViewport(0, 0, offscreenTextureWidth, offscreenTextureHeight);
    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ShapeBench::printGLError(__FILE__, __LINE__);

    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(sampleMeshBuffers.vaoID);
    glDrawElements(GL_TRIANGLES, scene.filteredSampleMesh.vertexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(additiveNoiseBuffers.vaoID);
    glDrawElements(GL_TRIANGLES, scene.filteredAdditiveNoise.vertexCount, GL_UNSIGNED_INT, nullptr);
    ShapeBench::printGLError(__FILE__, __LINE__);

    // Do visibility testing

    if(outFrameBuffer != nullptr) {
        glBindTexture(GL_TEXTURE_2D, renderTextureID);
        glReadPixels(0, 0, offscreenTextureWidth, offscreenTextureHeight, GL_RGB, GL_UNSIGNED_BYTE, outFrameBuffer);
    }
    if(outDepthBuffer != nullptr) {
        glBindTexture(GL_TEXTURE_2D, depthTextureID);
        glReadPixels (0, 0, offscreenTextureWidth, offscreenTextureHeight, GL_DEPTH_COMPONENT, GL_FLOAT, outDepthBuffer);
    }
    ShapeBench::printGLError(__FILE__, __LINE__);

    // Draw visible version

    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.5, 0.5, 0.5, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    ShapeBench::printGLError(__FILE__, __LINE__);

    fullscreenQuadShader.use();

    glBindVertexArray(screenQuadVAO.vaoID);
    ShapeBench::printGLError(__FILE__, __LINE__);
    glDisable(GL_DEPTH_TEST);
    if(outFrameBuffer != nullptr) {
        glBindTextureUnit(0, renderTextureID);
        glUniform1i(25, 0);
    } else if(outDepthBuffer != nullptr) {
        glBindTextureUnit(0, depthTextureID);
        glUniform1i(25, 1);
    }
    ShapeBench::printGLError(__FILE__, __LINE__);

    glm::mat4 fullscreenProjection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

    glUniformMatrix4fv(16, 1, false, glm::value_ptr(fullscreenProjection));
    ShapeBench::printGLError(__FILE__, __LINE__);

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    ShapeBench::printGLError(__FILE__, __LINE__);

    sampleMeshBuffers.destroy();
    additiveNoiseBuffers.destroy();

    // Flip buffers
    glfwSwapBuffers(window);

    // Allow other threads to take control of the context
    glfwMakeContextCurrent(nullptr);
}

// Mesh is assumed to be fit inside unit sphere
void ShapeBench::OccludedSceneGenerator::computeOccludedMesh(ShapeBench::OcclusionRendererSettings settings, ShapeBench::FilteredMeshPair &scene) {
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

    renderSceneToOffscreenBuffer(scene, settings, vertexColours.data(), localFramebufferCopy.data());

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
                    bool isVertex0 = mappedVertex == occludedSampleMesh.vertices[targetIndex + 0];
                    bool isVertex1 = mappedVertex == occludedSampleMesh.vertices[targetIndex + 1];
                    bool isVertex2 = mappedVertex == occludedSampleMesh.vertices[targetIndex + 2];
                    if(isVertex0 || isVertex1 || isVertex2) {
                        scene.mappedVertexIncluded.at(i) = true;
                        if(isVertex0) {
                            scene.mappedReferenceVertexIndices.at(i) = targetIndex + 0;
                        }
                        if(isVertex1) {
                            scene.mappedReferenceVertexIndices.at(i) = targetIndex + 1;
                        }
                        if(isVertex2) {
                            scene.mappedReferenceVertexIndices.at(i) = targetIndex + 2;
                        }
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
}

ShapeDescriptor::cpu::Mesh ShapeBench::OccludedSceneGenerator::computeRGBDMesh(ShapeBench::OcclusionRendererSettings settings, ShapeBench::FilteredMeshPair &scene, float& out_distanceBetweenPixels) {

    std::vector<float> depthBuffer(offscreenTextureWidth * offscreenTextureHeight);

    renderSceneToOffscreenBuffer(scene, settings, nullptr, nullptr, depthBuffer.data());

    std::unique_lock<std::mutex> filterLock{occlusionFilterLock};

    glm::mat4 objectProjection = glm::perspective(settings.fovy, (float) offscreenTextureWidth / (float) offscreenTextureHeight, settings.nearPlaneDistance, settings.farPlaneDistance);
    glm::mat4 positionTransformation = computeTransformationMatrix(settings.pitch, settings.roll, settings.yaw, settings.objectDistanceFromCamera);
    glm::mat4 objectTransformation = objectProjection * positionTransformation;

    ShapeDescriptor::cpu::PointCloud cloud(offscreenTextureWidth * offscreenTextureHeight);
    ShapeDescriptor::cpu::Mesh rgbdMesh(6 * offscreenTextureWidth * offscreenTextureHeight);

    for(int col = 0; col < offscreenTextureWidth; col++) {
        for(int row = 0; row < offscreenTextureHeight; row++) {
            float xCoord = 2.0f * (float(col) / float(offscreenTextureWidth)) - 1.0f;
            float yCoord = 2.0f * (float(row) / float(offscreenTextureHeight)) - 1.0f;
            float zCoord = depthBuffer.at(row * offscreenTextureWidth + col);
            glm::vec4 projected(xCoord, yCoord, zCoord, 1);
            glm::vec3 unprojected = glm::unProject({col, row, zCoord}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
            cloud.vertices[row * offscreenTextureWidth + col] = ShapeDescriptor::cpu::float3(unprojected.x, unprojected.y, unprojected.z);
        }
    }

    glm::vec4 depthCoordinate = objectTransformation * glm::vec4(0, 0, 0, 1);
    depthCoordinate /= (depthCoordinate.w != 0 ? depthCoordinate.w : 1);

    glm::vec3 unprojectedSample1 = glm::unProject({0, 0, depthCoordinate.z}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
    glm::vec3 unprojectedSample2 = glm::unProject({1, 0, depthCoordinate.z}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
    out_distanceBetweenPixels = length(unprojectedSample1 - unprojectedSample2);
    float cutoffDistance = settings.rgbdDepthCutoffFactor * out_distanceBetweenPixels;

    uint32_t nextBaseIndex = 0;
    for(int col = 0; col < offscreenTextureWidth - 1; col++) {
        for(int row = 0; row < offscreenTextureHeight - 1; row++) {
            ShapeDescriptor::cpu::float3 vertex0 = cloud.vertices[(row) * offscreenTextureWidth + (col)];
            ShapeDescriptor::cpu::float3 vertex1 = cloud.vertices[(row) * offscreenTextureWidth + (col + 1)];
            ShapeDescriptor::cpu::float3 vertex2 = cloud.vertices[(row + 1) * offscreenTextureWidth + (col + 1)];
            ShapeDescriptor::cpu::float3 vertex3 = cloud.vertices[(row + 1) * offscreenTextureWidth + (col)];

            glm::vec3 backgroundVertex0GLM = glm::unProject({col, row, 1}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
            glm::vec3 backgroundVertex1GLM = glm::unProject({col + 1, row, 1}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
            glm::vec3 backgroundVertex2GLM = glm::unProject({col + 1, row + 1, 1}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
            glm::vec3 backgroundVertex3GLM = glm::unProject({col, row + 1, 1}, positionTransformation, objectProjection, glm::vec4(0.0f, 0.0f, offscreenTextureWidth, offscreenTextureHeight));
            ShapeDescriptor::cpu::float3 backgroundVertex0(backgroundVertex0GLM.x, backgroundVertex0GLM.y, backgroundVertex0GLM.z);
            ShapeDescriptor::cpu::float3 backgroundVertex1(backgroundVertex1GLM.x, backgroundVertex1GLM.y, backgroundVertex1GLM.z);
            ShapeDescriptor::cpu::float3 backgroundVertex2(backgroundVertex2GLM.x, backgroundVertex2GLM.y, backgroundVertex2GLM.z);
            ShapeDescriptor::cpu::float3 backgroundVertex3(backgroundVertex3GLM.x, backgroundVertex3GLM.y, backgroundVertex3GLM.z);

            float distance20 = length(vertex0 - vertex2);
            float distance13 = length(vertex1 - vertex3);

            bool flipTriangle = distance20 > distance13;
            if(flipTriangle) {
                ShapeDescriptor::cpu::float3 temp = vertex0;
                vertex0 = vertex1;
                vertex1 = vertex2;
                vertex2 = vertex3;
                vertex3 = temp;

                temp = backgroundVertex0;
                backgroundVertex0 = backgroundVertex1;
                backgroundVertex1 = backgroundVertex2;
                backgroundVertex2 = backgroundVertex3;
                backgroundVertex3 = temp;

                std::swap(distance20, distance13);
            }

            ShapeDescriptor::cpu::float3 normal1 = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);
            ShapeDescriptor::cpu::float3 normal2 = ShapeDescriptor::computeTriangleNormal(vertex0, vertex2, vertex3);

            bool notBackgroundVertex0 = vertex0 != backgroundVertex0;
            bool notBackgroundVertex1 = vertex1 != backgroundVertex1;
            bool notBackgroundVertex2 = vertex2 != backgroundVertex2;
            bool notBackgroundVertex3 = vertex3 != backgroundVertex3;

            float distance01 = length(vertex1 - vertex0);
            float distance12 = length(vertex2 - vertex1);
            float distance23 = length(vertex3 - vertex2);
            float distance30 = length(vertex0 - vertex3);

            bool triangle0Valid = std::max(distance01, std::max(distance12, distance20)) <= cutoffDistance;
            bool triangle1Valid = std::max(distance20, std::max(distance23, distance30)) <= cutoffDistance;

            if(notBackgroundVertex0 && notBackgroundVertex1 && notBackgroundVertex2 && triangle0Valid) {
                rgbdMesh.vertices[nextBaseIndex + 0] = vertex0;
                rgbdMesh.vertices[nextBaseIndex + 1] = vertex1;
                rgbdMesh.vertices[nextBaseIndex + 2] = vertex2;
                rgbdMesh.normals[nextBaseIndex + 0] = normal1;
                rgbdMesh.normals[nextBaseIndex + 1] = normal1;
                rgbdMesh.normals[nextBaseIndex + 2] = normal1;
                nextBaseIndex += 3;
            }
            if(notBackgroundVertex0 && notBackgroundVertex2 && notBackgroundVertex3 && triangle1Valid) {
                rgbdMesh.vertices[nextBaseIndex + 0] = vertex0;
                rgbdMesh.vertices[nextBaseIndex + 1] = vertex2;
                rgbdMesh.vertices[nextBaseIndex + 2] = vertex3;
                rgbdMesh.normals[nextBaseIndex + 0] = normal2;
                rgbdMesh.normals[nextBaseIndex + 1] = normal2;
                rgbdMesh.normals[nextBaseIndex + 2] = normal2;
                nextBaseIndex += 3;
            }
        }
    }

    rgbdMesh.vertexCount = nextBaseIndex;

    //ShapeDescriptor::writeXYZ("cloud.xyz", cloud);
    //ShapeDescriptor::writeOBJ(rgbdMesh, "cloudmeshnew.obj");

    ShapeDescriptor::free(cloud);

    return rgbdMesh;
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
    glDeleteTextures(1, &depthTextureID);
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

    glGenTextures(1, &depthTextureID);
    glBindTexture(GL_TEXTURE_2D, depthTextureID);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, offscreenTextureWidth, offscreenTextureHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);

    glGenRenderbuffers(1, &renderBufferID);
    glBindRenderbuffer(GL_RENDERBUFFER, renderBufferID);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, offscreenTextureWidth, offscreenTextureHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, renderBufferID);


    glfwMakeContextCurrent(nullptr);
    isCreated = true;
}

glm::mat4 ShapeBench::OccludedSceneGenerator::computeTransformationMatrix(float pitch, float roll, float yaw, float objectDistanceFromCamera) {
    glm::mat4 positionTransformation = glm::translate(glm::mat4(1.0), glm::vec3(0, 0, -objectDistanceFromCamera));
    positionTransformation *= glm::rotate(glm::mat4(1.0), roll, glm::vec3(0, 0, 1));
    positionTransformation *= glm::rotate(glm::mat4(1.0), yaw, glm::vec3(1, 0, 0));
    positionTransformation *= glm::rotate(glm::mat4(1.0), pitch, glm::vec3(0, 1, 0));
    return positionTransformation;
}