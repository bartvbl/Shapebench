#include "OcclusionGenerator.h"
#include "glad/gl.h"
#include "utils/gl/GLUtils.h"
#include "utils/gl/VAOGenerator.h"
#include "utils/gl/Shader.h"
#include "utils/gl/ShaderLoader.h"
#include "glm/gtc/type_ptr.hpp"
#include "../../../lib/pmp-library/src/pmp/surface_mesh.h"
#include "../../../lib/pmp-library/src/pmp/io/io.h"
#include "../../../lib/pmp-library/src/pmp/algorithms/remeshing.h"
#include "../../../lib/pmp-library/src/pmp/types.h"
#include <iostream>
#include "GLFW/glfw3.h"
#include <random>

ShapeDescriptor::cpu::Mesh computeOccludedMesh() {
    std::random_device rd("/dev/urandom");
    size_t randomSeed = randomSeedParameter.value() != 0 ? randomSeedParameter.value() : rd();
    std::cout << "Random seed: " << randomSeed << std::endl;
    std::minstd_rand0 generator{randomSeed};
    std::uniform_real_distribution<float> distribution(0, 1);

    GLFWwindow* window = GLinitialise();

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
    BufferObject screenQuadVAO = generateVertexArray(screenQuadVertices.data(), screenQuadTexCoords.data(), screenQuadColours.data(), 6);

    Shader objectIDShader = loadShader("res/shaders/", "objectIDShader");
    Shader fullscreenQuadShader = loadShader("res/shaders/", "fullscreenquad");

    // Create offscreen renderer

    const unsigned int offscreenTextureWidth = 4 * 7680;
    const unsigned int offscreenTextureHeight = 4 * 4230;

    unsigned int fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glEnable(GL_DEPTH_TEST);

    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, offscreenTextureWidth, offscreenTextureHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, offscreenTextureWidth, offscreenTextureHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    std::vector<unsigned char> localFramebufferCopy(3 * offscreenTextureWidth * offscreenTextureHeight);

    std::cout << "Generating queries from objects in " << objectDirectory.value() << "..." << std::endl;

    std::vector<std::filesystem::path> haystackFiles = ShapeDescriptor::listDirectory(objectDirectory.value());

    for(unsigned int i = 0; i < haystackFiles.size(); i++) {
        std::cout << "Processing " << (i + 1) << "/" << haystackFiles.size() << ": " << haystackFiles.at(i) << std::endl;

        // Handle other events
        glfwPollEvents();

        // Flip buffers
        glfwSwapBuffers(window);

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

        BufferObject buffers = generateVertexArray(loadedMesh.vertices, loadedMesh.normals, vertexColours.data(), loadedMesh.vertexCount);

        objectIDShader.activate();

        float yaw = float(distribution(generator) * 2.0 * M_PI);
        float pitch = float((distribution(generator) - 0.5) * M_PI);
        float roll = float(distribution(generator) * 2.0 * M_PI);

        glm::mat4 objectProjection = glm::perspective(1.57f, (float) windowWidth / (float) windowHeight, 1.0f, 10000.0f);
        glm::mat4 positionTransformation = glm::translate(glm::vec3(0, 0, -200.0f));
        positionTransformation *= glm::rotate(roll, glm::vec3(0, 0, 1));
        positionTransformation *= glm::rotate(yaw, glm::vec3(1, 0, 0));
        positionTransformation *= glm::rotate(pitch, glm::vec3(0, 1, 0));
        positionTransformation *= glm::translate(-glm::vec3(averageSum.x, averageSum.y, averageSum.z));
        glm::mat4 objectTransformation = objectProjection * positionTransformation;
        glUniformMatrix4fv(16, 1, false, glm::value_ptr(objectTransformation));

        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, offscreenTextureWidth, offscreenTextureHeight);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glBindVertexArray(buffers.VAOID);
        glEnable(GL_DEPTH_TEST);
        glDrawElements(GL_TRIANGLES, loadedMesh.vertexCount, GL_UNSIGNED_INT, nullptr);

        // Do visibility testing

        glBindTexture(GL_TEXTURE_2D, texture);
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

        std::string filename = std::filesystem::path(haystackFiles.at(i)).filename();
        std::filesystem::path outputMeshFile = std::filesystem::path(targetDirectory.value()) / filename;
        ShapeDescriptor::writeOBJ(outMesh, outputMeshFile);

        if(enableTriangleRedistribution.value()) {
            std::cout << "Remeshing.. " << outputMeshFile.string() << std::endl;
            pmp::SurfaceMesh mesh;
            pmp::read(mesh, outputMeshFile.string());
            // Mario Botsch and Leif Kobbelt. A remeshing approach to multiresolution modeling. In Proceedings of Eurographics Symposium on Geometry Processing, pages 189–96, 2004.


            // Using the same approach as PMP library's remeshing tool
            pmp::Scalar totalEdgeLength(0);
            for (const auto& edgeInMesh : mesh.edges()) {
                totalEdgeLength += distance(mesh.position(mesh.vertex(edgeInMesh, 0)),
                                            mesh.position(mesh.vertex(edgeInMesh, 1)));
            }
            pmp::Scalar averageEdgeLength = totalEdgeLength / (pmp::Scalar) mesh.n_edges();

            pmp::uniform_remeshing(mesh, averageEdgeLength);
            pmp::write(outputMeshFile.string());
        }

        ShapeDescriptor::free(outMesh);

        // Draw visible version

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, windowWidth, windowHeight);
        glClearColor(0.5, 0.5, 0.5, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        fullscreenQuadShader.use();

        glBindVertexArray(screenQuadVAO.VAOID);
        glDisable(GL_DEPTH_TEST);
        glBindTextureUnit(0, texture);

        glm::mat4 fullscreenProjection = glm::ortho(0.0f, 1.0f, 0.0f, 1.0f, -1.0f, 1.0f);

        glUniformMatrix4fv(16, 1, false, glm::value_ptr(fullscreenProjection));

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        ShapeDescriptor::free(loadedMesh);

        glDeleteBuffers(1, &buffers.vertexBufferID);
        glDeleteBuffers(1, &buffers.normalBufferID);
        glDeleteBuffers(1, &buffers.colourBufferID);
        glDeleteBuffers(1, &buffers.indexBufferID);
        glDeleteVertexArrays(1, &buffers.VAOID);
    }

    glDeleteFramebuffers(1, &fbo);

    std::cout << std::endl << "Done." << std::endl;
}

ShapeDescriptor::cpu::Mesh createOccludedScene(const nlohmann::json &json,
                                               const ComputedConfig &config,
                                               const Dataset &dataset,
                                               uint64_t seed) {




    ShapeDescriptor::cpu::Mesh occludedMesh;

    return occludedMesh;
}
