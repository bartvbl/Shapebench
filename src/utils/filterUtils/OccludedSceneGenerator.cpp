#include "OccludedSceneGenerator.h"
#include "benchmarkCore/randomEngine.h"
#include "glm/ext/matrix_transform.hpp"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "utils/gl/GeometryBuffer.h"
#include "utils/gl/VAOGenerator.h"
#include <glm/glm.hpp>


struct GeomIDUniforms {
    glm::mat4 mvp_mat;
    bool writeDepth = false;
};

void objectID_vertexShader(float* vs_output, vec4* vertex_attribs, Shader_Builtins* builtins, void* uniformPointer)
{
    ((vec4*)vs_output)[0] = vertex_attribs[1]; //color

    GeomIDUniforms* uniforms = reinterpret_cast<GeomIDUniforms*>(uniformPointer);

    glm::vec4 inputVertex = glm::vec4(vertex_attribs[0].x, vertex_attribs[0].y, vertex_attribs[0].z, 1);
    glm::vec4 transformedVertex = uniforms->mvp_mat * inputVertex;
    vec4 outputVertex = {transformedVertex.x, transformedVertex.y, transformedVertex.z, transformedVertex.w};

    builtins->gl_Position = outputVertex;
}

void objectID_fragmentShader(float* fs_input, Shader_Builtins* builtins, void* uniformPointer)
{
    GeomIDUniforms* uniforms = reinterpret_cast<GeomIDUniforms*>(uniformPointer);
    if(uniforms->writeDepth) {
        builtins->gl_FragColor = vec4(builtins->gl_FragDepth, builtins->gl_FragDepth, builtins->gl_FragDepth, 1.0);
    } else {
        float depth = builtins->gl_FragDepth;
        uint32_t* depthAsInt = reinterpret_cast<uint32_t*>(&depth);
        vec4 colour = {
                float(((*depthAsInt) >> 24) & 0xFF) / 255.0f,
                float(((*depthAsInt) >> 16) & 0xFF) / 255.0f,
                float(((*depthAsInt) >> 8) & 0xFF) / 255.0f,
                float(((*depthAsInt) >> 0) & 0xFF) / 255.0f
        };
        builtins->gl_FragColor = colour;
    }
}

ShapeBench::OccludedSceneGenerator::OccludedSceneGenerator() {

}

ShapeBench::OccludedSceneGenerator::~OccludedSceneGenerator() {
    destroy();
}

void ShapeBench::OccludedSceneGenerator::renderSceneToOffscreenBuffer(ShapeBench::FilteredMeshPair& scene, OcclusionRendererSettings settings, ShapeDescriptor::cpu::float3* vertexColours, uint8_t* outFrameBuffer, float* outDepthBuffer) {
    glm::mat4 objectProjection = glm::perspective(settings.fovy, (float) offscreenTextureWidth / (float) offscreenTextureHeight, settings.nearPlaneDistance, settings.farPlaneDistance);
    glm::mat4 positionTransformation = computeTransformationMatrix(settings.pitch, settings.roll, settings.yaw, settings.objectDistanceFromCamera);
    glm::mat4 objectTransformation = objectProjection * positionTransformation;

    GeomIDUniforms geometryIDUniforms;

    std::unique_lock<std::mutex> filterLock{occlusionFilterLock};

    GeometryBuffer sampleMeshBuffers = generateVertexArray(scene.filteredSampleMesh.vertices,
                                                           scene.filteredSampleMesh.normals,
                                                           vertexColours,
                                                           scene.filteredSampleMesh.vertexCount);
    GeometryBuffer additiveNoiseBuffers = generateVertexArray(scene.filteredAdditiveNoise.vertices,
                                                              scene.filteredAdditiveNoise.normals,
                                                              vertexColours == nullptr ? nullptr : vertexColours + scene.filteredSampleMesh.vertexCount,
                                                              scene.filteredAdditiveNoise.vertexCount);
    glUseProgram(this->GeometryIDShader);
    geometryIDUniforms.mvp_mat = objectTransformation;
    geometryIDUniforms.writeDepth = outDepthBuffer != nullptr;
    pglSetUniform(&geometryIDUniforms);

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(sampleMeshBuffers.vaoID);
    glDrawElements(GL_TRIANGLES, scene.filteredSampleMesh.vertexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(additiveNoiseBuffers.vaoID);
    glDrawElements(GL_TRIANGLES, scene.filteredAdditiveNoise.vertexCount, GL_UNSIGNED_INT, nullptr);

    // Do visibility testing

    if(outFrameBuffer != nullptr) {
        std::copy(this->window.frameBuffer, this->window.frameBuffer + offscreenTextureWidth * offscreenTextureHeight * sizeof(unsigned int), outFrameBuffer);
    } else if(outDepthBuffer != nullptr) {
        float* frameBufferAsFloatBuffer = reinterpret_cast<float*>(this->window.frameBuffer);
        std::copy(frameBufferAsFloatBuffer, frameBufferAsFloatBuffer + offscreenTextureWidth * offscreenTextureHeight, outDepthBuffer);
    }

    sampleMeshBuffers.destroy();
    additiveNoiseBuffers.destroy();
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

    isDestroyed = true;
    free_glContext(window.context);
}

ShapeBench::SoftwareContext InitialiseSoftwareGL(uint32_t windowWidth, uint32_t windowHeight)
{
    ShapeBench::SoftwareContext context;
    context.context = new glContext;
    context.frameBuffer = new u32[windowWidth * windowHeight];

    if (!init_glContext(context.context, &context.frameBuffer, windowWidth, windowHeight, 32, 0xFF000000, 0x00FF0000, 0x0000FF00, 0x000000FF)) {
        puts("Failed to initialize glContext");
        exit(0);
    }

    std::cout << "    Created OpenGL context: " << glGetString(GL_VERSION) << std::endl;

    glClearColor(0.3, 0.3, 0.3, 1.0);

    return context;
}

void ShapeBench::OccludedSceneGenerator::init(uint32_t visibilityImageWidth, uint32_t visibilityImageHeight) {
    offscreenTextureWidth = visibilityImageWidth;
    offscreenTextureHeight = visibilityImageHeight;

    window = InitialiseSoftwareGL(offscreenTextureWidth, offscreenTextureHeight);

    GLenum interpolation[4] = { PGL_SMOOTH4 };
    GeometryIDShader = pglCreateProgram(objectID_vertexShader, objectID_fragmentShader, 4, interpolation, GL_FALSE);

    isCreated = true;
}

glm::mat4 ShapeBench::OccludedSceneGenerator::computeTransformationMatrix(float pitch, float roll, float yaw, float objectDistanceFromCamera) {
    glm::mat4 positionTransformation = glm::translate(glm::mat4(1.0), glm::vec3(0, 0, -objectDistanceFromCamera));
    positionTransformation *= glm::rotate(glm::mat4(1.0), roll, glm::vec3(0, 0, 1));
    positionTransformation *= glm::rotate(glm::mat4(1.0), yaw, glm::vec3(1, 0, 0));
    positionTransformation *= glm::rotate(glm::mat4(1.0), pitch, glm::vec3(0, 1, 0));
    return positionTransformation;
}

