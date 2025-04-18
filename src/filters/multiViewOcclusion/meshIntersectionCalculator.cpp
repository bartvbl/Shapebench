#include "meshIntersectionCalculator.h"
#include "fmt/format.h"

ShapeDescriptor::cpu::Mesh ShapeBench::computeCommonTrianglesMesh(const ShapeBench::FilteredMeshPair &model,
                                                                  const ShapeBench::FilteredMeshPair &scene) {
    if(model.remainingTrianglesFromOriginalMesh.size() != model.originalMesh.vertexCount ||
       scene.remainingTrianglesFromOriginalMesh.size() != scene.originalMesh.vertexCount ||
       model.originalMesh.vertexCount != scene.originalMesh.vertexCount) {
        throw std::runtime_error(fmt::format("Sanity check for computing common area triangle mesh failed. This can happen if a filter that precedes this one breaks the triangle correspondence by retriangulating the mesh surface. Relevant information:\n- original mesh sample count (model): {},\n- original mesh sample count (scene): {}\n- Length of list remaining triangles in mesh (model): {}\n- Length of list remaining triangles in mesh (scene): {}\n", model.originalMesh.vertexCount, scene.originalMesh.vertexCount, model.remainingTrianglesFromOriginalMesh.size(), scene.remainingTrianglesFromOriginalMesh.size()));
    }

    uint32_t nextTriangleVertexBaseIndex = 0;

    // The smallest number of triangles the two meshes can have in common is the entire smallest mesh
    uint32_t commonMeshVertexCount = std::min(model.filteredSampleMesh.vertexCount, scene.filteredSampleMesh.vertexCount);
    ShapeDescriptor::cpu::Mesh commonMesh(commonMeshVertexCount);

    for(uint32_t vertexIndex = 0; vertexIndex < model.originalMesh.vertexCount; vertexIndex += 3) {
        uint32_t modelActiveTriangleCount =
                (model.remainingTrianglesFromOriginalMesh.at(vertexIndex + 0) ? 1 : 0)
              + (model.remainingTrianglesFromOriginalMesh.at(vertexIndex + 1) ? 1 : 0)
              + (model.remainingTrianglesFromOriginalMesh.at(vertexIndex + 2) ? 1 : 0);
        if(modelActiveTriangleCount == 1 || modelActiveTriangleCount == 2) {
            throw std::runtime_error("Model mesh inconsistency detected!");
        }

        uint32_t sceneActiveTriangleCount =
                  (scene.remainingTrianglesFromOriginalMesh.at(vertexIndex + 0) ? 1 : 0)
                + (scene.remainingTrianglesFromOriginalMesh.at(vertexIndex + 1) ? 1 : 0)
                + (scene.remainingTrianglesFromOriginalMesh.at(vertexIndex + 2) ? 1 : 0);
        if(sceneActiveTriangleCount == 1 || sceneActiveTriangleCount == 2) {
            throw std::runtime_error("Scene mesh inconsistency detected!");
        }

        if(modelActiveTriangleCount == 3 && sceneActiveTriangleCount == 3) {
            // Triangle is shared. Add it to the output mesh
            if(nextTriangleVertexBaseIndex >= commonMesh.vertexCount) {
                throw std::runtime_error("Tried to create a merged mesh that was larger than the allocated size! This should be impossible.");
            }

            commonMesh.vertices[nextTriangleVertexBaseIndex] = model.originalMesh.vertices[vertexIndex];
            commonMesh.normals[nextTriangleVertexBaseIndex] = model.originalMesh.normals[vertexIndex];
            if(model.originalMesh.vertexColours != nullptr) {
                commonMesh.vertexColours[nextTriangleVertexBaseIndex] = model.originalMesh.vertexColours[vertexIndex];
            }
            if(model.originalMesh.textureCoordinates != nullptr && model.originalMesh.textureCoordinates != nullptr) {
                commonMesh.textureCoordinates[nextTriangleVertexBaseIndex] = model.originalMesh.textureCoordinates[vertexIndex];
            }
            nextTriangleVertexBaseIndex += 3;
        }
    }

    commonMesh.vertexCount = nextTriangleVertexBaseIndex;
    return commonMesh;
}

void ShapeBench::transformMesh(ShapeBench::FilteredMeshPair &mesh, glm::mat4 vertexTransformation, glm::mat3 normalTransformation) {
    transformMesh(mesh.filteredSampleMesh, vertexTransformation, normalTransformation);
    transformMesh(mesh.filteredAdditiveNoise, vertexTransformation, normalTransformation);

    for(ShapeDescriptor::OrientedPoint& mappedReferenceVertice : mesh.mappedReferenceVertices) {
        glm::vec4 transformedVertex = vertexTransformation * glm::vec4(internal::toGLM(mappedReferenceVertice.vertex), 1.0);
        mappedReferenceVertice.vertex = internal::fromGLM(glm::vec3(transformedVertex));
        glm::vec3 transformedNormal = normalTransformation * internal::toGLM(mappedReferenceVertice.normal);
        mappedReferenceVertice.normal = internal::fromGLM(transformedNormal);
    }
}

void ShapeBench::transformMesh(ShapeDescriptor::cpu::Mesh &mesh, glm::mat4 vertexTransformation,
                               glm::mat3 normalTransformation) {
    for(uint32_t vertexIndex = 0; vertexIndex < mesh.vertexCount; vertexIndex++) {
        glm::vec4 transformedVertex = vertexTransformation * glm::vec4(internal::toGLM(mesh.vertices[vertexIndex]), 1.0);
        mesh.vertices[vertexIndex] = internal::fromGLM(glm::vec3(transformedVertex));
        glm::vec3 transformedNormal = normalTransformation * internal::toGLM(mesh.normals[vertexIndex]);
        mesh.normals[vertexIndex] = internal::fromGLM(transformedNormal);
    }
}


