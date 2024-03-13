#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "remeshingFilter.h"

ShapeBench::RemeshingFilterOutput ShapeBench::remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config) {
    /*ShapeDescriptor::cpu::Mesh remeshedMesh = internal::remeshMesh(scene.filteredSampleMesh, config);
    ShapeDescriptor::free(scene.filteredSampleMesh);
    scene.filteredSampleMesh = remeshedMesh;

    if(scene.filteredAdditiveNoise.vertexCount > 0) {
        ShapeDescriptor::cpu::Mesh remeshedAdditiveMesh = internal::remeshMesh(scene.filteredAdditiveNoise, config);
        ShapeDescriptor::free(scene.filteredAdditiveNoise);
        scene.filteredAdditiveNoise = remeshedAdditiveMesh;
    }

    // Update reference points
    std::vector<float> bestDistances(scene.mappedReferenceVertices.size(), std::numeric_limits<float>::max());
    std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices = scene.mappedReferenceVertices;

    for(uint32_t meshVertexIndex = 0; meshVertexIndex < scene.filteredSampleMesh.vertexCount; meshVertexIndex++) {
        ShapeDescriptor::cpu::float3 meshVertex = scene.filteredSampleMesh.vertices[meshVertexIndex];
        for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
            ShapeDescriptor::OrientedPoint referencePoint = scene.mappedReferenceVertices.at(i);
            ShapeDescriptor::cpu::float3 referenceVertex = referencePoint.vertex;
            float distanceToReferenceVertex = length(referenceVertex - meshVertex);
            if(distanceToReferenceVertex < bestDistances.at(i)) {
                bestDistances.at(i) = distanceToReferenceVertex;
                scene.mappedReferenceVertices.at(i).vertex = meshVertex;
                scene.mappedReferenceVertices.at(i).normal = scene.filteredSampleMesh.normals[meshVertexIndex];
                scene.mappedReferenceVertexIndices.at(i) = meshVertexIndex;
            }
        }
    }

    ShapeBench::RemeshingFilterOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["triangle-shift-displacement-distance"] = length(scene.mappedReferenceVertices.at(i).vertex - originalReferenceVertices.at(i).vertex);
        output.metadata.push_back(entry);
    }

    malloc_trim(0);

    return output;*/
}