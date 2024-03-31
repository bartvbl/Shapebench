#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "NoisyCaptureFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/decimation.h"
#include "benchmarkCore/randomEngine.h"


ShapeBench::FilterOutput ShapeBench::NoisyCaptureFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset, uint64_t randomSeed) {


    // Update reference points
    std::vector<float> bestDistances(scene.mappedReferenceVertices.size(), std::numeric_limits<float>::max());
    std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices = scene.mappedReferenceVertices;

    for (uint32_t meshVertexIndex = 0; meshVertexIndex < scene.filteredSampleMesh.vertexCount; meshVertexIndex++) {
        ShapeDescriptor::cpu::float3 meshVertex = scene.filteredSampleMesh.vertices[meshVertexIndex];
        for (uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
            ShapeDescriptor::OrientedPoint referencePoint = originalReferenceVertices.at(i);
            ShapeDescriptor::cpu::float3 referenceVertex = referencePoint.vertex;
            float distanceToReferenceVertex = length(referenceVertex - meshVertex);
            if (distanceToReferenceVertex < bestDistances.at(i)) {
                bestDistances.at(i) = distanceToReferenceVertex;
                scene.mappedReferenceVertices.at(i).vertex = meshVertex;
                scene.mappedReferenceVertices.at(i).normal = scene.filteredSampleMesh.normals[meshVertexIndex];
                scene.mappedReferenceVertexIndices.at(i) = meshVertexIndex;
            }
        }
    }


    ShapeBench::FilterOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["mesh-resolution-vertex-displacement"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
        output.metadata.push_back(entry);
    }

    return output;
}

void ShapeBench::NoisyCaptureFilter::init(const nlohmann::json &config) {

}

void ShapeBench::NoisyCaptureFilter::destroy() {

}

void ShapeBench::NoisyCaptureFilter::saveCaches(const nlohmann::json& config) {

}

