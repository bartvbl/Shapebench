#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "NoisyCaptureFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/decimation.h"
#include "benchmarkCore/randomEngine.h"


ShapeBench::FilterOutput ShapeBench::NoisyCaptureFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset, uint64_t randomSeed) {
    ShapeBench::randomEngine randomEngine(randomSeed);
    ShapeBench::FilterOutput output;

    OcclusionRendererSettings renderSettings;
    renderSettings.nearPlaneDistance = config.at("filterSettings").at("noisyCapture").at("nearPlaneDistance");
    renderSettings.farPlaneDistance = config.at("filterSettings").at("noisyCapture").at("farPlaneDistance");
    renderSettings.fovy = config.at("filterSettings").at("noisyCapture").at("fovYAngleRadians");
    renderSettings.objectDistanceFromCamera = config.at("filterSettings").at("noisyCapture").at("objectDistanceFromCamera");

    std::uniform_real_distribution<float> distribution(0, 1);
    float yaw = float(distribution(randomEngine) * 2.0 * M_PI);
    float pitch = float((distribution(randomEngine) - 0.5) * M_PI);
    float roll = float(distribution(randomEngine) * 2.0 * M_PI);

    nlohmann::json entry;
    entry["noisy-capture-pitch"] = pitch;
    entry["noisy-capture-yaw"] = yaw;
    entry["noisy-capture-roll"] = roll;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        output.metadata.push_back(entry);
    }

    sceneGenerator.computeOccludedMesh(renderSettings, scene);

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

    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["noisy-capture-vertex-displacement"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
        output.metadata.push_back(entry);
    }

    return output;
}

void ShapeBench::NoisyCaptureFilter::init(const nlohmann::json &config) {
    uint32_t visibilityImageWidth = config.at("filterSettings").at("noisyCapture").at("visibilityImageResolution").at(0);
    uint32_t visibilityImageHeight = config.at("filterSettings").at("noisyCapture").at("visibilityImageResolution").at(1);
    sceneGenerator.init(visibilityImageWidth, visibilityImageHeight);
}

void ShapeBench::NoisyCaptureFilter::destroy() {
    sceneGenerator.destroy();
}

void ShapeBench::NoisyCaptureFilter::saveCaches(const nlohmann::json& config) {

}

