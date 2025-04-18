#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "NoisyCaptureFilter.h"
#include "benchmarkCore/randomEngine.h"


ShapeBench::FilterOutput ShapeBench::NoisyCaptureFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                                   const Dataset &dataset,
                                                   ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    ShapeBench::randomEngine randomEngine(randomSeed);
    ShapeBench::FilterOutput output;
    uint32_t initialVertexCount = scene.filteredSampleMesh.vertexCount;

    OcclusionRendererSettings renderSettings;
    renderSettings.nearPlaneDistance = config.at("filterSettings").at("depthCameraCapture").at("nearPlaneDistance");
    renderSettings.farPlaneDistance = config.at("filterSettings").at("depthCameraCapture").at("farPlaneDistance");
    renderSettings.fovy = config.at("filterSettings").at("depthCameraCapture").at("fovYAngleRadians");
    renderSettings.rgbdDepthCutoffFactor = config.at("filterSettings").at("depthCameraCapture").at("depthCutoffFactor");
    float objectPlacementMin = float(config.at("filterSettings").at("depthCameraCapture").at("objectDistanceFromCameraLimitNear"));
    float objectPlacementMax = float(config.at("filterSettings").at("depthCameraCapture").at("objectDistanceFromCameraLimitFar"));
    float displacementDistanceFactor = float(config.at("filterSettings").at("depthCameraCapture").at("mappedVertexDisplacementCutoffFactor"));

    std::uniform_real_distribution<float> distanceDistribution(objectPlacementMin, objectPlacementMax);
    renderSettings.objectDistanceFromCamera = distanceDistribution(randomEngine);

    std::uniform_real_distribution<float> distribution(0, 1);
    renderSettings.yaw = float(distribution(randomEngine) * 2.0 * M_PI);
    renderSettings.pitch = float((distribution(randomEngine) - 0.5) * M_PI);
    renderSettings.roll = float(distribution(randomEngine) * 2.0 * M_PI);

    float displacementCutoff = 0;
    ShapeDescriptor::cpu::Mesh sceneMesh = sceneGenerator.computeRGBDMesh(renderSettings, scene, displacementCutoff);
    displacementCutoff *= displacementDistanceFactor;
    ShapeDescriptor::free(scene.filteredSampleMesh);
    ShapeDescriptor::free(scene.filteredAdditiveNoise);
    scene.filteredSampleMesh = sceneMesh;
    // This filter cannot easily distinguish what is additive noise, so it is all merged into the sample mesh instead

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
                // Leave the normal intact
                //scene.mappedReferenceVertices.at(i).normal = scene.filteredSampleMesh.normals[meshVertexIndex];
                scene.mappedReferenceVertexIndices.at(i) = meshVertexIndex;
            }
        }
    }


    for(uint32_t i = 0; i < bestDistances.size(); i++) {
        if(bestDistances.at(i) > displacementCutoff) {
            scene.mappedVertexIncluded.at(i) = false;
        }
    }

    // Not an entirely correct way to map all vertices, but this is the closest you can probably get
    // In any case, the portion that is not visible from the camera is straight up removed, so no orientation changes
    scene.sampleMeshTransformation *= glm::mat4(1.0);
    scene.sampleMeshNormalTransformation *= glm::mat3(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::mat4(1.0);
    }

    nlohmann::json entry;
    entry["depth-camera-capture-pitch"] = renderSettings.pitch;
    entry["depth-camera-capture-yaw"] = renderSettings.yaw;
    entry["depth-camera-capture-roll"] = renderSettings.roll;
    entry["depth-camera-capture-distance-from-camera"] = renderSettings.objectDistanceFromCamera;
    entry["depth-camera-capture-initial-vertex-count"] = initialVertexCount;
    entry["depth-camera-capture-filtered-vertex-count"] = scene.filteredSampleMesh.vertexCount;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        output.metadata.push_back(entry);
        output.metadata.at(i)["depth-camera-capture-displaced-distance"] = bestDistances.at(i);
    }

    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["depth-camera-capture-vertex-displacement"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
        output.metadata.push_back(entry);
    }

    return output;
}

void ShapeBench::NoisyCaptureFilter::init(const nlohmann::json &config, bool invalidateCaches) {
    uint32_t visibilityImageWidth = config.at("filterSettings").at("depthCameraCapture").at("visibilityImageResolution").at(0);
    uint32_t visibilityImageHeight = config.at("filterSettings").at("depthCameraCapture").at("visibilityImageResolution").at(1);
    sceneGenerator.init(visibilityImageWidth, visibilityImageHeight);
}

void ShapeBench::NoisyCaptureFilter::destroy() {
    sceneGenerator.destroy();
}

void ShapeBench::NoisyCaptureFilter::saveCaches(const nlohmann::json& config) {

}

ShapeBench::FilterOutput
ShapeBench::NoisyCaptureFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                            ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset,
                                            ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::NoisyCaptureFilter::getFilterName() const {
    return "noisy-capture";
}


