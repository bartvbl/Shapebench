#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmarkCore/Dataset.h"

namespace ShapeBench {
    inline ShapeDescriptor::cpu::Mesh readDatasetMesh(const nlohmann::json &config, const DatasetEntry &datasetEntry) {
        const std::filesystem::path &pathInDataset = datasetEntry.meshFile;
        float computedBoundingSphereRadius = datasetEntry.computedObjectRadius;
        ShapeDescriptor::cpu::float3 computedBoundingSphereCentre = datasetEntry.computedObjectCentre;

        std::filesystem::path datasetBasePath = config.at("compressedDatasetRootDir");
        std::filesystem::path currentMeshPath = datasetBasePath / pathInDataset;
        // Use compressed mesh file as fallback
        if(!std::filesystem::exists(currentMeshPath)) {
            currentMeshPath = currentMeshPath.replace_extension(".cm");
        }
        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(currentMeshPath);

        // Scale mesh down to a unit sphere
        float scaleFactor = 1.0f / float(computedBoundingSphereRadius);
        for (uint32_t i = 0; i < mesh.vertexCount; i++) {
            mesh.vertices[i] = scaleFactor * (mesh.vertices[i] - computedBoundingSphereCentre);
        }
        return mesh;
    }
}