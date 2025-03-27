#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "nlohmann/json.hpp"
#include "dataset/Dataset.h"
#include "sha1.hpp"
#include "dataset/miniballGenerator.h"
#include "utils/FileHasher.h"

namespace ShapeBench {
    inline void moveAndScaleMesh(ShapeDescriptor::cpu::Mesh& mesh, const DatasetEntry &datasetEntry) {
        float computedBoundingSphereRadius = std::max<float>(float(datasetEntry.computedObjectRadius), 0.0000001f);
        ShapeDescriptor::cpu::float3 computedBoundingSphereCentre = {float(datasetEntry.computedObjectCentre.at(0)),
                                                                     float(datasetEntry.computedObjectCentre.at(1)),
                                                                     float(datasetEntry.computedObjectCentre.at(2))};

        // Scale mesh down to a unit sphere
        float scaleFactor = 1.0f / float(computedBoundingSphereRadius);
        for (uint32_t i = 0; i < mesh.vertexCount; i++) {
            mesh.vertices[i] = scaleFactor * (mesh.vertices[i] - computedBoundingSphereCentre);
            mesh.normals[i] = normalize(mesh.normals[i]);
        }
    }

    inline ShapeDescriptor::cpu::Mesh readDatasetMesh(const nlohmann::json &config, ShapeBench::LocalDatasetCache* cache, const DatasetEntry &datasetEntry) {
        if(!config.contains("datasetSettings")) {
            throw std::runtime_error("Configuration is missing the key 'datasetSettings'. Aborting.");
        }
        std::filesystem::path datasetBasePath = config.at("datasetSettings").at("objaverseRootDir");
        std::filesystem::path compressedDatasetBasePath = config.at("datasetSettings").at("compressedRootDir");
        const std::filesystem::path &pathInDataset = datasetEntry.meshFile;

        std::filesystem::path originalMeshPath = datasetBasePath / pathInDataset;
        std::filesystem::path compressedMeshPath = compressedDatasetBasePath / pathInDataset;
        compressedMeshPath = compressedMeshPath.replace_extension(".cm");

        cache->acquireFile(compressedMeshPath, pathInDataset, datasetEntry.uncompressedMeshFileIntegrityDigest);

        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(compressedMeshPath);

        std::string readMeshSHA1 = ShapeBench::computeMeshHash(mesh);
        if(config.at("datasetSettings").at("verifyFileIntegrity") && readMeshSHA1 != datasetEntry.meshIntegrityDigest) {
            // The compression library turned out to not produce identical files :(
            // Need to decompress the data first, then compute a hash instead.
            throw std::logic_error("FATAL: SHA1 digest of mesh read from file " + compressedMeshPath.string() + " did not match the one from the dataset cache file. ");
        }

        if(config.at("datasetSettings").at("verifyMiniballComputation") && mesh.vertexCount > 0) {
            ShapeBench::Miniball ball = computeMiniball(mesh);
            ShapeBench::Miniball storedBall;
            storedBall.radius = datasetEntry.computedObjectRadius;
            storedBall.origin = datasetEntry.computedObjectCentre;
            verifyMiniballValidity(ball, storedBall);
        }

        moveAndScaleMesh(mesh, datasetEntry);

        cache->returnFile(compressedMeshPath);

        return mesh;
    }
}
