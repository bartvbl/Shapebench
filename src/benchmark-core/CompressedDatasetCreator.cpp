#include <iostream>
#include <vector>
#include "CompressedDatasetCreator.h"
#include "shapeDescriptor/utilities/fileutils.h"
#include "json.hpp"
#include "shapeDescriptor/utilities/read/GLTFLoader.h"
#include "shapeDescriptor/utilities/read/PointCloudLoader.h"
#include "shapeDescriptor/utilities/write/CompressedGeometryFile.h"
#include "shapeDescriptor/utilities/read/MeshLoader.h"
#include "shapeDescriptor/utilities/read/CompressedGeometryFile.h"
#include "shapeDescriptor/utilities/hash/MeshHasher.h"
#include <shapeDescriptor/utilities/free/mesh.h>
#include <shapeDescriptor/utilities/free/pointCloud.h>

void Shapebench::computeCompressedDataSet(const std::filesystem::path &originalDatasetDirectory,
                                          const std::filesystem::path &compressedDatasetDirectory,
                                          std::filesystem::path metadataFile) {
    const std::vector<std::filesystem::path> datasetFiles = ShapeDescriptor::utilities::listDirectoryAndSubdirectories(originalDatasetDirectory);

    // TODO: once data is available, do a hash check of all files
    std::cout << "    Found " << datasetFiles.size() << " files." << std::endl;

    nlohmann::json datasetCache = {};
    datasetCache["metadata"]["baseDatasetRootDir"] = std::filesystem::absolute(originalDatasetDirectory).string();
    datasetCache["metadata"]["compressedDatasetRootDir"] = std::filesystem::absolute(compressedDatasetDirectory).string();
    datasetCache["metadata"]["cacheDirectory"] = std::filesystem::absolute(metadataFile).string();

    datasetCache["files"] = {};
    size_t nextID = 0;
    unsigned int pointCloudCount = 0;
    unsigned int hashMismatches = 0;
    size_t processedMeshCount = 0;
    #pragma omp parallel for schedule(dynamic) default(none) shared(hashMismatches, processedMeshCount, std::cout, datasetFiles, originalDatasetDirectory, nextID, datasetCache, compressedDatasetDirectory, pointCloudCount, metadataFile)
    for(size_t i = 0; i < datasetFiles.size(); i++) {
        #pragma omp atomic
        processedMeshCount++;
        if(processedMeshCount % 100 == 99) {
            std::cout << "\r    Computing dataset cache.. Processed: " << (processedMeshCount+1) << "/" << datasetFiles.size() << " (" << std::round(10000.0*(double(processedMeshCount+1)/double(datasetFiles.size())))/100.0 << "%), found " << pointCloudCount << " point clouds, " << hashMismatches << " mismatched hashes" << std::flush;
        }

        nlohmann::json datasetEntry;
        datasetEntry["id"] = i;
        std::filesystem::path filePath = std::filesystem::relative(std::filesystem::absolute(datasetFiles.at(i)), originalDatasetDirectory);
        datasetEntry["filePath"] = filePath;
        bool isPointCloud = ShapeDescriptor::utilities::gltfContainsPointCloud(datasetFiles.at(i));
        datasetEntry["isPointCloud"] = isPointCloud;
        std::filesystem::path compressedMeshPath = compressedDatasetDirectory / filePath;
        compressedMeshPath.replace_extension(".cm");
        try {
            if(isPointCloud) {
                #pragma omp atomic
                pointCloudCount++;
                ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::utilities::loadPointCloud(datasetFiles.at(i));
                ShapeDescriptor::utilities::writeCompressedGeometryFile(cloud, compressedMeshPath, true);
                datasetEntry["vertexCount"] = cloud.pointCount;
                ShapeDescriptor::free::pointCloud(cloud);
            } else {
                ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(datasetFiles.at(i));
                ShapeDescriptor::utilities::writeCompressedGeometryFile(mesh, compressedMeshPath, true);
                datasetEntry["vertexCount"] = mesh.vertexCount;
                uint32_t hash = ShapeDescriptor::hashMesh(mesh);
                datasetEntry["meshHash"] = hash;
                ShapeDescriptor::free::mesh(mesh);

                ShapeDescriptor::cpu::Mesh readMesh = ShapeDescriptor::utilities::readMeshFromCompressedGeometryFile(compressedMeshPath);
                uint32_t readBackHash = ShapeDescriptor::hashMesh(readMesh);
                ShapeDescriptor::free::mesh(readMesh);
                if(hash != readBackHash) {
                    std::cout << "\n!! HASH MISMATCH " + compressedMeshPath.string() + "\n" << std::flush;
                    datasetEntry["compressedMeshHash"] = readBackHash;
                    #pragma omp atomic
                    hashMismatches++;
                }
            }
        } catch(std::runtime_error& e) {
            std::cout << "!! ERROR: FILE FAILED TO PARSE: " + filePath.string() + "\n   REASON: " + e.what() + "\n" << std::flush;
            datasetEntry["parseFailed"] = true;
            datasetEntry["parseFailedReason"] = e.what();
            datasetEntry["vertexCount"] = -1;
        }

        #pragma omp critical
        {
            datasetCache["files"].push_back(datasetEntry);
            nextID++;

            if((nextID + 1) % 50000 == 0) {
                std::cout << std::endl << "Writing backup JSON.. " << std::endl;
                std::ofstream outCacheStream {metadataFile};
                outCacheStream << datasetCache.dump(4);
            }
        };
    }
    std::cout << std::endl;

    std::ofstream outCacheStream {metadataFile};
    outCacheStream << datasetCache.dump(4);
}
