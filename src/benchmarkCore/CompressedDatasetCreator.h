#pragma once

#include <filesystem>
#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "json.hpp"
#include "nlohmann/json.hpp"
#include "sha1.hpp"
#include "Seb.h"
#include "utils/prettyprint.h"
#include "replication/RandomSubset.h"

namespace ShapeBench {
    // Takes in a dataset of file formats supported by the libShapeDescriptor library and compresses it using the library's compact mesh format
    // Can optionally produce a JSON file with dataset metadata
    inline nlohmann::json computeOrReadDatasetCache(const nlohmann::json& replicationConfiguration,
                                                    const std::filesystem::path &originalDatasetDirectory,
                                                    const std::filesystem::path &compressedDatasetDirectory,
                                                    const std::filesystem::path &metadataFile,
                                                    uint64_t recomputeRandomSeed) {

        std::cout << "Searching for uncompressed dataset files.." << std::endl;

        const std::vector<std::filesystem::path> datasetFiles = ShapeDescriptor::listDirectoryAndSubdirectories(originalDatasetDirectory);
        nlohmann::json datasetCache = {};
        bool previousCacheFound = std::filesystem::exists(metadataFile);

        // Replication settings
        bool forceInvalidateCache = replicationConfiguration.at("recomputeEntirely");
        bool recomputeRandomSubset = replicationConfiguration.at("verifyRandomSubset");

        if(forceInvalidateCache && recomputeRandomSubset) {
            std::cout << "WARNING: recomputation of the entire dataset is enabled, but also the verification of a random subset. The latter is meaningless in this context and is disabled." << std::endl;
            recomputeRandomSubset = false;
        }

        ShapeBench::RandomSubset replicationSubset;
        if(recomputeRandomSubset) {
            uint32_t numberOfFilesToRecompute = replicationConfiguration.at("randomSubsetSize");
            std::cout << "Replication of compressed dataset and dataset metadata enabled, randomly replicating a subset of " << numberOfFilesToRecompute << " files." << std::endl;
            replicationSubset = ShapeBench::RandomSubset(0, datasetFiles.size(), numberOfFilesToRecompute, recomputeRandomSeed);
        }



        if(previousCacheFound && !forceInvalidateCache) {
            std::filesystem::path bakPath = metadataFile;
            bakPath.replace_extension(".bak.json");
            if(!std::filesystem::exists(bakPath) || std::filesystem::file_size(bakPath) != std::filesystem::file_size(metadataFile)) {
                std::filesystem::copy(metadataFile, bakPath, std::filesystem::copy_options::overwrite_existing);
            }

            std::ifstream inputStream{metadataFile};
            datasetCache = nlohmann::json::parse(inputStream);
            std::cout << "Loaded dataset cache.. (contains " << datasetCache.at("files").size() << " files)" << std::endl;

            if(!std::filesystem::exists(originalDatasetDirectory)) {
                std::cout << "    Uncompressed dataset was not found. Falling back to using compressed dataset as-is" << std::endl;
                return datasetCache;
            }
        } else {
            std::cout << "Dataset cache was not found." << std::endl;
            std::cout << "Creating dataset cache.. (found " << datasetFiles.size() << " files)" << std::endl;
            datasetCache["metadata"]["baseDatasetRootDir"] = std::filesystem::absolute(originalDatasetDirectory).string();
            datasetCache["metadata"]["compressedDatasetRootDir"] = std::filesystem::absolute(compressedDatasetDirectory).string();
            datasetCache["metadata"]["cacheDirectory"] = std::filesystem::absolute(metadataFile).string();

            datasetCache["files"] = {};
            // Creating stubs
            for(uint32_t i = 0; i < datasetFiles.size(); i++) {
                nlohmann::json datasetEntry;
                datasetEntry["id"] = i;
                datasetCache["files"].push_back(datasetEntry);
            }
        }

        unsigned int pointCloudCount = 0;
        size_t processedMeshCount = 0;
        bool newMeshesLoaded = false;

        std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic) default(none) shared(replicationSubset, processedMeshCount, newMeshesLoaded, std::cout, datasetFiles, originalDatasetDirectory, datasetCache, compressedDatasetDirectory, pointCloudCount, metadataFile)
        for(size_t i = 0; i < datasetFiles.size(); i++) {
            bool entryIsMissing = !datasetCache["files"].at(i).contains("isPointCloud");
            bool shouldBeReplicated = replicationSubset.contains(i);

            // Skip if this entry has been processed before
            if(entryIsMissing || shouldBeReplicated) {
                newMeshesLoaded = true;
                nlohmann::json datasetEntry;
                if(shouldBeReplicated) {
                    datasetEntry = datasetCache["files"].at(i);
                    if(datasetEntry.at("id") != i) {
                        throw std::logic_error("ID of entry " + std::to_string(i) + " does not match the expected one! (read: " + std::to_string(uint32_t(datasetEntry.at("id"))) + ")");
                    }
                }
                datasetEntry["id"] = i;
                std::filesystem::path filePath = std::filesystem::relative(std::filesystem::absolute(datasetFiles.at(i)), originalDatasetDirectory);
                bool isPointCloud = false;
                if(filePath.extension() == ".glb") {
                    isPointCloud = ShapeDescriptor::gltfContainsPointCloud(datasetFiles.at(i));
                }
                datasetEntry["filePath"] = filePath;
                std::string originalFileSha1 = SHA1::from_file(filePath.string());
                if(!entryIsMissing) {
                    if(datasetEntry.at("originalFileSha1") != originalFileSha1) {
                        throw std::logic_error("FATAL: file digest of file " + filePath.string() + " did not match the one on record!");
                    }
                } else {
                }
                datasetEntry["originalFileSha1"] = originalFileSha1;

                datasetEntry["isPointCloud"] = isPointCloud;
                std::filesystem::path compressedMeshPath = compressedDatasetDirectory / filePath;
                compressedMeshPath.replace_extension(".cm");

                std::vector<double> coordinate(3);
                std::vector<Seb::Point<double>> vertices;

                try {
                    if (isPointCloud) {
                        #pragma omp atomic
                        pointCloudCount++;
                        ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::loadPointCloud(datasetFiles.at(i));
                        ShapeDescriptor::writeCompressedGeometryFile(cloud, compressedMeshPath, true);
                        datasetEntry["vertexCount"] = cloud.pointCount;
                        std::string compressedFileSha1 = SHA1::from_file(compressedMeshPath.string());
                        if(!entryIsMissing) {
                            if(datasetEntry.at("compressedFileSha1") != compressedFileSha1) {
                                throw std::logic_error("FATAL: file digest of compressed file " + filePath.string() + " did not match the one on record!");
                            }
                        } else {
                        }
                        datasetEntry["compressedFileSha1"] = compressedFileSha1;


                        // Integrity check
                        ShapeDescriptor::cpu::PointCloud readCloud = ShapeDescriptor::readPointCloudFromCompressedGeometryFile(
                                compressedMeshPath);
                        if (ShapeDescriptor::comparePointCloud(cloud, readCloud)) {
                            throw std::logic_error("!! POINT CLOUD HASH MISMATCH " + compressedMeshPath.string());
                        }
                        ShapeDescriptor::free(readCloud);

                        vertices.reserve(cloud.pointCount);
                        for (uint32_t vertex = 0; vertex < cloud.pointCount; vertex++) {
                            ShapeDescriptor::cpu::float3 point = cloud.vertices[vertex];
                            coordinate.at(0) = point.x;
                            coordinate.at(1) = point.y;
                            coordinate.at(2) = point.z;
                            vertices.emplace_back(3, coordinate.begin());
                        }

                        ShapeDescriptor::free(cloud);
                    } else {
                        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(datasetFiles.at(i));
                        ShapeDescriptor::writeCompressedGeometryFile(mesh, compressedMeshPath, true);
                        datasetEntry["vertexCount"] = mesh.vertexCount;
                        std::string compressedFileSha1 = SHA1::from_file(compressedMeshPath.string());
                        if(!entryIsMissing) {
                            if(datasetEntry.at("compressedFileSha1") != compressedFileSha1) {
                                throw std::logic_error("FATAL: file digest of compressed file " + filePath.string() + " did not match the one on record!");
                            }
                        } else {
                        }
                        datasetEntry["compressedFileSha1"] = compressedFileSha1;

                        // Integrity check
                        ShapeDescriptor::cpu::Mesh readMesh = ShapeDescriptor::loadMesh(compressedMeshPath);

                        if (!ShapeDescriptor::compareMesh(mesh, readMesh)) {
                            throw std::logic_error("!! MESH HASH MISMATCH " + compressedMeshPath.string());
                        }
                        ShapeDescriptor::free(readMesh);

                        vertices.reserve(mesh.vertexCount);
                        for (uint32_t vertex = 0; vertex < mesh.vertexCount; vertex++) {
                            ShapeDescriptor::cpu::float3 point = mesh.vertices[vertex];
                            coordinate.at(0) = point.x;
                            coordinate.at(1) = point.y;
                            coordinate.at(2) = point.z;
                            vertices.emplace_back(3, coordinate.begin());
                        }

                        ShapeDescriptor::free(mesh);
                    }
                } catch (std::runtime_error &e) {
                    std::cout << "!! ERROR: FILE FAILED TO PARSE: " + filePath.string() + "\n   REASON: " + e.what() + "\n"
                              << std::flush;
                    datasetEntry["parseFailed"] = true;
                    datasetEntry["parseFailedReason"] = e.what();
                    datasetEntry["vertexCount"] = -1;
                }

                if (!vertices.empty()) {
                    Seb::Smallest_enclosing_ball<double> ball(3, vertices);
                    double boundingRadius = ball.radius();
                    for (int j = 0; j < 3; j++) {
                        coordinate.at(j) = ball.center_begin()[j];
                    }
                    if(!entryIsMissing) {
                        if(datasetEntry.at("boundingSphereRadius") != boundingRadius) {
                            throw std::logic_error("FATAL: The computed bounding sphere radius deviates from the one in the cache: " + std::to_string(double(datasetEntry.at("boundingSphereRadius"))) + " vs " + std::to_string(boundingRadius));
                        }
                        if(datasetEntry.at("boundingSphereCentre").at(0) != coordinate.at(0)
                        || datasetEntry.at("boundingSphereCentre").at(1) != coordinate.at(1)
                        || datasetEntry.at("boundingSphereCentre").at(2) != coordinate.at(2)) {
                            throw std::logic_error("FATAL: The computed bounding sphere coordinate deviated from the one in the cache: (" +
                                                      std::to_string(double(datasetEntry.at("boundingSphereCentre").at(0))) + ", "
                                                    + std::to_string(double(datasetEntry.at("boundingSphereCentre").at(1))) + ", "
                                                    + std::to_string(double(datasetEntry.at("boundingSphereCentre").at(2)))
                                                    + ") vs ("
                                                    + std::to_string(coordinate.at(0)) + ", "
                                                    + std::to_string(coordinate.at(1)) + ", "
                                                    + std::to_string(coordinate.at(2)) + ")");
                        }
                    }
                    datasetEntry["boundingSphereCentre"] = {coordinate.at(0), coordinate.at(1), coordinate.at(2)};
                    datasetEntry["boundingSphereRadius"] = boundingRadius;
                }


                #pragma omp critical
                {
                    datasetCache["files"].at(i) = datasetEntry;

                    if ((i + 1) % 10000 == 0) {
                        std::cout << std::endl << "Writing backup JSON.. " << std::endl;
                        std::ofstream outCacheStream{metadataFile};
                        outCacheStream << datasetCache.dump(4);
                    }
                    if ((i + 1) % 100 == 0) {
                        malloc_trim(0);
                    }
                }
            }
#pragma omp critical
            {
                processedMeshCount++;

                if(processedMeshCount % 100 == 99 || processedMeshCount + 1 == datasetFiles.size()) {
                    std::cout << "\r     ";
                    ShapeBench::drawProgressBar(processedMeshCount + 1, datasetFiles.size());
                    std::cout << " " << (processedMeshCount+1) << "/" << datasetFiles.size() << " (" << std::round(10000.0*(double(processedMeshCount+1)/double(datasetFiles.size())))/100.0 << "%)      ";
                    if(newMeshesLoaded) {
                        std::cout << ", found " << pointCloudCount << " point clouds";
                    }
                    std::cout << std::flush;
                }
            }
        }
        std::cout << std::endl;

        std::chrono::time_point<std::chrono::steady_clock> endTime = std::chrono::steady_clock::now();
        if(newMeshesLoaded) {
            std::cout << "    Compressed dataset was successfully computed. Total duration: ";
            ShapeBench::printDuration(endTime - startTime);
            std::cout << std::endl;

            std::ofstream outCacheStream {metadataFile};
            outCacheStream << datasetCache.dump(4);
        } else {
            std::cout << "    Dataset cache loaded successfully" << std::endl;
        }

        if(recomputeRandomSubset) {
            // If a mismatch occurs, the program exits.
            // Therefore, if the program reaches here, it means no mismatches were detected.
            std::cout << "    Replication: Integrity of random subset of input files was successfully verified." << std::endl;
            std::cout << "    Replication: Metadata computed for random subset of input files was successfully verified." << std::endl;
        }


        return datasetCache;
    }

}
