#pragma once

#include <filesystem>
#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "json.hpp"
#include "nlohmann/json.hpp"
#include "sha1.hpp"
#include "Seb.h"
#include "utils/prettyprint.h"

namespace ShapeBench {
    // Takes in a dataset of file formats supported by the libShapeDescriptor library and compresses it using the library's compact mesh format
    // Can optionally produce a JSON file with dataset metadata
    inline nlohmann::json computeOrReadDatasetCache(const nlohmann::json& replicationConfiguration,
                                                         const std::filesystem::path &originalDatasetDirectory,
                                                         const std::filesystem::path &compressedDatasetDirectory,
                                                         const std::filesystem::path &metadataFile) {

        std::cout << "Searching for uncompressed dataset files.." << std::endl;

        const std::vector<std::filesystem::path> datasetFiles = ShapeDescriptor::listDirectoryAndSubdirectories(originalDatasetDirectory);
        nlohmann::json datasetCache = {};
        bool previousCacheFound = std::filesystem::exists(metadataFile);

        if(previousCacheFound) {
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

#pragma omp parallel for schedule(dynamic) default(none) shared(processedMeshCount, newMeshesLoaded, std::cout, datasetFiles, originalDatasetDirectory, datasetCache, compressedDatasetDirectory, pointCloudCount, metadataFile)
        for(size_t i = 0; i < datasetFiles.size(); i++) {


            // Skip if this entry has been processed before
            if(!datasetCache["files"].at(i).contains("isPointCloud")) {
                newMeshesLoaded = true;
                nlohmann::json datasetEntry;
                datasetEntry["id"] = i;
                std::filesystem::path filePath = std::filesystem::relative(std::filesystem::absolute(datasetFiles.at(i)), originalDatasetDirectory);
                bool isPointCloud = false;
                if(filePath.extension() == ".glb") {
                    isPointCloud = ShapeDescriptor::gltfContainsPointCloud(datasetFiles.at(i));
                }
                datasetEntry["filePath"] = filePath;

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
                        datasetEntry["sha1"] = SHA1::from_file(compressedMeshPath.string());
                        ShapeDescriptor::free(cloud);
                    } else {
                        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(datasetFiles.at(i));
                        ShapeDescriptor::writeCompressedGeometryFile(mesh, compressedMeshPath, true);
                        datasetEntry["vertexCount"] = mesh.vertexCount;
                        datasetEntry["sha1"] = SHA1::from_file(compressedMeshPath.string());
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


        return datasetCache;
    }

}