#include <iostream>
#include <vector>
#include "CompressedDatasetCreator.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "utils/prettyprint.h"
#include <Seb.h>
#include <malloc.h>

nlohmann::json ShapeBench::computeOrReadDatasetCache(const std::filesystem::path &originalDatasetDirectory,
                                           const std::filesystem::path &compressedDatasetDirectory,
                                           std::filesystem::path metadataFile) {

    std::cout << "Searching for uncompressed dataset files.." << std::endl;
    const std::vector<std::filesystem::path> datasetFiles = ShapeDescriptor::listDirectoryAndSubdirectories(originalDatasetDirectory);

    nlohmann::json datasetCache = {};
    bool previousCacheFound = std::filesystem::exists(metadataFile);
    if(previousCacheFound) {
        std::cout << "Loading dataset cache.. (found " << datasetFiles.size() << " files)" << std::endl;
        std::ifstream inputStream{metadataFile};
        datasetCache = nlohmann::json::parse(inputStream);

        if(datasetFiles.empty()) {
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
    unsigned int hashMismatches = 0;
    size_t processedMeshCount = 0;
    bool newMeshesLoaded = false;

    std::chrono::time_point<std::chrono::steady_clock> startTime = std::chrono::steady_clock::now();

    #pragma omp parallel for schedule(dynamic) default(none) shared(hashMismatches, processedMeshCount, newMeshesLoaded, std::cout, datasetFiles, originalDatasetDirectory, datasetCache, compressedDatasetDirectory, pointCloudCount, metadataFile)
    for(size_t i = 0; i < datasetFiles.size(); i++) {


        // Skip if this entry has been processed before
        if(!datasetCache["files"].at(i).contains("isPointCloud")) {
            newMeshesLoaded = true;
            nlohmann::json datasetEntry;
            datasetEntry["id"] = i;
            std::filesystem::path filePath = std::filesystem::relative(std::filesystem::absolute(datasetFiles.at(i)),
                                                                       originalDatasetDirectory);
            datasetEntry["filePath"] = filePath;
            bool isPointCloud = ShapeDescriptor::gltfContainsPointCloud(datasetFiles.at(i));
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

                    // Integrity check
                    ShapeDescriptor::cpu::PointCloud readCloud = ShapeDescriptor::readPointCloudFromCompressedGeometryFile(
                            compressedMeshPath);
                    if (ShapeDescriptor::comparePointCloud(cloud, readCloud)) {
                        std::cout << "\n!! POINT CLOUD HASH MISMATCH " + compressedMeshPath.string() + "\n"
                                  << std::flush;
                        #pragma omp atomic
                        hashMismatches++;
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

                    // Integrity check
                    ShapeDescriptor::cpu::Mesh readMesh = ShapeDescriptor::loadMesh(compressedMeshPath);

                    if (!ShapeDescriptor::compareMesh(mesh, readMesh)) {
                        std::cout << "\n!! MESH HASH MISMATCH " + compressedMeshPath.string() + "\n" << std::flush;
                        #pragma omp atomic
                        hashMismatches++;
                        ShapeDescriptor::writeOBJ(readMesh, compressedMeshPath.replace_extension(".obj"));
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
                if ((i + 1) % 2500 == 0) {
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
                std::cout << " " << (processedMeshCount+1) << "/" << datasetFiles.size() << " (" << std::round(10000.0*(double(processedMeshCount+1)/double(datasetFiles.size())))/100.0 << "%)";
                if(newMeshesLoaded) {
                    std::cout << ", found " << pointCloudCount << " point clouds, " << hashMismatches << " mismatched hashes";
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
    } else {
        std::cout << "    Dataset cache loaded successfully" << std::endl;
    }


    std::ofstream outCacheStream {metadataFile};
    outCacheStream << datasetCache.dump(4);

    return datasetCache;
}
