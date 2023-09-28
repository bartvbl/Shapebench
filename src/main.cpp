

#include "arrrgh.hpp"
#include "shapeDescriptor/utilities/CUDAContextCreator.h"
#include "shapeDescriptor/utilities/CUDAAvailability.h"
#include "shapeDescriptor/utilities/read/GLTFLoader.h"
#include "shapeDescriptor/utilities/dump/meshDumper.h"
#include "shapeDescriptor/utilities/fileutils.h"
#include "shapeDescriptor/utilities/free/mesh.h"
#include "benchmark-core/MissingBenchmarkConfigurationException.h"
#include "benchmark-core/constants.h"
#include "shapeDescriptor/utilities/read/MeshLoader.h"
#include <tiny_gltf.h>
#include "shapeDescriptor/utilities/write/CompressedMesh.h"
#include "shapeDescriptor/utilities/read/PointCloudLoader.h"
#include "shapeDescriptor/utilities/free/pointCloud.h"
#include <memory>
#include <nlohmann/json.hpp>




int main(int argc, const char** argv) {
    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& configurationFile = parser.add<std::string>(
            "configuration-file", "Location of the file from which to read the experimental configuration", '\0', arrrgh::Optional, "../cfg/config.json");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Use the GPU with the given ID (as shown in nvidia-smi)", '\0', arrrgh::Optional, -1);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    if(forceGPU.value() != -1) {
        std::cout << "Forcing GPU " << forceGPU.value() << std::endl;
        ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    if(!ShapeDescriptor::isCUDASupportAvailable()) {
        throw std::runtime_error("This benchmark requires CUDA support to operate.");
    }

    // ---------------------------------------------------------

    std::cout << "Reading configuration.." << std::endl;
    const std::filesystem::path configurationFileLocation(configurationFile.value());
    if(!std::filesystem::exists(configurationFileLocation)) {
        throw std::runtime_error("The specified configuration file was not found at: " + std::filesystem::absolute(configurationFileLocation).string());
    }
    std::ifstream inputStream(configurationFile.value());
    const nlohmann::json configuration = nlohmann::json::parse(inputStream);


    if(!configuration.contains("cacheDirectory")) {
        throw Shapebench::MissingBenchmarkConfigurationException("cacheDirectory");
    }
    const std::filesystem::path cacheDirectory = configuration.at("cacheDirectory");
    if(!std::filesystem::exists(cacheDirectory)) {
        std::cout << "    Cache directory was not found. Creating a new one at: " << cacheDirectory.string() << std::endl;
        std::filesystem::create_directories(cacheDirectory);
    }


    if(!configuration.contains("compressedDatasetRootDir")) {
        throw Shapebench::MissingBenchmarkConfigurationException("compressedDatasetRootDir");
    }
    const std::filesystem::path baseDatasetDirectory = configuration.at("objaverseDatasetRootDir");
    const std::filesystem::path derivedDatasetDirectory = configuration.at("compressedDatasetRootDir");
    const std::filesystem::path datasetCacheFile = cacheDirectory / Shapebench::datasetCacheFileName;
    if(!std::filesystem::exists(datasetCacheFile) || true) {
        std::cout << "No dataset cache file was found. Analysing dataset.. (this will likely take multiple hours)" << std::endl;
        const std::vector<std::filesystem::path> datasetFiles = ShapeDescriptor::utilities::listDirectoryAndSubdirectories(baseDatasetDirectory);

        // TODO: once data is available, do a hash check of all files
        std::cout << "    Found " << datasetFiles.size() << " files." << std::endl;

        nlohmann::json datasetCache = {};
        datasetCache["metadata"]["baseDatasetRootDir"] = std::filesystem::absolute(baseDatasetDirectory).string();
        datasetCache["metadata"]["compressedDatasetRootDir"] = std::filesystem::absolute(derivedDatasetDirectory).string();
        datasetCache["metadata"]["configurationFile"] = std::filesystem::absolute(configurationFileLocation).string();
        datasetCache["metadata"]["cacheDirectory"] = std::filesystem::absolute(cacheDirectory).string();

        datasetCache["files"] = {};
        size_t nextID = 0;
        unsigned int pointCloudCount = 0;
        size_t processedMeshCount = 0;
        #pragma omp parallel for schedule(dynamic) default(none) shared(processedMeshCount, std::cout, datasetFiles, baseDatasetDirectory, nextID, datasetCache, derivedDatasetDirectory, pointCloudCount, datasetCacheFile)
        for(size_t i = 0; i < datasetFiles.size(); i++) {
            #pragma omp atomic
            processedMeshCount++;
            if(processedMeshCount % 100 == 99) {
                std::cout << "\r    Computing dataset cache.. Processed: " << (processedMeshCount+1) << "/" << datasetFiles.size() << " (" << std::round(10000.0*(double(processedMeshCount+1)/double(datasetFiles.size())))/100.0 << "%), excluded " << pointCloudCount << " point clouds" << std::flush;
            }

            nlohmann::json datasetEntry;
            datasetEntry["id"] = nextID;
            std::filesystem::path filePath = std::filesystem::relative(std::filesystem::absolute(datasetFiles.at(i)), baseDatasetDirectory);
            datasetEntry["filePath"] = filePath;
            bool isPointCloud = ShapeDescriptor::utilities::gltfContainsPointCloud(datasetFiles.at(i));
            datasetEntry["isPointCloud"] = isPointCloud;
            std::filesystem::path compressedMeshPath = derivedDatasetDirectory / filePath;
            compressedMeshPath.replace_extension(".cm");
            if(isPointCloud) {
                #pragma omp atomic
                pointCloudCount++;
                ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::utilities::loadPointCloud(datasetFiles.at(i));
                ShapeDescriptor::utilities::writeCompressedGeometryFile(cloud, compressedMeshPath, true);
                ShapeDescriptor::free::pointCloud(cloud);
            } else {
                ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadMesh(datasetFiles.at(i));
                ShapeDescriptor::utilities::writeCompressedGeometryFile(mesh, compressedMeshPath, true);
                ShapeDescriptor::free::mesh(mesh);
            }
            #pragma omp critical
            {
                datasetCache["files"].push_back(datasetEntry);
                nextID++;

                if((nextID + 1) % 50000 == 0) {
                    std::cout << std::endl << "Writing backup JSON.. " << std::endl;
                    std::ofstream outCacheStream {datasetCacheFile};
                    outCacheStream << datasetCache.dump(4);
                }
            };
        }
        std::cout << std::endl;

        std::ofstream outCacheStream {datasetCacheFile};
        outCacheStream << datasetCache.dump(4);
    }



    std::cout << "Performing parameter estimation.." << std::endl;


}
