

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
#include "tiny_gltf.h"
#include <nlohmann/json.hpp>

bool meshIsPointCloud(const std::filesystem::path& file) {
    std::ifstream inputStream{file};

    std::array<unsigned int, 3> fileHeader {0, 0, 0};
    inputStream.read((char*)fileHeader.data(), sizeof(fileHeader));
    assert(fileHeader.at(0) == 0x46546C67);
    assert(fileHeader.at(1) == 2);
    unsigned int totalSize = fileHeader.at(2);

    unsigned int headerChunkLength;
    unsigned int ignored_headerChunkType;
    inputStream.read((char*) &headerChunkLength, sizeof(unsigned int));
    inputStream.read((char*) &ignored_headerChunkType, sizeof(unsigned int));

    std::string jsonChunkContents;
    jsonChunkContents.resize(headerChunkLength);
    inputStream.read(jsonChunkContents.data(), headerChunkLength);

    nlohmann::json jsonHeader = nlohmann::json::parse(jsonChunkContents);

    for(const nlohmann::json& meshElement : jsonHeader.at("meshes")) {
        for(const nlohmann::json& primitive : meshElement.at("primitives")) {
            if(primitive.at("mode") == TINYGLTF_MODE_POINTS) {
                return true;
            }
        }
    }
    return false;
}

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


    if(!configuration.contains("datasetRootDir")) {
        throw Shapebench::MissingBenchmarkConfigurationException("datasetRootDir");
    }
    const std::filesystem::path datasetDirectory = configuration.at("datasetRootDir");
    const std::filesystem::path datasetCacheFile = cacheDirectory / Shapebench::datasetCacheFileName;
    if(!std::filesystem::exists(datasetCacheFile) || true) {
        std::cout << "No dataset cache file was found. Analysing dataset.." << std::endl;
        const std::vector<std::filesystem::path> datasetFiles = ShapeDescriptor::utilities::listDirectoryAndSubdirectories(datasetDirectory);

        // TODO: once data is available, do a hash check of all files
        std::cout << "    Found " << datasetFiles.size() << " files." << std::endl;

        nlohmann::json datasetCache = {};
        datasetCache["metadata"]["datasetDirectory"] = std::filesystem::absolute(datasetDirectory).string();
        datasetCache["metadata"]["configurationFile"] = std::filesystem::absolute(configurationFileLocation).string();
        datasetCache["metadata"]["cacheDirectory"] = std::filesystem::absolute(cacheDirectory).string();

        datasetCache["files"] = {};
        size_t nextID = 0;
        for(size_t i = 0; i < datasetFiles.size(); i++) {
            std::cout << "\r    Computing dataset cache.. Processed: " << (i+1) << "/" << datasetFiles.size() << " (" << (100.0*(double(i+1)/double(datasetFiles.size()))) << "%)" << std::flush;

            nlohmann::json datasetEntry;
            datasetEntry["id"] = nextID;
            datasetEntry["filePath"] = std::filesystem::absolute(datasetFiles.at(i));
            bool isPointCloud = meshIsPointCloud(datasetFiles.at(i));
            datasetEntry["isPointCloud"] = isPointCloud;

            datasetCache["files"].push_back(datasetEntry);
            nextID++;
        }

        std::ofstream outCacheStream {datasetCacheFile};
        outCacheStream << datasetCache.dump(4);
    }



    std::cout << "Performing parameter estimation.." << std::endl;


}
