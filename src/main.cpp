

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
#include "shapeDescriptor/utilities/write/CompressedGeometryFile.h"
#include "shapeDescriptor/utilities/read/PointCloudLoader.h"
#include "shapeDescriptor/utilities/free/pointCloud.h"
#include "benchmark-core/CompressedDatasetCreator.h"
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
    if(!std::filesystem::exists(datasetCacheFile) || !std::filesystem::exists(derivedDatasetDirectory) || true) {
        std::cout << "Dataset metadata or compressed dataset was not found." << std::endl
                  << "Computing compressed dataset.. (this will likely take multiple hours)" << std::endl;
        Shapebench::computeCompressedDataSet(baseDatasetDirectory, derivedDatasetDirectory, datasetCacheFile);
    }



    std::cout << "Performing parameter estimation.." << std::endl;


}
