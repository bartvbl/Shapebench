#include "arrrgh.hpp"
#include <shapeDescriptor/shapeDescriptor.h>
#include "benchmarkCore/MissingBenchmarkConfigurationException.h"
#include "benchmarkCore/constants.h"
#include "benchmarkCore/CompressedDatasetCreator.h"
#include "benchmarkCore/Dataset.h"
#include "methods/QUICCIMethod.h"
#include "methods/SIMethod.h"
#include "benchmarkCore/ComputedConfig.h"
#include "filters/additiveNoise/additiveNoiseFilter.h"
#include "benchmarkCore/experimentRunner.h"
#include "methods/3DSCMethod.h"


int main(int argc, const char** argv) {
    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& configurationFile = parser.add<std::string>(
            "configuration-file", "Location of the file from which to read the experimental configuration", '\0', arrrgh::Optional, "../cfg/config.json");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Use the GPU with the given ID (as shown in nvidia-smi)", '\0', arrrgh::Optional, 0);

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

    if(!ShapeDescriptor::isCUDASupportAvailable()) {
        throw std::runtime_error("This benchmark requires CUDA support to operate.");
    }

    //ShapeDescriptor::createCUDAContext(forceGPU.value());

    // ---------------------------------------------------------

    // Special case meshes used as sanity checks
    //ShapeDescriptor::cpu::Mesh inMesh = ShapeDescriptor::loadMesh("/mnt/DATASETS/objaverse/hf-objaverse-v1/glbs/000-031/006eccb258c94370bdfd26205491d135.glb");
    //ShapeDescriptor::writeOBJ(inMesh, "smallMesh.obj");
    //inMesh = ShapeDescriptor::loadMesh("/mnt/DATASETS/objaverse/hf-objaverse-v1/glbs/000-000/0554eaf578284d2b8fa3493a1f1d56c6.glb");
    //ShapeDescriptor::writeOBJ(inMesh, "alarm.obj");

    std::cout << "Reading configuration.." << std::endl;
    const std::filesystem::path configurationFileLocation(configurationFile.value());
    if(!std::filesystem::exists(configurationFileLocation)) {
        throw std::runtime_error("The specified configuration file was not found at: " + std::filesystem::absolute(configurationFileLocation).string());
    }
    std::ifstream inputStream(configurationFile.value());
    const nlohmann::json configuration = nlohmann::json::parse(inputStream);


    if(!configuration.contains("cacheDirectory")) {
        throw ShapeBench::MissingBenchmarkConfigurationException("cacheDirectory");
    }
    const std::filesystem::path cacheDirectory = configuration.at("cacheDirectory");
    if(!std::filesystem::exists(cacheDirectory)) {
        std::cout << "    Cache directory was not found. Creating a new one at: " << cacheDirectory.string() << std::endl;
        std::filesystem::create_directories(cacheDirectory);
    }


    if(!configuration.contains("compressedDatasetRootDir")) {
        throw ShapeBench::MissingBenchmarkConfigurationException("compressedDatasetRootDir");
    }
    const std::filesystem::path baseDatasetDirectory = configuration.at("objaverseDatasetRootDir");
    const std::filesystem::path derivedDatasetDirectory = configuration.at("compressedDatasetRootDir");
    const std::filesystem::path datasetCacheFile = cacheDirectory / ShapeBench::datasetCacheFileName;

    ShapeBench::Dataset dataset = ShapeBench::Dataset::computeOrLoadCached(baseDatasetDirectory, derivedDatasetDirectory, datasetCacheFile);

    ShapeBench::initPhysics();

    uint64_t randomSeed = configuration.at("randomSeed");
    testMethod<ShapeBench::QUICCIMethod, ShapeDescriptor::QUICCIDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    testMethod<ShapeBench::SIMethod, ShapeDescriptor::SpinImageDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    //testMethod<ShapeBench::ShapeContextMethod, ShapeDescriptor::ShapeContextDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
}
