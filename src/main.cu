

#include "arrrgh.hpp"
#include <shapeDescriptor/shapeDescriptor.h>
#include "benchmark-core/MissingBenchmarkConfigurationException.h"
#include "benchmark-core/constants.h"
#include <tiny_gltf.h>
#include "benchmark-core/CompressedDatasetCreator.h"
#include "benchmark-core/Dataset.h"
#include "benchmark-core/SupportRadiusEstimation.h"
#include "methods/QUICCIMethod.h"
#include "methods/SIMethod.h"
#include "benchmark-core/ComputedConfig.h"
#include <memory>
#include <nlohmann/json.hpp>
#include <random>

template<typename DescriptorMethod, typename DescriptorType>
void testMethod(const nlohmann::json& configuration, const std::filesystem::path configFileLocation, const Dataset& dataset, uint64_t randomSeed) {
    std::mt19937_64 engine(randomSeed);
    std::filesystem::path computedConfigFilePath = configFileLocation.parent_path() / std::string(configuration.at("computedConfigFile"));
    std::cout << "Main config file: " << configFileLocation.string() << std::endl;
    std::cout << "Computed values config file: " << computedConfigFilePath.string() << std::endl;
    ComputedConfig computedConfig(computedConfigFilePath);
    const std::string methodName = DescriptorMethod::getName();

    // Getting a support radius

    float supportRadius = 0;
    if(!computedConfig.containsKey(methodName, "supportRadius")) {
        std::cout << "No support radius has been computed yet for this method." << std::endl;
        std::cout << "Performing support radius estimation.." << std::endl;
        supportRadius = Shapebench::estimateSupportRadius<DescriptorMethod, DescriptorType>(configuration, dataset, engine());
        std::cout << "    Chosen support radius: " << supportRadius << std::endl;
        computedConfig.setFloatAndSave(methodName, "supportRadius", supportRadius);
    } else {
        supportRadius = computedConfig.getFloat(methodName, "supportRadius");
    }

    //

}

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

    ShapeDescriptor::createCUDAContext(forceGPU.value());

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
    if(!std::filesystem::exists(datasetCacheFile) || !std::filesystem::exists(derivedDatasetDirectory)) {
        std::cout << "Dataset metadata or compressed dataset was not found." << std::endl
                  << "Computing compressed dataset.. (this will likely take multiple hours)" << std::endl;
        Shapebench::computeCompressedDataSet(baseDatasetDirectory, derivedDatasetDirectory, datasetCacheFile);
    }


    std::cout << "Reading dataset cache.." << std::endl;
    Dataset dataset;
    dataset.load(datasetCacheFile);


    uint64_t randomSeed = configuration.at("randomSeed");
    testMethod<Shapebench::QUICCIMethod, ShapeDescriptor::QUICCIDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    //testMethod<Shapebench::SIMethod, ShapeDescriptor::SpinImageDescriptor>(configuration, dataset, randomSeed);


}
