#include "DatasetLoader.h"
#include "benchmarkCore/MissingBenchmarkConfigurationException.h"
#include "benchmarkCore/constants.h"
#include "CompressedDatasetCreator.h"

ShapeBench::Dataset ShapeBench::computeOrLoadCache(const ShapeBench::BenchmarkConfiguration& setup) {
    const std::filesystem::path cacheDirectory = setup.configuration.at("cacheDirectory");
    if(!std::filesystem::exists(cacheDirectory)) {
        std::cout << "    Cache directory was not found. Creating a new one at: " << cacheDirectory.string() << std::endl;
        std::filesystem::create_directories(cacheDirectory);
    }

    if(!setup.configuration.at("datasetSettings").contains("compressedRootDir")) {
        throw ShapeBench::MissingBenchmarkConfigurationException("compressedRootDir");
    }
    const std::filesystem::path baseDatasetDirectory = setup.configuration.at("datasetSettings").at("objaverseRootDir");
    const std::filesystem::path derivedDatasetDirectory = setup.configuration.at("datasetSettings").at("compressedRootDir");
    const std::filesystem::path datasetCacheFile = cacheDirectory / ShapeBench::datasetCacheFileName;
    nlohmann::json replicationConfiguration = setup.configuration.contains("replicationOverrides")
                                                 && setup.configuration.at("replicationOverrides").contains("datasetCache")
                                                 ? setup.configuration.at("replicationOverrides").at("datasetCache") : nlohmann::json();
    uint64_t replicationRandomSeed = setup.configuration.at("replicationOverrides").at("replicationRandomSeed");
    nlohmann::json datasetCacheJson = ShapeBench::computeOrReadDatasetCache(replicationConfiguration, baseDatasetDirectory, derivedDatasetDirectory, datasetCacheFile, replicationRandomSeed);
    ShapeBench::Dataset dataset;
    dataset.loadCache(datasetCacheJson);
    return dataset;
}