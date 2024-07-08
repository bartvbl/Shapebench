#include "arrrgh.hpp"
#include "dataset/Dataset.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include "benchmarkCore/MissingBenchmarkConfigurationException.h"
#include "benchmarkCore/constants.h"
#include "methods/QUICCIMethod.h"
#include "methods/SIMethod.h"
#include "filters/additiveNoise/additiveNoiseFilter.h"
#include "benchmarkCore/experimentRunner.h"
#include "methods/3DSCMethod.h"
#include "methods/RoPSMethod.h"
#include "methods/RICIMethod.h"
#include "methods/USCMethod.h"
#include "methods/SHOTMethod.h"
#include "benchmarkCore/BenchmarkConfiguration.h"
#include "dataset/DatasetLoader.h"


nlohmann::json readConfiguration(std::filesystem::path filePath);
void patchReplicationConfiguration(nlohmann::json &replicationConfig, nlohmann::json &regularConfig, const ShapeBench::ReplicationSettings& replicationSettings);


int main(int argc, const char** argv) {
    const std::string replicationDisabledString = "UNSPECIFIED";

    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& configurationFile = parser.add<std::string>(
            "configuration-file", "Location of the file from which to read the experimental configuration", '\0', arrrgh::Optional, "../cfg/config.json");
    const auto& replicateFromConfiguration = parser.add<std::string>(
            "replicate-results-file", "Path to a results file that should be replicated. Enables replication mode. Overwrites the configuration specified in the --configuration-file parameter, except for a few specific entries that are system specific.", '\0', arrrgh::Optional, replicationDisabledString);

    try {
        parser.parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    ShapeBench::BenchmarkConfiguration setup;
    setup.replicationFilePath = replicateFromConfiguration.value();
    setup.configurationFilePath = configurationFile.value();


    // --- Initialise replication mode ---
    if(setup.replicationFilePath != replicationDisabledString) {
        if(!std::filesystem::exists(setup.replicationFilePath)) {
            throw std::runtime_error("The specified results file does not appear to exist. Exiting.");
        }

        std::cout << "Replication mode enabled." << std::endl;
        setup.replicationSettings.enabled = true;
        std::cout << "    This will mostly override the specified configuration in the main configuration file." << std::endl;
        std::cout << "    Reading results file.." << std::endl;
        std::ifstream inputStream(setup.replicationFilePath);
        nlohmann::json resultsFileContents = nlohmann::json::parse(inputStream);
        setup.configuration = resultsFileContents.at("configuration");
        setup.computedConfiguration = resultsFileContents.at("computedConfiguration");
        setup.replicationSettings.methodName = resultsFileContents.at("method").at("name");
        setup.replicationSettings.experimentIndex = resultsFileContents.at("experiment").at("index");
    }



    // --- Compute experiment configuration ---
    std::cout << "Reading configuration.." << std::endl;
    const std::filesystem::path configurationFileLocation(configurationFile.value());
    bool configurationFileExists = std::filesystem::exists(configurationFileLocation);
    if(!configurationFileExists && !setup.replicationSettings.enabled) {
        throw std::runtime_error("The specified configuration file was not found at: " + std::filesystem::absolute(configurationFileLocation).string());
    }

    nlohmann::json mainConfigFileContents;
    if(configurationFileExists) {
        mainConfigFileContents = readConfiguration(configurationFile.value());
    }

    if(!setup.replicationSettings.enabled) {
        setup.configuration = mainConfigFileContents;
        if(!setup.configuration.contains("cacheDirectory")) {
            throw ShapeBench::MissingBenchmarkConfigurationException("cacheDirectory");
        }
    } else {
        // For the purposes of replication, some configuration entries need to be adjusted to the
        // environment where the results are replicated. This is done by copying all relevant configuration
        // entries, and overwriting them where relevant.
        patchReplicationConfiguration(setup.configuration, mainConfigFileContents, setup.replicationSettings);
    }



    // --- Compute and load dataset ---
    setup.dataset = ShapeBench::computeOrLoadCache(setup);

    double datasetSizeLimitGB = setup.configuration.at("datasetSettings").at("cacheSizeLimitGB");
    uint64_t datasetSizeLimitBytes = uint64_t(datasetSizeLimitGB * 1024.0 * 1024.0 * 1024.0);
    std::filesystem::path datasetCacheDirectory = setup.configuration.at("datasetSettings").at("compressedRootDir");
    std::string datasetDownloadBaseURL = setup.configuration.at("datasetSettings").at("objaverseDownloadBaseURL");
    ShapeBench::LocalDatasetCache* fileCache = new ShapeBench::LocalDatasetCache(datasetCacheDirectory, datasetDownloadBaseURL,  datasetSizeLimitBytes);



    // --- Run experiments ---
    const nlohmann::json& methodSettings = setup.configuration.at("methodSettings");
    if(methodSettings.at(ShapeBench::QUICCIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::QUICCIMethod, ShapeDescriptor::QUICCIDescriptor>(setup, fileCache);
    }
    if(methodSettings.at(ShapeBench::RICIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::RICIMethod, ShapeDescriptor::RICIDescriptor>(setup, fileCache);
    }
    if(methodSettings.at(ShapeBench::RoPSMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::RoPSMethod, ShapeDescriptor::RoPSDescriptor>(setup, fileCache);
    }
    if(methodSettings.at(ShapeBench::SIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::SIMethod, ShapeDescriptor::SpinImageDescriptor>(setup, fileCache);
    }
    if(methodSettings.at(ShapeBench::USCMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::USCMethod, ShapeDescriptor::UniqueShapeContextDescriptor>(setup, fileCache);
    }
    if(methodSettings.at(ShapeBench::ShapeContextMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::ShapeContextMethod, ShapeDescriptor::ShapeContextDescriptor>(setup, fileCache);
    }
    if(methodSettings.at(ShapeBench::SHOTMethod<>::getName()).at("enabled")) {
        testMethod<ShapeBench::SHOTMethod<>, ShapeDescriptor::SHOTDescriptor<>>(setup, fileCache);
    }

    // Disabled and WIP. Horrendously slow.
    //testMethod<ShapeBench::FPFHMethod, ShapeDescriptor::FPFHDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);

    delete fileCache;
}

void patchKey(nlohmann::json& replicationConfig, nlohmann::json& regularConfig, std::string key) {
    if(regularConfig.contains(key)) {
        replicationConfig[key].merge_patch(regularConfig.at(key));
    }
}

void patchReplicationConfiguration(nlohmann::json &replicationConfig, nlohmann::json &regularConfig, const ShapeBench::ReplicationSettings& replicationSettings) {
    patchKey(replicationConfig, regularConfig, "replicationOverrides");
    patchKey(replicationConfig, regularConfig, "datasetSettings");
    patchKey(replicationConfig, regularConfig, "cacheDirectory");
    patchKey(replicationConfig, regularConfig, "resultsDirectory");
    patchKey(replicationConfig, regularConfig, "computedConfigFile");
    patchKey(replicationConfig, regularConfig, "verboseOutput");

    // Each results file only contains one active method and experiment
    // So we ensure here that in both cases only one is marked as active
    // We first enable the one active method
    for(const auto& methodEntry : replicationConfig.at("methodSettings").items()) {
        replicationConfig["methodSettings"][methodEntry.key()]["enabled"] = methodEntry.key() == replicationSettings.methodName;
    }

    // And ensure here that only the experiment being replicated is active
    for(int i = 0; i < replicationConfig.at("experimentsToRun").size(); i++) {
        replicationConfig.at("experimentsToRun").at(i).at("enabled") = i == replicationSettings.experimentIndex;
    }
}

nlohmann::json readConfiguration(std::filesystem::path filePath) {
    std::ifstream inputStream(filePath);
    nlohmann::json configuration = nlohmann::json::parse(inputStream);
    nlohmann::json originalConfiguration = configuration;
    if(configuration.contains("includes")) {
        std::filesystem::path containingDirectory = filePath.parent_path();
        for(std::string dependencyPathString : configuration.at("includes")) {
            nlohmann::json subConfiguration = readConfiguration(containingDirectory / dependencyPathString);
            configuration.merge_patch(subConfiguration);
        }
        // Delete includes key to ensure it does not get parsed twice
        configuration.erase("includes");
        originalConfiguration.erase("includes");
    }
    // We want the base file to override any values provided by any file it includes
    configuration.merge_patch(originalConfiguration);
    return configuration;
}
