#pragma once

#include <random>
#include <iostream>
#include "json.hpp"
#include "Dataset.h"
#include "ComputedConfig.h"
#include "support-radius-estimation/SupportRadiusEstimation.h"
#include "utils/progressBar.h"

template<typename DescriptorMethod, typename DescriptorType>
ShapeDescriptor::cpu::array<DescriptorType> computeReferenceDescriptors(const std::vector<ShapeBench::VertexInDataset>& representativeSet, const nlohmann::json& config, const ShapeBench::Dataset& dataset, uint64_t randomSeed, float supportRadius) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ShapeBench::randomEngine randomEngine(randomSeed);
    std::vector<uint64_t> randomSeeds(representativeSet.size());
    for(uint32_t i = 0; i < representativeSet.size(); i++) {
        randomSeeds.at(i) = randomEngine();
    }

    ShapeDescriptor::cpu::array<DescriptorType> representativeDescriptors(representativeSet.size());
    uint32_t completedCount = 0;
    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < representativeSet.size(); i++) {
        ShapeDescriptor::OrientedPoint descriptorOrigin;
        ShapeBench::VertexInDataset vertex = representativeSet.at(i);
        const ShapeBench::DatasetEntry& entry = dataset.at(vertex.meshID);
        ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(config, entry);
        descriptorOrigin.vertex = mesh.vertices[vertex.vertexIndex];
        descriptorOrigin.normal = mesh.normals[vertex.vertexIndex];
        representativeDescriptors[i] = ShapeBench::computeSingleDescriptor<DescriptorMethod, DescriptorType>(mesh, descriptorOrigin, config, supportRadius, randomSeeds.at(i));
        ShapeDescriptor::free(mesh);

        #pragma omp atomic
        completedCount++;

        if(completedCount % 100 == 0 || completedCount == representativeSet.size()) {
            std::cout << "\r    ";
            ShapeBench::drawProgressBar(completedCount, representativeSet.size());
            std::cout << " " << completedCount << "/" << representativeSet.size() << std::flush;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    uint32_t timeInSeconds = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    uint32_t timeInHours = timeInSeconds / 3600;
    timeInSeconds -= timeInHours * 3600;
    uint32_t timeInMinutes = timeInSeconds / 60;
    timeInSeconds -= timeInMinutes * 60;
    std::cout << std::endl;
    std::cout << "    Complete." << std::endl;
    std::cout << "    Elapsed time: " << timeInHours << ":" << timeInMinutes << ":" << timeInSeconds << std::endl;

    return representativeDescriptors;
}

template<typename DescriptorType, typename DescriptorMethod>
ShapeDescriptor::cpu::array<DescriptorType> computeDescriptorsOrLoadCached(
        const nlohmann::json &configuration,
        const ShapeBench::Dataset &dataset,
        float supportRadius,
        uint64_t representativeSetRandomSeed,
        const std::vector<ShapeBench::VertexInDataset> &representativeSet,
        std::string name) {
    const std::filesystem::path descriptorCacheFile = std::filesystem::path(std::string(configuration.at("cacheDirectory"))) / (name + "Descriptors-" + DescriptorMethod::getName() + ".dat");
    ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors;
    if(!std::filesystem::exists(descriptorCacheFile)) {
        std::cout << "    No cached " + name + " descriptors were found." << std::endl;
        std::cout << "    Computing " + name + " descriptors.." << std::endl;
        referenceDescriptors = computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, configuration, dataset, representativeSetRandomSeed, supportRadius);
        std::cout << "    Finished computing " + name + " descriptors. Writing archive file.." << std::endl;
        ShapeDescriptor::writeCompressedDescriptors<DescriptorType>(descriptorCacheFile, referenceDescriptors);

        std::cout << "    Checking integrity of written data.." << std::endl;
        std::cout << "        Reading written file.." << std::endl;
        ShapeDescriptor::cpu::array<DescriptorType> readDescriptors = ShapeDescriptor::readCompressedDescriptors<DescriptorType>(descriptorCacheFile, 8);
        assert(referenceDescriptors.length == readDescriptors.length);
        for(uint32_t i = 0; i < referenceDescriptors.length; i++) {
            char* basePointerA = reinterpret_cast<char*>(&referenceDescriptors.content[i]);
            char* basePointerB = reinterpret_cast<char*>(&readDescriptors.content[i]);
            for(uint32_t j = 0; j < sizeof(DescriptorType); j++) {
                if(basePointerA[j] != basePointerB[j]) {
                    throw std::runtime_error("Descriptors at index " + std::to_string(i) + " are not identical!");
                }
            }
        }
        std::cout << "    Check complete, no errors detected." << std::endl;
        ShapeDescriptor::free(readDescriptors);
    } else {
        std::cout << "    Loading cached " + name + " descriptors.." << std::endl;
        referenceDescriptors = ShapeDescriptor::readCompressedDescriptors<DescriptorType>(descriptorCacheFile, 8);
    }
    return referenceDescriptors;
}

template<typename DescriptorMethod, typename DescriptorType>
void testMethod(const nlohmann::json& configuration, const std::filesystem::path configFileLocation, const ShapeBench::Dataset& dataset, uint64_t randomSeed) {
    std::cout << std::endl << "========== TESTING METHOD " << DescriptorMethod::getName() << " ==========" << std::endl;
    std::cout << "Initialising.." << std::endl;
    ShapeBench::randomEngine engine(randomSeed);
    std::filesystem::path computedConfigFilePath = configFileLocation.parent_path() / std::string(configuration.at("computedConfigFile"));
    std::cout << "    Main config file: " << configFileLocation.string() << std::endl;
    std::cout << "    Computed values config file: " << computedConfigFilePath.string() << std::endl;
    ShapeBench::ComputedConfig computedConfig(computedConfigFilePath);
    const std::string methodName = DescriptorMethod::getName();

    // Getting a support radius
    std::cout << "Determining support radius.." << std::endl;
    float supportRadius = 0;
    if(!computedConfig.containsKey(methodName, "supportRadius")) {
        std::cout << "    No support radius has been computed yet for this method." << std::endl;
        std::cout << "    Performing support radius estimation.." << std::endl;
        supportRadius = ShapeBench::estimateSupportRadius<DescriptorMethod, DescriptorType>(configuration, dataset, engine());
        std::cout << "    Chosen support radius: " << supportRadius << std::endl;
        computedConfig.setFloatAndSave(methodName, "supportRadius", supportRadius);
    } else {
        supportRadius = computedConfig.getFloat(methodName, "supportRadius");
        std::cout << "    Cached support radius was found for this method: " << supportRadius << std::endl;
    }

    // Computing sample descriptors and their distance to the representative set
    uint32_t representativeSetSize = configuration.at("commonExperimentSettings").at("representativeSetSize");
    uint32_t sampleSetSize = configuration.at("commonExperimentSettings").at("sampleSetSize");

    // Compute reference descriptors, or load them from a cache file
    std::cout << "Computing reference descriptor set.." << std::endl;
    std::vector<ShapeBench::VertexInDataset> representativeSet = dataset.sampleVertices(engine(), representativeSetSize);
    ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod>(configuration, dataset, supportRadius, engine(), representativeSet, "reference");

    // Computing sample descriptors, or load them from a cache file
    std::vector<ShapeBench::VertexInDataset> sampleVerticesSet = dataset.sampleVertices(engine(), sampleSetSize);
    //ShapeDescriptor::cpu::array<DescriptorType> cleanSampleDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod>(configuration, dataset, supportRadius, representativeSetRandomSeed, sampleVerticesSet, "sample");

    // Running experiments

    std::cout << "Cleaning up.." << std::endl;
    ShapeDescriptor::free(referenceDescriptors);

    std::cout << "Experiments completed." << std::endl;
}