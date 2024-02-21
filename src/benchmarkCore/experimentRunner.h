#pragma once

#include <exception>
#include <random>
#include <iostream>
#include "json.hpp"
#include "Dataset.h"
#include "ComputedConfig.h"
#include "supportRadiusEstimation/SupportRadiusEstimation.h"
#include "utils/progressBar.h"
#include "filters/subtractiveNoise/OcclusionFilter.h"
#include "filters/captureNoise/remeshingFilter.h"
#include "filters/captureNoise/normalNoiseFilter.h"
#include "filters/additiveNoise/AdditiveNoiseCache.h"
#include "results/ExperimentResult.h"
#include "benchmarkCore/common-procedures/areaEstimator.h"
#include "benchmarkCore/common-procedures/referenceIndexer.h"

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
    const uint32_t representativeSetSize = configuration.at("commonExperimentSettings").at("representativeSetSize");
    const uint32_t sampleSetSize = configuration.at("commonExperimentSettings").at("sampleSetSize");
    const uint32_t verticesPerSampleObject = configuration.at("commonExperimentSettings").at("verticesToTestPerSampleObject");

    // Compute reference descriptors, or load them from a cache file
    std::cout << "Computing reference descriptor set.." << std::endl;
    std::vector<ShapeBench::VertexInDataset> representativeSet = dataset.sampleVertices(engine(), representativeSetSize, 1);
    ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod>(configuration, dataset, supportRadius, engine(), representativeSet, "reference");

    // Computing sample descriptors, or load them from a cache file
    std::cout << "Computing sample descriptor set.." << std::endl;
    std::vector<ShapeBench::VertexInDataset> sampleVerticesSet = dataset.sampleVertices(engine(), sampleSetSize, configuration.at("commonExperimentSettings").at("verticesToTestPerSampleObject"));
    ShapeDescriptor::cpu::array<DescriptorType> cleanSampleDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod>(configuration, dataset, supportRadius, engine(), sampleVerticesSet, "sample");

    // Initialise filter caches
    std::cout << "Initialising filter caches.." << std::endl;
    ShapeBench::AdditiveNoiseCache additiveCache;
    ShapeBench::loadAdditiveNoiseCache(additiveCache, configuration);
    std::cout << "    Loaded Additive Noise filter cache" << std::endl;

    // Running experiments
    const uint32_t experimentCount = configuration.at("experimentsToRun").size();
    std::cout << "Running experiments.." << std::endl;
    uint64_t experimentBaseRandomSeed = engine();

    for(uint32_t experimentIndex = 0; experimentIndex < experimentCount; experimentIndex++) {
        ShapeBench::ExperimentResult experimentResult;
        experimentResult.methodName = DescriptorMethod::getName();
        experimentResult.usedConfiguration = configuration;
        experimentResult.usedComputedConfiguration = computedConfig;
        experimentResult.experimentRandomSeed = experimentBaseRandomSeed;

        const nlohmann::json& experimentConfig = configuration.at("experimentsToRun").at(experimentIndex);
        std::cout << "Experiment " << (experimentIndex + 1) << "/" << experimentCount << ": " << experimentConfig.at("name") << std::endl;

        ShapeBench::randomEngine experimentSeedEngine(experimentBaseRandomSeed);

        for(uint32_t sampleVertexIndex = 0; sampleVertexIndex < sampleSetSize; sampleVertexIndex += verticesPerSampleObject) {
            uint64_t experimentInstanceRandomSeed = experimentSeedEngine();
            ShapeBench::randomEngine experimentInstanceRandomEngine(experimentInstanceRandomSeed);

// Enable for debugging
//if(sampleVertexIndex < 444) {continue;}

            std::cout << "Vertex " << sampleVertexIndex << std::endl;

            ShapeBench::VertexInDataset firstSampleVertex = sampleVerticesSet.at(sampleVertexIndex);
            const ShapeBench::DatasetEntry &entry = dataset.at(firstSampleVertex.meshID);
            ShapeDescriptor::cpu::Mesh originalSampleMesh = ShapeBench::readDatasetMesh(configuration, entry);

            ShapeBench::FilteredMeshPair filteredMesh;
            filteredMesh.originalMesh = originalSampleMesh.clone();
            filteredMesh.filteredSampleMesh = originalSampleMesh.clone();

            filteredMesh.mappedReferenceVertices.resize(verticesPerSampleObject);
            filteredMesh.originalReferenceVertices.resize(verticesPerSampleObject);
            filteredMesh.referenceVertexIndices.resize(verticesPerSampleObject);
            for(uint32_t i = 0; i < verticesPerSampleObject; i++) {
                ShapeBench::VertexInDataset sampleVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                filteredMesh.originalReferenceVertices.at(i).vertex = filteredMesh.originalMesh.vertices[sampleVertex.vertexIndex];
                filteredMesh.originalReferenceVertices.at(i).normal = filteredMesh.originalMesh.normals[sampleVertex.vertexIndex];
                filteredMesh.mappedReferenceVertices.at(i) = filteredMesh.originalReferenceVertices.at(i);
                filteredMesh.referenceVertexIndices.at(i) = sampleVertex.vertexIndex;
            }

            try {
                for (uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
                    uint64_t filterRandomSeed = experimentInstanceRandomEngine();
                    const nlohmann::json &filterConfig = experimentConfig.at("filters").at(filterStepIndex);
                    const std::string &filterType = filterConfig.at("type");
                    if (filterType == "additive-noise") {
                        ShapeBench::applyAdditiveNoiseFilter(configuration, filteredMesh, dataset, filterRandomSeed,additiveCache);
                    } else if (filterType == "subtractive-noise") {
                        ShapeBench::applyOcclusionFilter(configuration, filteredMesh, filterRandomSeed);
                    } else if (filterType == "repeated-capture") {
                        ShapeBench::remesh(filteredMesh);
                    } else if (filterType == "normal-noise") {
                        ShapeBench::applyNormalNoiseFilter(filteredMesh);
                    }
                }

                // Collect data here
                ShapeBench::ExperimentResultsEntry resultsEntry;
                const uint64_t areaEstimationRandomSeed = experimentInstanceRandomEngine();
                const uint64_t pointCloudSamplingSeed = experimentInstanceRandomEngine();

                ShapeDescriptor::cpu::Mesh combinedMesh = filteredMesh.combinedFilteredMesh();

                //filteredMesh.writeFilteredMesh(DescriptorMethod::getName() + "-" + ShapeDescriptor::generateUniqueFilenameString() + "-" + std::to_string(experimentInstanceRandomSeed) + ".obj");

                for(uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntry.sourceVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                    resultsEntry.filteredDescriptorRank = 0;
                    resultsEntry.originalVertexLocation = filteredMesh.originalReferenceVertices.at(i);
                    resultsEntry.filteredVertexLocation = filteredMesh.mappedReferenceVertices.at(i);

                    ShapeBench::AreaEstimate areaEstimate = ShapeBench::estimateAreaInSupportVolume<DescriptorMethod>(filteredMesh,
                                                                                                                      resultsEntry.originalVertexLocation,
                                                                                                                      resultsEntry.filteredVertexLocation,
                                                                                                                      supportRadius,
                                                                                                                      configuration,
                                                                                                                      areaEstimationRandomSeed);



                    DescriptorType filteredPointDescriptor = ShapeBench::computeSingleDescriptor<DescriptorMethod, DescriptorType>(
                            combinedMesh, resultsEntry.filteredVertexLocation, configuration, supportRadius, pointCloudSamplingSeed);

                    uint32_t imageIndex = ShapeBench::computeImageIndex<DescriptorMethod, DescriptorType>(cleanSampleDescriptors[sampleVertexIndex + i], filteredPointDescriptor, referenceDescriptors);
                    std::cout << areaEstimate.addedAdrea << ", " << imageIndex << std::endl;

                    resultsEntry.fractionAddedNoise = areaEstimate.addedAdrea;
                    resultsEntry.fractionSurfacePartiality = areaEstimate.subtractiveArea;
                }

                ShapeDescriptor::free(combinedMesh);

                experimentResult.vertexResults.push_back(resultsEntry);

                // 1. DONE - Compute sample descriptor on clean mesh
                // 2. Compute modified descriptor based on filtered mesh and its corresponding sample point
                // 3. Compute distance from sample -> modified descriptor, and distance from sample -> all descriptors in reference set
                // 4. Compute rank of sample
                // Record metadata about "difficulty" of this sample
            } catch(const std::exception& e) {
                std::cout << "    Failed to process vertex " << sampleVertexIndex << ": " << e.what() << std::endl;
            }

            ShapeDescriptor::free(originalSampleMesh);
            filteredMesh.free();

            bool isLastVertexIndex = sampleVertexIndex + 1 == sampleSetSize;
            if (sampleVertexIndex % 100 == 0 || isLastVertexIndex) {
                std::cout << "\r    ";
                ShapeBench::drawProgressBar(sampleVertexIndex, sampleSetSize);
                std::cout << " " << (sampleVertexIndex + 1) << "/" << sampleSetSize << std::flush;
            }

            if (sampleVertexIndex % 500 == 0) {
                std::cout << std::endl << "    Writing caches.." << std::endl;
                ShapeBench::saveAdditiveNoiseCache(additiveCache, configuration);
            }
        }
    }

    std::cout << std::endl << "    Writing caches.." << std::endl;
    ShapeBench::saveAdditiveNoiseCache(additiveCache, configuration);

    std::cout << "Cleaning up.." << std::endl;
    ShapeDescriptor::free(referenceDescriptors);

    std::cout << "Experiments completed." << std::endl;
}
