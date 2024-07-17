#pragma once

#include <exception>
#include <random>
#include <iostream>
#include <mutex>
#include <omp.h>
#include "json.hpp"
#include <semaphore>
#include "dataset/Dataset.h"
#include "ComputedConfig.h"
#include "supportRadiusEstimation/SupportRadiusEstimation.h"
#include "utils/prettyprint.h"
#include "filters/subtractiveNoise/OcclusionFilter.h"
#include "filters/triangleShift/AlternateTriangulationFilter.h"
#include "filters/additiveNoise/AdditiveNoiseCache.h"
#include "filters/FilteredMeshPair.h"
#include "results/ExperimentResult.h"
#include "benchmarkCore/common-procedures/areaEstimator.h"
#include "benchmarkCore/common-procedures/referenceIndexer.h"
#include "results/ResultDumper.h"
#include "filters/normalVectorDeviation/normalNoiseFilter.h"
#include "filters/supportRadiusDeviation/supportRadiusNoise.h"
#include "filters/noisyCapture/NoisyCaptureFilter.h"
#include "filters/gaussianNoise/gaussianNoiseFilter.h"
#include "fmt/format.h"
#include "BenchmarkConfiguration.h"
#include "replication/RandomSubset.h"
#include "replication/ExperimentResultsValidator.h"

template <typename T>
class lockGuard
{
    T &m_;

public:
    lockGuard(T &m) : m_(m)
    {
        m_.acquire();
    }
    ~lockGuard()
    {
        m_.release();
    }
};

template<typename DescriptorMethod, typename DescriptorType>
std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> computeReferenceDescriptors(const std::vector<ShapeBench::VertexInDataset>& representativeSet,
                                                                                                 const nlohmann::json& config,
                                                                                                 const ShapeBench::Dataset& dataset,
                                                                                                 ShapeBench::LocalDatasetCache* fileCache,
                                                                                                 uint64_t randomSeed,float supportRadius,
                                                                                                 ShapeBench::RandomSubset* randomSubsetToReplicate = nullptr) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ShapeBench::randomEngine randomEngine(randomSeed);
    std::vector<uint64_t> randomSeeds(representativeSet.size());
    for(uint32_t i = 0; i < representativeSet.size(); i++) {
        randomSeeds.at(i) = randomEngine();
    }

    std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> representativeDescriptors(representativeSet.size());
    uint32_t completedCount = 0;

    #pragma omp parallel for schedule(dynamic) default(none) shared(representativeSet, randomSubsetToReplicate, supportRadius, fileCache, dataset, config, randomSeeds, representativeDescriptors, completedCount, std::cout)
    for(int i = 0; i < representativeSet.size(); i++) {
        uint32_t currentMeshID = representativeSet.at(i).meshID;
        if(i > 0 && representativeSet.at(i - 1).meshID == currentMeshID) {
            continue;
        }

        bool shouldReplicate = randomSubsetToReplicate != nullptr;
        bool sequenceContainsItemToReplicate = randomSubsetToReplicate != nullptr && randomSubsetToReplicate->contains(i);

        uint32_t sameMeshCount = 1;
        for(int j = i + 1; j < representativeSet.size(); j++) {
            uint32_t meshID = representativeSet.at(j).meshID;
            if(shouldReplicate && randomSubsetToReplicate->contains(j)) {
                sequenceContainsItemToReplicate = true;
            }
            if(currentMeshID != meshID) {
                break;
            }
            sameMeshCount++;
        }

        if(shouldReplicate && !sequenceContainsItemToReplicate) {
            continue;
        }

        std::vector<DescriptorType> outputDescriptors(sameMeshCount);
        std::vector<ShapeDescriptor::OrientedPoint> descriptorOrigins(sameMeshCount);
        std::vector<float> radii(sameMeshCount, supportRadius);

        const ShapeBench::DatasetEntry& entry = dataset.at(representativeSet.at(i).meshID);
        ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(config, fileCache, entry);

        for(int j = 0; j < sameMeshCount; j++) {
            uint32_t entryIndex = j + i;
            ShapeBench::VertexInDataset vertex = representativeSet.at(entryIndex);
            descriptorOrigins.at(j).vertex = mesh.vertices[vertex.vertexIndex];
            descriptorOrigins.at(j).normal = mesh.normals[vertex.vertexIndex];
        }

        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originArray(descriptorOrigins.size(), descriptorOrigins.data());

        ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(mesh, originArray, config, radii, randomSeeds.at(i), randomSeeds.at(i), outputDescriptors);
        for(int j = 0; j < sameMeshCount; j++) {
            uint32_t entryIndex = j + i;
            ShapeBench::VertexInDataset vertex = representativeSet.at(entryIndex);
            representativeDescriptors.at(entryIndex).descriptor = outputDescriptors.at(j);
            representativeDescriptors.at(entryIndex).meshID = vertex.meshID;
            representativeDescriptors.at(entryIndex).vertexIndex = vertex.vertexIndex;
            representativeDescriptors.at(entryIndex).vertex = descriptorOrigins.at(j);
        }

        ShapeDescriptor::free(mesh);

        #pragma omp atomic
        completedCount += sameMeshCount;

        if(completedCount % 100 == 0 || completedCount == representativeSet.size()) {
            std::cout << "\r    ";
            ShapeBench::drawProgressBar(completedCount, representativeSet.size());
            std::cout << " " << completedCount << "/" << representativeSet.size() << std::flush;
            malloc_trim(0);
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "    Complete." << std::endl;
    std::cout << "    Elapsed time: ";
    ShapeBench::printDuration(end - begin);
    std::cout << std::endl;

    return representativeDescriptors;
}



template<typename DescriptorType, typename DescriptorMethod, typename inputRepresentativeSetType>
std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> computeDescriptorsOrLoadCached(
        const nlohmann::json &configuration,
        const ShapeBench::Dataset &dataset,
        ShapeBench::LocalDatasetCache* fileCache,
        float supportRadius,
        uint64_t representativeSetRandomSeed,
        const std::vector<inputRepresentativeSetType> &representativeSet,
        std::string name) {
    const std::filesystem::path descriptorCacheFile = std::filesystem::path(std::string(configuration.at("cacheDirectory"))) / (name + "Descriptors-" + DescriptorMethod::getName() + ".dat");
    std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> referenceDescriptors;

    bool replicateSubset = configuration.at("replicationOverrides").at(name + "DescriptorSet").at("recomputeRandomSubset");
    uint32_t randomSubsetSize = configuration.at("replicationOverrides").at(name + "DescriptorSet").at("randomSubsetSize");
    bool replicateEntirely = configuration.at("replicationOverrides").at(name + "DescriptorSet").at("recomputeEntirely");

    if(!std::filesystem::exists(descriptorCacheFile)) {
        std::cout << "    No cached " + name + " descriptors were found." << std::endl;
        std::cout << "    Computing " + name + " descriptors.." << std::endl;
        referenceDescriptors = computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, configuration, dataset, fileCache, representativeSetRandomSeed, supportRadius);
        std::cout << "    Finished computing " + name + " descriptors. Writing archive file.." << std::endl;
        ShapeDescriptor::writeCompressedDescriptors<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>(descriptorCacheFile, {referenceDescriptors.size(), referenceDescriptors.data()});

        std::cout << "    Checking integrity of written data.." << std::endl;
        std::cout << "        Reading written file.." << std::endl;
        ShapeDescriptor::cpu::array<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> readDescriptors = ShapeDescriptor::readCompressedDescriptors<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>(descriptorCacheFile, 8);
        assert(referenceDescriptors.size() == readDescriptors.length);
        for(uint32_t i = 0; i < referenceDescriptors.size(); i++) {
            char* basePointerA = reinterpret_cast<char*>(&referenceDescriptors.at(i));
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
        ShapeDescriptor::cpu::array<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> readDescriptors = ShapeDescriptor::readCompressedDescriptors<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>(descriptorCacheFile, 8);
        referenceDescriptors.resize(readDescriptors.length);
        std::copy(readDescriptors.content, readDescriptors.content + readDescriptors.length, referenceDescriptors.begin());
        ShapeDescriptor::free(readDescriptors);
        std::cout << "    Successfully loaded " << referenceDescriptors.size() << " descriptors" << std::endl;
        std::string errorMessage = "Mismatch detected between the cached number of descriptors ("
                + std::to_string(referenceDescriptors.size()) + ") and the requested number of descriptors ("
                + std::to_string(representativeSet.size()) + ").";
        if(referenceDescriptors.size() < representativeSet.size()) {
            throw std::runtime_error("ERROR: " + errorMessage + " The number of cached descriptors is not sufficient. "
                                                                "Consider regenerating the cache by deleting the file: " + descriptorCacheFile.string());
        } else if(referenceDescriptors.size() > representativeSet.size()) {
            std::cout << "    WARNING: " + errorMessage + " Since the cache contains more descriptors than necessary, execution will continue." << std::endl;
        }

        if(replicateSubset || replicateEntirely) {
            if(replicateSubset && replicateEntirely) {
                std::cout << "    NOTE: replication of a random subset *and* the entire descriptor set was requested. The entire set will be replicated." << std::endl;
            }
            std::cout << "    Replication of " << name << " descriptor set has been enabled. Performing replication.." << std::endl;
            uint32_t numberOfDescriptorsToReplicate = replicateEntirely ? readDescriptors.length : std::min<uint32_t>(readDescriptors.length, randomSubsetSize);
            uint64_t replicationRandomSeed = configuration.at("replicationOverrides").at("replicationRandomSeed");
            ShapeBench::RandomSubset subset(0, readDescriptors.length, numberOfDescriptorsToReplicate, replicationRandomSeed);
            std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> replicatedDescriptors;
            std::cout << "    Computing " << numberOfDescriptorsToReplicate << " descriptors.." << std::endl;
            replicatedDescriptors = computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, configuration, dataset, fileCache, representativeSetRandomSeed, supportRadius, &subset);
            std::cout << "    Comparing computed descriptors against those in the cache.." << std::endl;
            for(uint32_t descriptorIndex = 0; descriptorIndex < representativeSet.size(); descriptorIndex++) {
                if(!subset.contains(descriptorIndex)) {
                    continue;
                }
                uint8_t* descriptorA = reinterpret_cast<uint8_t*>(&replicatedDescriptors.at(descriptorIndex).descriptor);
                uint8_t* descriptorB = reinterpret_cast<uint8_t*>(&referenceDescriptors.at(descriptorIndex).descriptor);
                for(uint32_t byteIndex = 0; byteIndex < sizeof(DescriptorType); byteIndex++) {
                    if(descriptorA[byteIndex] != descriptorB[byteIndex]) {
                        throw std::logic_error("FATAL: Descriptors at index " + std::to_string(descriptorIndex) + " failed to replicate at byte " + std::to_string(byteIndex) + "!");
                    }
                }
            }
            std::cout << "    All descriptors in the replicated set were identical to those cached on disk." << std::endl;
            std::cout << "    Replication successful." << std::endl;
        }
    }
    return referenceDescriptors;
}

template<typename DescriptorMethod, typename DescriptorType>
void testMethod(const ShapeBench::BenchmarkConfiguration& setup, ShapeBench::LocalDatasetCache* fileCache) {
    std::cout << std::endl << "========== TESTING METHOD " << DescriptorMethod::getName() << " ==========" << std::endl;
    std::cout << "Initialising.." << std::endl;
    uint64_t randomSeed = setup.configuration.at("randomSeed");
    ShapeBench::randomEngine engine(randomSeed);
    const nlohmann::json& configuration = setup.configuration;
    const ShapeBench::Dataset& dataset = setup.dataset;

    std::filesystem::path computedConfigFilePath = setup.configurationFilePath.parent_path() / std::string(configuration.at("computedConfigFile"));
    DescriptorMethod::init(configuration);
    std::cout << "    Main config file: " << setup.configurationFilePath.string() << std::endl;
    std::cout << "    Computed values config file: " << computedConfigFilePath.string() << std::endl;
    ShapeBench::ComputedConfig computedConfig(computedConfigFilePath);
    const std::string methodName = DescriptorMethod::getName();

    std::filesystem::path resultsDirectory = configuration.at("resultsDirectory");
    if (!std::filesystem::exists(resultsDirectory)) {
        std::cout << "    Creating results directory.." << std::endl;
        std::filesystem::create_directories(resultsDirectory);
    }

    bool enablePRCComparisonMode = configuration.contains("enableComparisonToPRC") && configuration.at("enableComparisonToPRC");
    if(enablePRCComparisonMode) {
        std::cout << "Comparison against the Precision-Recall Curve evaluation strategy is enabled." << std::endl;
    }

    // Getting a support radius
    std::cout << "Determining support radius.." << std::endl;
    float supportRadius = 0;

    bool shouldReplicateSupportRadiusEntirely = configuration.at("replicationOverrides").at("supportRadius").at("recomputeEntirely");
    bool shouldReplicateSupportRadiusPartially = configuration.at("replicationOverrides").at("supportRadius").at("recomputeSingleRadius");
    bool shouldReplicateSupportRadius = shouldReplicateSupportRadiusEntirely || shouldReplicateSupportRadiusPartially;

    if (!computedConfig.containsKey(methodName, "supportRadius")) {
        std::cout << "    No support radius has been computed yet for this method." << std::endl;
        std::cout << "    Performing support radius estimation.." << std::endl;
        supportRadius = ShapeBench::estimateSupportRadius<DescriptorMethod, DescriptorType>(configuration, dataset, fileCache, engine());
        std::cout << "    Chosen support radius: " << supportRadius << std::endl;
        computedConfig.setFloatAndSave(methodName, "supportRadius", supportRadius);
    } else {
        supportRadius = computedConfig.getFloat(methodName, "supportRadius");
        std::cout << "    Cached support radius was found for this method: " << supportRadius << std::endl;
        uint64_t supportRadiusRandomSeed = engine(); // Used for RNG consistency

        if(shouldReplicateSupportRadius) {
            std::cout << "    Replication of support radius was requested. Performing replication.." << std::endl;
            float replicatedSupportRadius = ShapeBench::estimateSupportRadius<DescriptorMethod, DescriptorType>(configuration, dataset, fileCache, supportRadiusRandomSeed);
            if(shouldReplicateSupportRadiusEntirely) {
                if(replicatedSupportRadius != supportRadius) {
                    throw std::logic_error("FATAL: replicated support radius does not match the one that was computed previously! Original: " + std::to_string(supportRadius) + ", replicated: " + std::to_string(replicatedSupportRadius));
                }
                std::cout << "    Support radius has been successfully replicated." << std::endl;
            }
            std::cout << "    The computed distance statistics must be validated upon completion of the benchmark executable." << std::endl;
            std::cout << "    If you ran the benchmark through the replication script, this will be done automatically." << std::endl;
        }
    }

    // Computing sample descriptors and their distance to the representative set
    const uint32_t representativeSetSize = configuration.at("commonExperimentSettings").at("representativeSetSize");
    const uint32_t sampleSetSize = configuration.at("commonExperimentSettings").at("sampleSetSize");
    const uint32_t verticesPerSampleObject = configuration.at("commonExperimentSettings").at("verticesToTestPerSampleObject");
    const uint32_t verticesPerReferenceObject = configuration.at("commonExperimentSettings").at("verticesToTestPerReferenceObject");

    // Compute reference descriptors, or load them from a cache file
    std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> referenceDescriptors;
    std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>> cleanSampleDescriptors;
    std::vector<ShapeBench::VertexInDataset> representativeSet;
    std::vector<ShapeBench::VertexInDataset> sampleVerticesSet;

    std::cout << "Computing reference descriptor set.. (" << verticesPerReferenceObject << " vertices per object)" << std::endl;
    representativeSet = dataset.sampleVertices(engine(), representativeSetSize, verticesPerReferenceObject);
    referenceDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::VertexInDataset>(configuration, dataset, fileCache, supportRadius, engine(), representativeSet, "reference");

    std::cout << "Computing sample descriptor set.. (" << verticesPerSampleObject << " vertices per object)" << std::endl;
    sampleVerticesSet = dataset.sampleVertices(engine(), sampleSetSize, verticesPerSampleObject);
    cleanSampleDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::VertexInDataset>(configuration, dataset, fileCache, supportRadius, engine(), sampleVerticesSet, "sample");



    // Initialise filters
    std::unordered_map<std::string, std::unique_ptr<ShapeBench::Filter>> filterInstanceMap;
    filterInstanceMap.insert(std::make_pair("repeated-capture", new ShapeBench::AlternateTriangulationFilter()));
    filterInstanceMap.insert(std::make_pair("support-radius-deviation", new ShapeBench::SupportRadiusNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("normal-noise", new ShapeBench::NormalNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("additive-noise", new ShapeBench::AdditiveNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("subtractive-noise", new ShapeBench::OcclusionFilter()));
    filterInstanceMap.insert(std::make_pair("depth-camera-capture", new ShapeBench::NoisyCaptureFilter()));
    filterInstanceMap.insert(std::make_pair("gaussian-noise", new ShapeBench::GaussianNoiseFilter()));

    // Running experiments
    const uint32_t experimentCount = configuration.at("experimentsToRun").size();
    std::cout << "Running experiments.." << std::endl;
    uint64_t experimentBaseRandomSeed = engine();

    const uint32_t intermediateSaveFrequency = configuration.at("commonExperimentSettings").at("intermediateSaveFrequency");
    std::mutex resultWriteLock;

    bool enableIllustrationGenerationMode = configuration.at("illustrationDataGenerationOverride").at("enableIllustrationDataGeneration");
    uint32_t illustrativeObjectLimit = 0;
    uint32_t illustrativeObjectStride = 1;
    std::filesystem::path illustrativeObjectOutputDirectory;
    ShapeDescriptor::cpu::array<DescriptorType> illustrationImages;
    if(enableIllustrationGenerationMode) {
        std::cout << "    Illustration object generation mode is active, result generation is disabled." << std::endl;
        illustrativeObjectLimit = configuration.at("illustrationDataGenerationOverride").at("objectLimit");
        illustrativeObjectStride = configuration.at("illustrationDataGenerationOverride").at("objectStride");
        illustrativeObjectOutputDirectory = resultsDirectory / std::string(configuration.at("illustrationDataGenerationOverride").at("outputDirectory"));
        if(!std::filesystem::exists(illustrativeObjectOutputDirectory)) {
            std::filesystem::create_directories(illustrativeObjectOutputDirectory);
        }
        if(DescriptorMethod::getName() == "QUICCI") {
            std::filesystem::path referenceImageFile = illustrativeObjectOutputDirectory / "reference_descriptors.png";
            std::filesystem::path sampleImageFile = illustrativeObjectOutputDirectory / "sample_descriptors.png";
            uint32_t startIndex = configuration.at("illustrationDataGenerationOverride").at("referenceDescriptorStartIndex");
            uint32_t count = configuration.at("illustrationDataGenerationOverride").at("referenceDescriptorCount");
            //ShapeDescriptor::writeDescriptorImages({count, reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(referenceDescriptors.content) + startIndex}, referenceImageFile, false);
            //ShapeDescriptor::writeDescriptorImages({count, reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(cleanSampleDescriptors.content) + startIndex}, sampleImageFile, false);
            illustrationImages = ShapeDescriptor::cpu::array<DescriptorType>(count);
        }
    }







    // --- Running experiments ---
    for(uint32_t experimentIndex = 0; experimentIndex < experimentCount; experimentIndex++) {
        ShapeBench::ExperimentResult experimentResult;
        experimentResult.methodName = DescriptorMethod::getName();
        experimentResult.usedConfiguration = configuration;
        experimentResult.usedComputedConfiguration = computedConfig;
        experimentResult.experimentRandomSeed = experimentBaseRandomSeed;
        experimentResult.experimentIndex = experimentIndex;
        experimentResult.methodMetadata = DescriptorMethod::getMetadata();
        experimentResult.vertexResults.resize(sampleSetSize);

        const nlohmann::json &experimentConfig = configuration.at("experimentsToRun").at(experimentIndex);

        if (!experimentConfig.at("enabled")) {
            std::cout << "Experiment " << (experimentIndex + 1) << " is disabled. Skipping." << std::endl;
            continue;
        }

        std::string experimentName = experimentConfig.at("name");
        std::cout << "Experiment " << (experimentIndex + 1) << "/" << experimentCount << ": " << experimentName << std::endl;

        ShapeBench::RandomSubset replicationSubset;
        if(setup.replicationSettings.enabled) {
            std::cout << "    Replication mode enabled." << std::endl;
            bool replicateEntirely = configuration.at("replicationOverrides").at("experiment").at("recomputeEntirely");
            bool replicateSubset = configuration.at("replicationOverrides").at("experiment").at("recomputeRandomSubset");
            if(!(replicateEntirely || replicateSubset)) {
                std::cout << "    WARNING: Configuration file did not specify the extent to which the given results file should be replicated. It will be replicated entirely." << std::endl;
                replicateEntirely = true;
            } else if(replicateEntirely && replicateSubset) {
                replicateSubset = false;
            }

            uint32_t numberOfResultsToReplicate = sampleSetSize;
            if(replicateSubset) {
                numberOfResultsToReplicate = configuration.at("replicationOverrides").at("experiment").at("randomSubsetSize");
            }
            uint64_t replicationRandomSeed = configuration.at("replicationOverrides").at("replicationRandomSeed");
            replicationSubset = ShapeBench::RandomSubset(0, sampleSetSize, numberOfResultsToReplicate, replicationRandomSeed);
            std::cout << "    Replicating " << numberOfResultsToReplicate << " results.." << std::endl;
        }

        ShapeBench::randomEngine experimentSeedEngine(experimentBaseRandomSeed);
        uint32_t testedObjectCount = sampleSetSize / verticesPerSampleObject;
        std::vector<uint64_t> experimentRandomSeeds(testedObjectCount);
        for(uint32_t i = 0; i < testedObjectCount; i++) {
            experimentRandomSeeds.at(i) = experimentSeedEngine();
        }
        for(uint32_t i = 0; i < sampleSetSize; i++) {
            experimentResult.vertexResults.at(i).included = false;
        }

        for(uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
            std::string filterName = experimentConfig.at("filters").at(filterStepIndex).at("type");
            bool forceCacheInvalidation = setup.replicationSettings.enabled;
            filterInstanceMap.at(filterName)->init(configuration, forceCacheInvalidation);
        }

        std::vector<uint32_t> threadActivity;

        uint32_t threadsToLaunch = omp_get_max_threads();
        if(experimentConfig.contains("threadLimit")) {
            threadsToLaunch = experimentConfig.at("threadLimit");
        }

        #pragma omp parallel for schedule(dynamic) num_threads(threadsToLaunch) default(none) shared(setup, replicationSubset, fileCache, resultsDirectory, intermediateSaveFrequency, experimentResult, enablePRCComparisonMode, cleanSampleDescriptors, referenceDescriptors, illustrationImages, supportRadius, configuration, sampleSetSize, verticesPerSampleObject, illustrativeObjectStride, experimentRandomSeeds, sampleVerticesSet, dataset, enableIllustrationGenerationMode, resultWriteLock, threadActivity, std::cout, illustrativeObjectLimit, experimentIndex, experimentName, illustrativeObjectOutputDirectory, experimentConfig, filterInstanceMap)
        for (uint32_t sampleVertexIndex = 0; sampleVertexIndex < sampleSetSize; sampleVertexIndex += verticesPerSampleObject * illustrativeObjectStride) {
            if(setup.replicationSettings.enabled) {
                bool objectContainsResultToReplicate = false;
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    objectContainsResultToReplicate = objectContainsResultToReplicate || replicationSubset.contains(sampleVertexIndex + i);
                }
                if(!objectContainsResultToReplicate) {
                    continue;
                }
            }

            ShapeBench::randomEngine experimentInstanceRandomEngine(experimentRandomSeeds.at(sampleVertexIndex / verticesPerSampleObject));

            ShapeBench::VertexInDataset firstSampleVertex = sampleVerticesSet.at(sampleVertexIndex);
            uint32_t meshID = firstSampleVertex.meshID;

            const ShapeBench::DatasetEntry &entry = dataset.at(meshID);

            if(!enableIllustrationGenerationMode){
                std::unique_lock<std::mutex> writeLock{resultWriteLock};

                if (threadActivity.empty()) {
                    threadActivity.resize(omp_get_num_threads());
                }
                threadActivity.at(omp_get_thread_num()) = sampleVertexIndex;
            }

            if(enableIllustrationGenerationMode && sampleVertexIndex >= illustrativeObjectLimit) {
                continue;
            }

// Enable for debugging
//if(sampleVertexIndex < 36200) {continue;}
//if(experimentRandomSeeds.at(sampleVertexIndex / verticesPerSampleObject) != 6847108265827174418) {
//    continue;
//}


            ShapeDescriptor::cpu::Mesh originalSampleMesh = ShapeBench::readDatasetMesh(configuration, fileCache, entry);

            ShapeBench::FilteredMeshPair filteredMesh;
            filteredMesh.originalMesh = originalSampleMesh.clone();
            filteredMesh.filteredSampleMesh = originalSampleMesh.clone();

            filteredMesh.mappedReferenceVertices.resize(verticesPerSampleObject);
            filteredMesh.originalReferenceVertices.resize(verticesPerSampleObject);
            filteredMesh.mappedReferenceVertexIndices.resize(verticesPerSampleObject);
            filteredMesh.mappedVertexIncluded.resize(verticesPerSampleObject);
            for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                ShapeBench::VertexInDataset sampleVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                filteredMesh.originalReferenceVertices.at(i).vertex = filteredMesh.originalMesh.vertices[sampleVertex.vertexIndex];
                filteredMesh.originalReferenceVertices.at(i).normal = filteredMesh.originalMesh.normals[sampleVertex.vertexIndex];
                filteredMesh.mappedReferenceVertexIndices.at(i) = sampleVertex.vertexIndex;
                filteredMesh.mappedReferenceVertices.at(i) = filteredMesh.originalReferenceVertices.at(i);
                filteredMesh.mappedVertexIncluded.at(i) = true;
            }

            if(enableIllustrationGenerationMode && experimentIndex == 0) {
                std::string filename = DescriptorMethod::getName() + "-" + std::to_string(experimentRandomSeeds.at(sampleVertexIndex / verticesPerSampleObject)) + "-" + ShapeDescriptor::generateUniqueFilenameString()  + "_" + experimentName + "_unaltered.obj";
                std::filesystem::path outputFile = illustrativeObjectOutputDirectory / filename;
                ShapeBench::writeFilteredMesh<DescriptorMethod>(filteredMesh, outputFile);
            }



            std::vector<ShapeBench::ExperimentResultsEntry> resultsEntries(verticesPerSampleObject);

            try {
                // Run filters
                nlohmann::json filterMetadata;

                for(uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
                    uint64_t filterRandomSeed = experimentInstanceRandomEngine();
                    const nlohmann::json &filterConfig = experimentConfig.at("filters").at(filterStepIndex);
                    const std::string &filterType = filterConfig.at("type");

                    ShapeBench::FilterOutput output = filterInstanceMap.at(filterType)->apply(configuration, filteredMesh, dataset, fileCache, filterRandomSeed);
                    filterMetadata = output.metadata;

                    if(!filterMetadata.empty()) {
                        for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                            resultsEntries.at(i).filterOutput.merge_patch(filterMetadata.at(i));
                        }
                    }
                }

                // Filter execution complete, computing DDI and relevant statistics
                const uint64_t areaEstimationRandomSeed = experimentInstanceRandomEngine();
                const uint64_t pointCloudSamplingSeed = experimentInstanceRandomEngine();
                const uint64_t sampleDescriptorGenerationSeed = experimentInstanceRandomEngine();

                ShapeDescriptor::cpu::Mesh combinedMesh = filteredMesh.combinedFilteredMesh();

                if(enableIllustrationGenerationMode) {
                    std::string filename = DescriptorMethod::getName() + "-" + std::to_string(experimentRandomSeeds.at(sampleVertexIndex / verticesPerSampleObject))+ "-" + ShapeDescriptor::generateUniqueFilenameString() + "_" + experimentName + ".obj";
                    std::filesystem::path outputFile = illustrativeObjectOutputDirectory / filename;
                    ShapeBench::writeFilteredMesh<DescriptorMethod>(filteredMesh, outputFile, filteredMesh.mappedReferenceVertices.at(0), supportRadius, false);
                }

                std::vector<DescriptorType> filteredDescriptors(verticesPerSampleObject);
                std::vector<float> radii(verticesPerSampleObject, supportRadius);
                bool doDryRun = configuration.contains("debug_dryrun") && configuration.at("debug_dryrun");
                if(!doDryRun) {
                    ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(
                            combinedMesh,
                            {filteredMesh.mappedReferenceVertices.size(), filteredMesh.mappedReferenceVertices.data()},
                            configuration,
                            radii,
                            pointCloudSamplingSeed,
                            sampleDescriptorGenerationSeed,
                            filteredDescriptors);
                }


                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntries.at(i).included = filteredMesh.mappedVertexIncluded.at(i);
                    if(!resultsEntries.at(i).included) {
                        if(DescriptorMethod::getName() == "QUICCI" && enableIllustrationGenerationMode) {
                            illustrationImages.content[(sampleVertexIndex/illustrativeObjectStride) + i] = filteredDescriptors.at(i);
                        }
                        continue;
                    }

                    resultsEntries.at(i).sourceVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                    resultsEntries.at(i).filteredDescriptorRank = 0;
                    resultsEntries.at(i).originalVertexLocation = filteredMesh.originalReferenceVertices.at(i);
                    resultsEntries.at(i).filteredVertexLocation = filteredMesh.mappedReferenceVertices.at(i);

                    if(enableIllustrationGenerationMode) {
                        if(DescriptorMethod::getName() == "QUICCI") {
                            illustrationImages.content[(sampleVertexIndex/illustrativeObjectStride) + i] = filteredDescriptors.at(i);
                        }
                        continue;
                    }

                    uint32_t imageIndex = !doDryRun ? ShapeBench::computeImageIndex<DescriptorMethod, DescriptorType>(cleanSampleDescriptors.at(sampleVertexIndex + i), filteredDescriptors.at(i), referenceDescriptors) : 0;

                    if(enablePRCComparisonMode) {
                        ShapeBench::DescriptorOfVertexInDataset<DescriptorType> filteredDescriptor;
                        filteredDescriptor.vertex = filteredMesh.originalReferenceVertices.at(i);
                        filteredDescriptor.descriptor = filteredDescriptors.at(i);
                        filteredDescriptor.meshID = sampleVerticesSet.at(sampleVertexIndex + i).meshID;
                        filteredDescriptor.vertexIndex = sampleVerticesSet.at(sampleVertexIndex + i).vertexIndex;

                        resultsEntries.at(i).prcMetadata = ShapeBench::computePRCInfo<DescriptorMethod, DescriptorType>(filteredDescriptor, cleanSampleDescriptors.at(sampleVertexIndex + i), referenceDescriptors, cleanSampleDescriptors, false);
                    }

                    ShapeBench::AreaEstimate areaEstimate = !doDryRun ? ShapeBench::estimateAreaInSupportVolume<DescriptorMethod>(filteredMesh, resultsEntries.at(i).originalVertexLocation, resultsEntries.at(i).filteredVertexLocation, supportRadius, configuration, areaEstimationRandomSeed) : ShapeBench::AreaEstimate();

                    resultsEntries.at(i).filteredDescriptorRank = imageIndex;
                    resultsEntries.at(i).fractionAddedNoise = areaEstimate.addedAdrea;
                    resultsEntries.at(i).fractionSurfacePartiality = areaEstimate.subtractiveArea;
                }



                ShapeDescriptor::free(combinedMesh);

                if(!enableIllustrationGenerationMode) {
                    std::unique_lock<std::mutex> writeLock{resultWriteLock};
                    for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                        experimentResult.vertexResults.at(sampleVertexIndex + i) = resultsEntries.at(i);
                    }
                    if(configuration.contains("verboseOutput") && configuration.at("verboseOutput")) {
                        std::cout << "Added area, Remaining area, Rank" << (enablePRCComparisonMode ? ", NN distance, SNN distance, Tao, Vertex distance, In range, MeshID 1, MeshID 2, Same MeshID" : "") << std::endl;
                        for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                            if(resultsEntries.at(i).included) {
                                std::cout << fmt::format("Result: {:<10}, {:<10}, {:<10}", resultsEntries.at(i).fractionAddedNoise, resultsEntries.at(i).fractionSurfacePartiality, resultsEntries.at(i).filteredDescriptorRank);
                                if(enablePRCComparisonMode) {
                                    float nearestNeighbourDistance = resultsEntries.at(i).prcMetadata.distanceToNearestNeighbour;
                                    float secondNearestNeighbourDistance = resultsEntries.at(i).prcMetadata.distanceToSecondNearestNeighbour;
                                    float tao = secondNearestNeighbourDistance == 0 ? 0 : nearestNeighbourDistance / secondNearestNeighbourDistance;

                                    float distanceToNearestNeighbourVertex = length(resultsEntries.at(i).prcMetadata.nearestNeighbourVertexModel - resultsEntries.at(i).prcMetadata.nearestNeighbourVertexScene);
                                    bool isInRange = distanceToNearestNeighbourVertex <= (supportRadius/2.0);

                                    uint32_t modelMeshID = resultsEntries.at(i).prcMetadata.modelPointMeshID;
                                    uint32_t sceneMeshID = resultsEntries.at(i).prcMetadata.scenePointMeshID;
                                    bool isSameObject = modelMeshID == sceneMeshID;
                                    std::cout << fmt::format(", {:<10}, {:<10}, {:<6}, {:<10}, {:<10}, {:<6}",
                                                             tao,
                                                             distanceToNearestNeighbourVertex, (isInRange ? "true" : "false"),
                                                             modelMeshID, sceneMeshID, (isSameObject ? "true" : "false"));
                                }
                                std::cout << std::endl;
                            }
                        }
                    }
                };

                // 1. DONE - Compute sample descriptor on clean mesh
                // 2. DONE - Compute modified descriptor based on filtered mesh and its corresponding sample point
                // 3. Compute distance from sample -> modified descriptor, and distance from sample -> all descriptors in reference set
                // 4. Compute rank of sample
                // Record metadata about "difficulty" of this sample
            } catch (const std::exception &e) {
                std::cout << "    Failed to process vertex " << sampleVertexIndex << ": " << e.what() << std::endl;
            }

            ShapeDescriptor::free(originalSampleMesh);
            filteredMesh.free();

            {
                std::unique_lock<std::mutex> writeLock{resultWriteLock};
                bool isLastVertexIndex = sampleVertexIndex + 1 == sampleSetSize;
                if(sampleVertexIndex % 100 == 0 || isLastVertexIndex) {
                    std::cout << "\r    ";
                    ShapeBench::drawProgressBar(sampleVertexIndex, sampleSetSize);
                    std::cout << " " << (sampleVertexIndex) << "/" << sampleSetSize << std::endl;
                    malloc_trim(0);
                }

                if(!enableIllustrationGenerationMode){
                    std::cout << "Processing method " << DescriptorMethod::getName() << ", experiment " << experimentName << ": completed " << sampleVertexIndex << "/" << sampleSetSize << " - " << entry.vertexCount << " vertices - Threads: (";
                    for(uint32_t i = 0; i < threadActivity.size(); i++) {
                        std::cout << threadActivity.at(i) << (i + 1 < threadActivity.size() ? ", " : "");
                    }
                    std::cout << ")" << std::endl;

                    if(sampleVertexIndex % intermediateSaveFrequency == 0) {
                        std::cout << std::endl << "    Writing caches.." << std::endl;
                        writeExperimentResults(experimentResult, resultsDirectory, false, enablePRCComparisonMode, setup.replicationSettings.enabled);
                        if(!setup.replicationSettings.enabled) {
                            for(uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
                                std::string filterName = experimentConfig.at("filters").at(filterStepIndex).at("type");
                                filterInstanceMap.at(filterName)->saveCaches(configuration);
                            }
                        }
                    }

                    // Some slight race condition here, but does not matter since it's only used for printing
                    threadActivity.at(omp_get_thread_num()) = 0;
                }
            }
        }

        if(!enableIllustrationGenerationMode) {
            std::cout << "Writing experiment results file.." << std::endl;
            writeExperimentResults(experimentResult, resultsDirectory, true, enablePRCComparisonMode, setup.replicationSettings.enabled);
            std::cout << "Experiment complete." << std::endl;

            if(setup.replicationSettings.enabled) {
                ShapeBench::checkReplicatedExperimentResults<DescriptorMethod>(configuration, experimentResult, setup.replicationSettings.experimentResults);
            }
        } else {
            std::string fileName = "descriptors_" + DescriptorMethod::getName() + "_" + ShapeDescriptor::generateUniqueFilenameString() + "_" + std::string(experimentConfig.at("name")) + ".png";
            std::filesystem::path outputFilePath = illustrativeObjectOutputDirectory / fileName;
            if(DescriptorMethod::getName() == "QUICCI") {
                ShapeDescriptor::writeDescriptorImages({illustrationImages.length, reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(illustrationImages.content)}, outputFilePath, false);
            }
        }

        if(!setup.replicationSettings.enabled) {
            std::cout << "Writing caches.." << std::endl;
            for (uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
                std::string filterName = experimentConfig.at("filters").at(filterStepIndex).at("type");
                filterInstanceMap.at(filterName)->saveCaches(configuration);
            }
        }
        for (uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
            std::string filterName = experimentConfig.at("filters").at(filterStepIndex).at("type");
            filterInstanceMap.at(filterName)->destroy();
        }
    }


    std::cout << "Cleaning up.." << std::endl;
    if(enableIllustrationGenerationMode && DescriptorMethod::getName() == "QUICCI") {
        ShapeDescriptor::free(illustrationImages);
    }

    std::cout << "Experiments completed." << std::endl;
}

