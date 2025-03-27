#pragma once

#include <exception>
#include <random>
#include <iostream>
#include <mutex>
#include <omp.h>
#include "nlohmann/json.hpp"
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
#include "cachedDescriptorLoader.h"
#include "filters/pointCloudResolution/pointCloudResolutionFilter.h"

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

ShapeBench::FilteredMeshPair initialiseFilteredMeshPair(const uint32_t verticesPerSampleObject,
                                                        const std::vector<ShapeBench::VertexInDataset> &sampleVerticesSet,
                                                        const uint32_t sampleVertexIndex,
                                                        const ShapeDescriptor::cpu::Mesh &meshToBeFiltered) {
    ShapeBench::FilteredMeshPair filteredMesh;
    filteredMesh.originalMesh = meshToBeFiltered.clone();
    filteredMesh.filteredSampleMesh = meshToBeFiltered.clone();

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
    return filteredMesh;
}

inline nlohmann::json applyFilterSequence(
                    ShapeBench::LocalDatasetCache *fileCache,
                    const std::string filterSequenceJSONEntryLabel,
                    const nlohmann::json &configuration,
                    const ShapeBench::Dataset &dataset,
                    const std::unordered_map<std::string, std::unique_ptr<ShapeBench::Filter>> &filterInstanceMap,
                    const uint32_t verticesPerSampleObject,
                    const nlohmann::json &experimentConfig,
                    ShapeBench::randomEngine &experimentInstanceRandomEngine,
                    std::vector<ShapeBench::ExperimentResultsEntry> &resultsEntries,
                    ShapeBench::FilteredMeshPair &filteredSceneObject) {
    nlohmann::json filterMetadata {};
    if(!experimentConfig.contains(filterSequenceJSONEntryLabel)) {
        // No filters specified. Leave object unmodified
        return filterMetadata;
    }
    for(uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at(filterSequenceJSONEntryLabel).size(); filterStepIndex++) {
        uint64_t filterRandomSeed = experimentInstanceRandomEngine();
        const nlohmann::json &filterConfig = experimentConfig.at(filterSequenceJSONEntryLabel).at(filterStepIndex);
        const std::string &filterType = filterConfig.at("type");

        ShapeBench::FilterOutput output = filterInstanceMap.at(filterType)->apply(configuration, filteredSceneObject, dataset, fileCache, filterRandomSeed);

        if(!output.metadata.empty()) {
            for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                filterMetadata.merge_patch(output.metadata.at(i));
            }
        }
    }
    return filterMetadata;
}

template<typename DescriptorMethod, typename DescriptorType>
void testMethod(const ShapeBench::BenchmarkConfiguration& setup, ShapeBench::LocalDatasetCache* fileCache) {

    // Initialising benchmark
    std::cout << std::endl << "========== TESTING METHOD " << DescriptorMethod::getName() << " ==========" << std::endl;
    std::cout << "Initialising.." << std::endl;
    uint64_t randomSeed = setup.configuration.at("randomSeed");
    ShapeBench::randomEngine engine(randomSeed);
    const nlohmann::json& configuration = setup.configuration;
    const ShapeBench::Dataset& dataset = setup.dataset;
    const std::string methodName = DescriptorMethod::getName();

    std::filesystem::path computedConfigFilePath = setup.configurationFilePath.parent_path() / std::string(configuration.at("computedConfigFile"));
    DescriptorMethod::init(configuration);
    std::cout << "    Main config file: " << setup.configurationFilePath.string() << std::endl;
    std::cout << "    Computed values config file: " << computedConfigFilePath.string() << std::endl;
    ShapeBench::ComputedConfig computedConfig(computedConfigFilePath);

    std::filesystem::path resultsDirectory = configuration.at("resultsDirectory");
    if (!std::filesystem::exists(resultsDirectory)) {
        std::cout << "    Creating results directory.." << std::endl;
        std::filesystem::create_directories(resultsDirectory);
    }

    bool enablePRCComparisonMode = configuration.contains("enableComparisonToPRC") && configuration.at("enableComparisonToPRC");
    if(enablePRCComparisonMode) {
        std::cout << "Comparison against the Precision-Recall Curve evaluation strategy is enabled." << std::endl;
    }

    // Initialise filters
    std::unordered_map<std::string, std::unique_ptr<ShapeBench::Filter>> filterInstanceMap;
    filterInstanceMap.insert(std::make_pair("repeated-capture", new ShapeBench::AlternateTriangulationFilter()));
    filterInstanceMap.insert(std::make_pair("support-radius-deviation", new ShapeBench::SupportRadiusNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("normal-noise", new ShapeBench::NormalNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("additive-noise", new ShapeBench::AdditiveNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("subtractive-noise", new ShapeBench::OcclusionFilter()));
    filterInstanceMap.insert(std::make_pair("depth-camera-capture", new ShapeBench::NoisyCaptureFilter()));
    filterInstanceMap.insert(std::make_pair("gaussian-noise", new ShapeBench::GaussianNoiseFilter()));
    filterInstanceMap.insert(std::make_pair("point-cloud-resolution", new ShapeBench::PointCloudResolutionFilter()));

    // Getting a support radius
    std::cout << "Determining support radius.." << std::endl;
    float supportRadius = 0;

    bool shouldReplicateSupportRadiusEntirely = configuration.at("replicationOverrides").at("supportRadius").at("recomputeEntirely");
    bool shouldReplicateSupportRadiusPartially = configuration.at("replicationOverrides").at("supportRadius").at("recomputeSingleRadius");
    bool shouldReplicateSupportRadius = shouldReplicateSupportRadiusEntirely || shouldReplicateSupportRadiusPartially;
    bool supportRadiusHasBeenCalculated = computedConfig.containsKey(methodName, "supportRadius");

    if (!supportRadiusHasBeenCalculated) {
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
    std::vector<ShapeBench::VertexInDataset> representativeSet;
    std::vector<ShapeBench::VertexInDataset> sampleVerticesSet;

    std::cout << "Computing reference descriptor set.. (" << verticesPerReferenceObject << " vertices per object)" << std::endl;
    representativeSet = dataset.sampleVertices(engine(), representativeSetSize, verticesPerReferenceObject);
    referenceDescriptors = ShapeBench::computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::VertexInDataset>(configuration, dataset, fileCache, supportRadius, engine(), representativeSet, "reference");

    std::cout << "Sampling vertices.. (" << verticesPerSampleObject << " vertices per object)" << std::endl;
    sampleVerticesSet = dataset.sampleVertices(engine(), sampleSetSize, verticesPerSampleObject);

    // This replaces a call to engine() that was done here in a previous version of the benchmark
    // Leaving it in here maintains compatibility with the results produced by that version.
    engine();

    // Running experiments
    const uint32_t experimentCount = configuration.at("experimentsToRun").size();
    std::cout << "Running experiments.." << std::endl;
    uint64_t experimentBaseRandomSeed = engine();

    const uint32_t intermediateSaveFrequency = configuration.at("commonExperimentSettings").at("intermediateSaveFrequency");
    std::cout << "    Results will be saved every " << intermediateSaveFrequency << " samples." << std::endl;
    std::mutex resultWriteLock;



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
                std::cout << "    Configuration file did not specify the extent to which the given results file should be replicated." << std::endl;
                std::cout << "    If your intent was to replicate the support radius or cached descriptors, you can ignore this message." << std::endl;
                std::cout << "    If you want to replicate experimental results, you should enable either full replication, or replicate a subset." << std::endl;
                std::cout << "        When using the included python script: you can change this in the replication settings under \"experiment\"" << std::endl;
                std::cout << "        When editing the configuration file directory: enable one of the boolean options under replicationOverrides > experiment" << std::endl;
                std::cout << "Exiting." << std::endl;
                return;
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
        uint32_t completedCount = 0;
        std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();

        uint32_t threadsToLaunch = omp_get_max_threads();
        if(experimentConfig.contains("threadLimit")) {
            threadsToLaunch = experimentConfig.at("threadLimit");
        }

        #pragma omp parallel for schedule(dynamic) num_threads(threadsToLaunch) default(none) shared(setup, replicationSubset, completedCount, startTime, fileCache, sampleVerticesSet, resultsDirectory, intermediateSaveFrequency, experimentResult, enablePRCComparisonMode, referenceDescriptors, supportRadius, configuration, sampleSetSize, verticesPerSampleObject, experimentRandomSeeds, dataset, resultWriteLock, threadActivity, std::cout, experimentIndex, experimentName, experimentConfig, filterInstanceMap)
        for (uint32_t sampleVertexIndex = 0; sampleVertexIndex < sampleSetSize; sampleVertexIndex += verticesPerSampleObject) {
            // This section processes a single sample object, filters it, and computes the DDI and PRC curve statistics

            // Skip this sample object if none of the results for its sample vertices need to be replicated
            if(setup.replicationSettings.enabled) {
                bool objectContainsResultToReplicate = false;
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    objectContainsResultToReplicate = objectContainsResultToReplicate || replicationSubset.contains(sampleVertexIndex + i);
                }
                if(!objectContainsResultToReplicate) {
                    continue;
                }
            }

            // Record what this thread is working on. Used for reporting progress while the experiment is running
            {
                std::unique_lock<std::mutex> writeLock{resultWriteLock};
                if (threadActivity.empty()) {
                    threadActivity.resize(omp_get_num_threads());
                }
                threadActivity.at(omp_get_thread_num()) = sampleVertexIndex;
            }

            // Initialising several needed variables
            ShapeBench::randomEngine experimentInstanceRandomEngine(experimentRandomSeeds.at(sampleVertexIndex / verticesPerSampleObject));
            std::vector<ShapeBench::ExperimentResultsEntry> resultsEntries(verticesPerSampleObject);

            ShapeBench::VertexInDataset firstSampleVertex = sampleVerticesSet.at(sampleVertexIndex);
            uint32_t meshID = firstSampleVertex.meshID;
            const ShapeBench::DatasetEntry &entry = dataset.at(meshID);
            ShapeDescriptor::cpu::Mesh originalSampleMesh = ShapeBench::readDatasetMesh(configuration, fileCache, entry);

            // Creating the "mesh filtering context" object
            ShapeBench::FilteredMeshPair filteredSceneObject = initialiseFilteredMeshPair(verticesPerSampleObject, sampleVerticesSet, sampleVertexIndex, originalSampleMesh);
            ShapeBench::FilteredMeshPair filteredModelObject = initialiseFilteredMeshPair(verticesPerSampleObject, sampleVerticesSet, sampleVertexIndex, originalSampleMesh);

            // Run filters
            try {
                // Run each of the filters in this experiment in the sequence defined in the JSON file
                // This is done both on the scene and model object (depending on whether any filters are defined for these)
                nlohmann::json sceneFilterMetadata = applyFilterSequence(fileCache, "filters", configuration, dataset,
                                                                     filterInstanceMap, verticesPerSampleObject,
                                                                     experimentConfig, experimentInstanceRandomEngine,
                                                                     resultsEntries, filteredSceneObject);

                nlohmann::json modelFilterMetadata = applyFilterSequence(fileCache, "modelFilters", configuration, dataset,
                                                                         filterInstanceMap, verticesPerSampleObject,
                                                                         experimentConfig, experimentInstanceRandomEngine,
                                                                         resultsEntries, filteredModelObject);


                // Filter execution complete, computing DDI and relevant statistics
                const uint64_t areaEstimationRandomSeed = experimentInstanceRandomEngine();
                const uint64_t modelPointCloudSamplingSeed = experimentInstanceRandomEngine();
                const uint64_t modelDescriptorGenerationSeed = experimentInstanceRandomEngine();
                const uint64_t filteredPointCloudSamplingSeed = experimentInstanceRandomEngine();
                const uint64_t filteredDescriptorGenerationSeed = experimentInstanceRandomEngine();

                ShapeDescriptor::cpu::Mesh combinedSceneMesh = filteredSceneObject.combinedFilteredMesh();
                ShapeDescriptor::cpu::Mesh combinedModelMesh = filteredModelObject.combinedFilteredMesh();

                std::vector<DescriptorType> modelDescriptors(verticesPerSampleObject);
                std::vector<DescriptorType> filteredDescriptors(verticesPerSampleObject);
                std::vector<float> radii(verticesPerSampleObject, supportRadius);

                ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(
                        combinedModelMesh,
                        {filteredModelObject.mappedReferenceVertices.size(), filteredModelObject.mappedReferenceVertices.data()},
                        configuration,
                        radii,
                        modelPointCloudSamplingSeed,
                        modelDescriptorGenerationSeed,
                        filteredModelObject.pointCloudConversionScaleFactor,
                        modelDescriptors);

                ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(
                        combinedSceneMesh,
                        {filteredSceneObject.mappedReferenceVertices.size(), filteredSceneObject.mappedReferenceVertices.data()},
                        configuration,
                        radii,
                        filteredPointCloudSamplingSeed,
                        filteredDescriptorGenerationSeed,
                        filteredSceneObject.pointCloudConversionScaleFactor,
                        filteredDescriptors);


                // Compute statistics for the model and scene objects
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    // Both filter sequences must succeed for the result to be valid
                    resultsEntries.at(i).included = filteredSceneObject.mappedVertexIncluded.at(i) && filteredModelObject.mappedVertexIncluded.at(i);
                    if(!resultsEntries.at(i).included) {
                        continue;
                    }

                    resultsEntries.at(i).sourceVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                    resultsEntries.at(i).modelObject.originalVertexLocation = filteredModelObject.originalReferenceVertices.at(i);
                    resultsEntries.at(i).modelObject.filteredVertexLocation = filteredModelObject.mappedReferenceVertices.at(i);
                    resultsEntries.at(i).modelObject.filterOutput = modelFilterMetadata;
                    resultsEntries.at(i).sceneObject.originalVertexLocation = filteredSceneObject.originalReferenceVertices.at(i);
                    resultsEntries.at(i).sceneObject.filteredVertexLocation = filteredSceneObject.mappedReferenceVertices.at(i);
                    resultsEntries.at(i).sceneObject.filterOutput = sceneFilterMetadata;


                    // Compute Descriptor Distance Index value
                    ShapeBench::DescriptorOfVertexInDataset<DescriptorType> modelFilteredDescriptor;
                    modelFilteredDescriptor.vertex = filteredModelObject.originalReferenceVertices.at(i);
                    modelFilteredDescriptor.descriptor = modelDescriptors.at(i);
                    modelFilteredDescriptor.meshID = sampleVerticesSet.at(sampleVertexIndex + i).meshID;
                    modelFilteredDescriptor.vertexIndex = sampleVerticesSet.at(sampleVertexIndex + i).vertexIndex;

                    ShapeBench::DescriptorOfVertexInDataset<DescriptorType> sceneFilteredDescriptor;
                    sceneFilteredDescriptor.vertex = filteredSceneObject.originalReferenceVertices.at(i);
                    sceneFilteredDescriptor.descriptor = filteredDescriptors.at(i);
                    sceneFilteredDescriptor.meshID = sampleVerticesSet.at(sampleVertexIndex + i).meshID;
                    sceneFilteredDescriptor.vertexIndex = sampleVerticesSet.at(sampleVertexIndex + i).vertexIndex;

                    ShapeBench::MatchingPerformanceMeasure performanceResult = ShapeBench::computeAchievedMatchingPerformance<DescriptorMethod, DescriptorType>(modelFilteredDescriptor, sceneFilteredDescriptor, referenceDescriptors, enablePRCComparisonMode);
                    resultsEntries.at(i).prcMetadata = performanceResult.prcMeasure;
                    resultsEntries.at(i).filteredDescriptorRank = performanceResult.descriptorDistanceIndex;


                    // Compute clutter and occlusion values
                    ShapeBench::AreaEstimate modelAreaEstimate;
                    modelAreaEstimate.addedArea = 0;
                    modelAreaEstimate.subtractiveArea = 1;
                    bool modelObjectHasFilters = experimentConfig.contains("modelFilters") && !experimentConfig.at("modelFilters").empty();
                    // Can use default values if object was not modified by a filter
                    if(modelObjectHasFilters) {
                        modelAreaEstimate = ShapeBench::estimateAreaInSupportVolume<DescriptorMethod>(filteredModelObject, resultsEntries.at(i).modelObject.originalVertexLocation, resultsEntries.at(i).modelObject.filteredVertexLocation, supportRadius, configuration, areaEstimationRandomSeed);
                    }
                    resultsEntries.at(i).modelObject.fractionAddedNoise = modelAreaEstimate.addedArea;
                    resultsEntries.at(i).modelObject.fractionSurfacePartiality = modelAreaEstimate.subtractiveArea;

                    ShapeBench::AreaEstimate sceneAreaEstimate;
                    sceneAreaEstimate.addedArea = 0;
                    sceneAreaEstimate.subtractiveArea = 1;
                    bool sceneObjectHasFilters = experimentConfig.contains("filters") && !experimentConfig.at("filters").empty();
                    if(sceneObjectHasFilters) {
                        sceneAreaEstimate = ShapeBench::estimateAreaInSupportVolume<DescriptorMethod>(filteredSceneObject, resultsEntries.at(i).sceneObject.originalVertexLocation, resultsEntries.at(i).sceneObject.filteredVertexLocation, supportRadius, configuration, areaEstimationRandomSeed);
                    }
                    resultsEntries.at(i).sceneObject.fractionAddedNoise = sceneAreaEstimate.addedArea;
                    resultsEntries.at(i).sceneObject.fractionSurfacePartiality = sceneAreaEstimate.subtractiveArea;
                }


                ShapeDescriptor::free(combinedModelMesh);
                ShapeDescriptor::free(combinedSceneMesh);

                {
                    std::unique_lock<std::mutex> writeLock{resultWriteLock};
                    for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                        experimentResult.vertexResults.at(sampleVertexIndex + i) = resultsEntries.at(i);
                    }
                    if(configuration.contains("verboseOutput") && configuration.at("verboseOutput")) {
                        std::cout << "Added area, Remaining area, Rank" << (enablePRCComparisonMode ? ", NN distance, SNN distance, Tao, Vertex distance, In range, MeshID 1, MeshID 2, Same MeshID" : "") << std::endl;
                        for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                            if(resultsEntries.at(i).included) {
                                std::cout << fmt::format("Result: {:<10}, {:<10}, {:<10}", resultsEntries.at(i).sceneObject.fractionAddedNoise, resultsEntries.at(i).sceneObject.fractionSurfacePartiality, resultsEntries.at(i).filteredDescriptorRank);
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
            } catch (const std::exception &e) {
                std::cout << "    Failed to process vertex " << sampleVertexIndex << ": " << e.what() << std::endl;
            }

            ShapeDescriptor::free(originalSampleMesh);
            filteredSceneObject.free();
            filteredModelObject.free();

            {
                std::unique_lock<std::mutex> writeLock{resultWriteLock};
                completedCount += verticesPerSampleObject;
                bool isLastVertexIndex = completedCount == sampleSetSize;
                if(sampleVertexIndex % 100 == 0 || isLastVertexIndex) {
                    std::cout << "\r    ";
                    ShapeBench::drawProgressBar(completedCount, sampleSetSize);
                    std::cout << " " << (completedCount) << "/" << sampleSetSize;
                    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
                    std::chrono::duration<uint64_t, std::nano> elapsedTimeThusFar = currentTime - startTime;
                    uint64_t elapsedTimeNanoseconds = elapsedTimeThusFar.count();
                    double elapsedTimeSeconds = double(elapsedTimeNanoseconds) / 1000000000.0;
                    double expectedTotalTimeSeconds = elapsedTimeSeconds * (double(sampleSetSize) / double(completedCount));
                    std::chrono::duration<uint64_t, std::nano> expectedTotalTime
                            = std::chrono::nanoseconds(uint64_t(expectedTotalTimeSeconds * 1000000000.0));
                    std::cout << " - Time taken: " << ShapeBench::durationToString(elapsedTimeThusFar);
                    std::cout << "/" << ShapeBench::durationToString(expectedTotalTime);
                    std::cout << ", remaining: " << ShapeBench::durationToString(expectedTotalTime - elapsedTimeThusFar) << std::endl;


                    // Force "garbage collection" in libc
                    malloc_trim(0);
                }


                std::cout << "Processing method " << DescriptorMethod::getName() << ", experiment " << experimentName << ": completed " << sampleVertexIndex << "/" << sampleSetSize << " - " << entry.vertexCount << " vertices - Threads: (";
                for(uint32_t i = 0; i < threadActivity.size(); i++) {
                    std::cout << threadActivity.at(i) << (i + 1 < threadActivity.size() ? ", " : "");
                }
                std::cout << ")" << std::endl;

                if(sampleVertexIndex % intermediateSaveFrequency == 0) {
                    std::cout << std::endl << "    Writing caches.." << std::endl;
                    writeExperimentResults(experimentResult, resultsDirectory, false, enablePRCComparisonMode, setup.replicationSettings);
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

        std::cout << "Writing experiment results file.." << std::endl;
        writeExperimentResults(experimentResult, resultsDirectory, true, enablePRCComparisonMode, setup.replicationSettings);
        std::cout << "Experiment complete." << std::endl;

        if(setup.replicationSettings.enabled) {
            ShapeBench::checkReplicatedExperimentResults(configuration, DescriptorMethod::getName(), experimentName,experimentResult, setup.replicationSettings.experimentResults);
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

    std::cout << "Experiments completed." << std::endl;
}