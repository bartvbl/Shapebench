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
ShapeDescriptor::cpu::array<DescriptorType> computeReferenceDescriptors(const std::vector<ShapeBench::ChosenVertexPRC>& representativeSet, const nlohmann::json& config, const ShapeBench::Dataset& dataset, uint64_t randomSeed, float supportRadius) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    ShapeBench::randomEngine randomEngine(randomSeed);
    std::vector<uint64_t> randomSeeds(representativeSet.size());
    for(uint32_t i = 0; i < representativeSet.size(); i++) {
        randomSeeds.at(i) = randomEngine();
    }

    ShapeDescriptor::cpu::array<DescriptorType> representativeDescriptors(representativeSet.size());
    uint32_t completedCount = 0;

    #pragma omp parallel for schedule(dynamic) default(none) shared(representativeSet, supportRadius, dataset, config, randomSeeds, representativeDescriptors, completedCount, std::cout)
    for(int i = 0; i < representativeSet.size(); i++) {
        uint32_t currentMeshID = representativeSet.at(i).meshID;
        if(i > 0 && representativeSet.at(i - 1).meshID == currentMeshID) {
            continue;
        }
        uint32_t sameMeshCount = 1;
        for(int j = i + 1; j < representativeSet.size(); j++) {
            uint32_t meshID = representativeSet.at(j).meshID;
            if(currentMeshID != meshID) {
                break;
            }
            sameMeshCount++;
        }
        std::vector<DescriptorType> outputDescriptors(sameMeshCount);
        std::vector<ShapeDescriptor::OrientedPoint> descriptorOrigins(sameMeshCount);
        std::vector<float> radii(sameMeshCount, supportRadius);

        const ShapeBench::DatasetEntry& entry = dataset.at(representativeSet.at(i).meshID);
        ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(config, entry);

        for(int j = 0; j < sameMeshCount; j++) {
            uint32_t entryIndex = j + i;
            descriptorOrigins.at(j) = representativeSet.at(entryIndex).vertex;
        }

        ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> originArray(descriptorOrigins.size(), descriptorOrigins.data());

        ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(mesh, originArray, config, radii, randomSeeds.at(i), randomSeeds.at(i), outputDescriptors);
        for(int j = 0; j < sameMeshCount; j++) {
            uint32_t entryIndex = j + i;
            representativeDescriptors[entryIndex] = outputDescriptors.at(j);
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

    #pragma omp parallel for schedule(dynamic) default(none) shared(representativeSet, supportRadius, dataset, config, randomSeeds, representativeDescriptors, completedCount, std::cout)
    for(int i = 0; i < representativeSet.size(); i++) {
        uint32_t currentMeshID = representativeSet.at(i).meshID;
        if(i > 0 && representativeSet.at(i - 1).meshID == currentMeshID) {
            continue;
        }
        uint32_t sameMeshCount = 1;
        for(int j = i + 1; j < representativeSet.size(); j++) {
            uint32_t meshID = representativeSet.at(j).meshID;
            if(currentMeshID != meshID) {
                break;
            }
            sameMeshCount++;
        }
        std::vector<DescriptorType> outputDescriptors(sameMeshCount);
        std::vector<ShapeDescriptor::OrientedPoint> descriptorOrigins(sameMeshCount);
        std::vector<float> radii(sameMeshCount, supportRadius);

        const ShapeBench::DatasetEntry& entry = dataset.at(representativeSet.at(i).meshID);
        ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(config, entry);

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
            representativeDescriptors[entryIndex] = outputDescriptors.at(j);
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
ShapeDescriptor::cpu::array<DescriptorType> computeDescriptorsOrLoadCached(
        const nlohmann::json &configuration,
        const ShapeBench::Dataset &dataset,
        float supportRadius,
        uint64_t representativeSetRandomSeed,
        const std::vector<inputRepresentativeSetType> &representativeSet,
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
        std::cout << "    Successfully loaded " << referenceDescriptors.length << " descriptors" << std::endl;
        std::string errorMessage = "Mismatch detected between the cached number of descriptors ("
                + std::to_string(referenceDescriptors.length) + ") and the requested number of descriptors ("
                + std::to_string(representativeSet.size()) + ").";
        if(referenceDescriptors.length < representativeSet.size()) {
            throw std::runtime_error("ERROR: " + errorMessage + " The number of cached descriptors is not sufficient. "
                                                                "Consider regenerating the cache by deleting the file: " + descriptorCacheFile.string());
        } else if(referenceDescriptors.length > representativeSet.size()) {
            std::cout << "    WARNING: " + errorMessage + " Since the cache contains more descriptors than necessary, execution will continue." << std::endl;
        }
    }
    return referenceDescriptors;
}

template<typename DescriptorMethod, typename DescriptorType>
void testMethod(const nlohmann::json& configuration, const std::filesystem::path configFileLocation, const ShapeBench::Dataset& dataset, uint64_t randomSeed) {
    std::cout << std::endl << "========== TESTING METHOD " << DescriptorMethod::getName() << " ==========" << std::endl;
    std::cout << "Initialising.." << std::endl;
    ShapeBench::randomEngine engine(randomSeed);
    std::filesystem::path computedConfigFilePath = configFileLocation.parent_path() / std::string(configuration.at("computedConfigFile"));
    DescriptorMethod::init(configuration);
    std::cout << "    Main config file: " << configFileLocation.string() << std::endl;
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
    if (!computedConfig.containsKey(methodName, "supportRadius")) {
        std::cout << "    No support radius has been computed yet for this method." << std::endl;
        std::cout << "    Performing support radius estimation.." << std::endl;
        supportRadius = ShapeBench::estimateSupportRadius<DescriptorMethod, DescriptorType>(configuration, dataset, engine());
        std::cout << "    Chosen support radius: " << supportRadius << std::endl;
        computedConfig.setFloatAndSave(methodName, "supportRadius", supportRadius);
    } else {
        supportRadius = computedConfig.getFloat(methodName, "supportRadius");
        std::cout << "    Cached support radius was found for this method: " << supportRadius << std::endl;
        engine(); // Used for RNG consistency
    }

    // Computing sample descriptors and their distance to the representative set
    const uint32_t representativeSetSize = configuration.at("commonExperimentSettings").at("representativeSetSize");
    const uint32_t sampleSetSize = configuration.at("commonExperimentSettings").at("sampleSetSize");
    const uint32_t verticesPerSampleObject = configuration.at("commonExperimentSettings").at("verticesToTestPerSampleObject");
    const uint32_t verticesPerReferenceObject = configuration.at("commonExperimentSettings").at("verticesToTestPerReferenceObject");
    const uint32_t representativeSetObjectCount = representativeSetSize / verticesPerSampleObject;
    const uint32_t sampleSetObjectCount = sampleSetSize / verticesPerSampleObject;

    // Compute reference descriptors, or load them from a cache file
    ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors;
    ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptorsPRC;
    ShapeDescriptor::cpu::array<DescriptorType> cleanSampleDescriptors;
    std::vector<ShapeBench::VertexInDataset> representativeSet;
    std::vector<ShapeBench::ChosenVertexPRC> representativeSetPRC;
    std::vector<ShapeBench::VertexInDataset> sampleVerticesSet;
    std::vector<ShapeBench::ChosenVertexPRC> sampleSetPRC;

    std::cout << "Computing reference descriptor set.. (" << verticesPerReferenceObject << " vertices per object)" << std::endl;
    representativeSet = dataset.sampleVertices(engine(), representativeSetSize, verticesPerReferenceObject);
    referenceDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::VertexInDataset>(configuration, dataset, supportRadius, engine(), representativeSet, "reference");

    if(!enablePRCComparisonMode) {
        std::cout << "Computing sample descriptor set.. (" << verticesPerSampleObject << " vertices per object)" << std::endl;
        sampleVerticesSet = dataset.sampleVertices(engine(), sampleSetSize, verticesPerSampleObject);
        cleanSampleDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::VertexInDataset>(configuration, dataset, supportRadius, engine(), sampleVerticesSet, "sample");
    } else {
        assert(representativeSetSize == sampleSetSize);
        assert(verticesPerReferenceObject == verticesPerSampleObject);
        // Both the PRC and ShapeBench strategies have the same set of sample vertices
        // The reference set of ShapeBench is the same it always was
        // The one for PRC is the same set of objects, but different vertices.

        // Both paths of this if statement use the primary random number generator 4 times
        std::cout << "Computing reference and sample descriptor set.." << std::endl;
        std::cout << "    The reference and sampling set generation mode has been overridden (caused by PRC comparison mode)" << std::endl;
        std::vector<ShapeBench::VertexInDataset> randomObjectList = dataset.sampleVertices(engine(), sampleSetObjectCount, 1);
        representativeSetPRC.resize(representativeSetObjectCount * verticesPerReferenceObject);
        sampleSetPRC.resize(sampleSetObjectCount * verticesPerSampleObject);
        uint32_t completedCount = 0;

        ShapeBench::randomEngine PRCEngine(engine());
        std::vector<uint32_t> cloudSamplingSeeds(2 * sampleSetObjectCount);
        for(uint32_t i = 0; i < cloudSamplingSeeds.size(); i++) {
            cloudSamplingSeeds.at(i) = PRCEngine();
        }

        std::cout << "    Computing PRC surface point samples.." << std::endl;
        #pragma omp parallel for schedule(dynamic)
        for(uint32_t objectIndex = 0; objectIndex < sampleSetObjectCount; objectIndex++) {
            uint32_t vertexBaseIndex = verticesPerReferenceObject * objectIndex;

            ShapeDescriptor::cpu::Mesh mesh = ShapeBench::readDatasetMesh(configuration, dataset.at(randomObjectList.at(objectIndex).meshID));
            ShapeDescriptor::cpu::PointCloud referenceCloud = ShapeDescriptor::sampleMesh(mesh, verticesPerReferenceObject, cloudSamplingSeeds.at(2 * objectIndex + 0));
            ShapeDescriptor::cpu::PointCloud sampleCloud = ShapeDescriptor::sampleMesh(mesh, verticesPerReferenceObject, cloudSamplingSeeds.at(2 * objectIndex + 1));

            for(uint32_t vertexInObjectIndex = 0; vertexInObjectIndex < verticesPerReferenceObject; vertexInObjectIndex++) {
                uint32_t vertexIndex = vertexBaseIndex + vertexInObjectIndex;
                uint32_t chosenMeshID = randomObjectList.at(objectIndex).meshID;

                ShapeDescriptor::OrientedPoint referenceVertex;
                referenceVertex.vertex = referenceCloud.vertices[vertexInObjectIndex];
                referenceVertex.normal = referenceCloud.normals[vertexInObjectIndex];
                representativeSetPRC.at(vertexIndex).meshID = chosenMeshID;
                representativeSetPRC.at(vertexIndex).vertex = referenceVertex;

                ShapeDescriptor::OrientedPoint sampleVertex;
                sampleVertex.vertex = sampleCloud.vertices[vertexInObjectIndex];
                sampleVertex.normal = sampleCloud.normals[vertexInObjectIndex];
                sampleSetPRC.at(vertexIndex).meshID = chosenMeshID;
                sampleSetPRC.at(vertexIndex).vertex = sampleVertex;
            }

            #pragma omp atomic
            completedCount++;

            std::cout << "\r    ";
            ShapeBench::drawProgressBar(completedCount, sampleSetObjectCount);
            std::cout << " " << completedCount << "/" << sampleSetObjectCount << std::flush;

            ShapeDescriptor::free(referenceCloud);
            ShapeDescriptor::free(sampleCloud);
            ShapeDescriptor::free(mesh);
            malloc_trim(0);
        }
        std::cout << std::endl;

        referenceDescriptorsPRC = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::ChosenVertexPRC>(configuration, dataset, supportRadius, PRCEngine(), representativeSetPRC, "reference_PRC");
        cleanSampleDescriptors = computeDescriptorsOrLoadCached<DescriptorType, DescriptorMethod, ShapeBench::ChosenVertexPRC>(configuration, dataset, supportRadius, PRCEngine(), sampleSetPRC, "sample_PRC");
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
            ShapeDescriptor::writeDescriptorImages({count, reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(referenceDescriptors.content) + startIndex}, referenceImageFile, false);
            ShapeDescriptor::writeDescriptorImages({count, reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(cleanSampleDescriptors.content) + startIndex}, sampleImageFile, false);
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
            filterInstanceMap.at(filterName)->init(configuration);
        }

        std::vector<uint32_t> threadActivity;

        uint32_t threadsToLaunch = omp_get_max_threads();
        if(experimentConfig.contains("threadLimit")) {
            threadsToLaunch = experimentConfig.at("threadLimit");
        }

        #pragma omp parallel for schedule(dynamic) num_threads(threadsToLaunch) default(none) shared(resultsDirectory, intermediateSaveFrequency, experimentResult, representativeSetPRC, referenceDescriptorsPRC, sampleSetPRC, enablePRCComparisonMode, cleanSampleDescriptors, referenceDescriptors, illustrationImages, supportRadius, configuration, sampleSetSize, verticesPerSampleObject, illustrativeObjectStride, experimentRandomSeeds, sampleVerticesSet, dataset, enableIllustrationGenerationMode, resultWriteLock, threadActivity, std::cout, illustrativeObjectLimit, experimentIndex, experimentName, illustrativeObjectOutputDirectory, experimentConfig, filterInstanceMap)
        for (uint32_t sampleVertexIndex = 0; sampleVertexIndex < sampleSetSize; sampleVertexIndex += verticesPerSampleObject * illustrativeObjectStride) {
            ShapeBench::randomEngine experimentInstanceRandomEngine(experimentRandomSeeds.at(sampleVertexIndex / verticesPerSampleObject));

            uint32_t meshID = 0;
            if(!enablePRCComparisonMode) {
                ShapeBench::VertexInDataset firstSampleVertex = sampleVerticesSet.at(sampleVertexIndex);
                meshID = firstSampleVertex.meshID;
            } else {
                meshID = sampleSetPRC.at(sampleVertexIndex).meshID;
            }

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


            ShapeDescriptor::cpu::Mesh originalSampleMesh = ShapeBench::readDatasetMesh(configuration, entry);

            ShapeBench::FilteredMeshPair filteredMesh;
            filteredMesh.originalMesh = originalSampleMesh.clone();
            filteredMesh.filteredSampleMesh = originalSampleMesh.clone();

            filteredMesh.mappedReferenceVertices.resize(verticesPerSampleObject);
            filteredMesh.originalReferenceVertices.resize(verticesPerSampleObject);
            filteredMesh.mappedReferenceVertexIndices.resize(verticesPerSampleObject);
            filteredMesh.mappedVertexIncluded.resize(verticesPerSampleObject);
            for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                if(!enablePRCComparisonMode) {
                    ShapeBench::VertexInDataset sampleVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                    filteredMesh.originalReferenceVertices.at(i).vertex = filteredMesh.originalMesh.vertices[sampleVertex.vertexIndex];
                    filteredMesh.originalReferenceVertices.at(i).normal = filteredMesh.originalMesh.normals[sampleVertex.vertexIndex];
                    filteredMesh.mappedReferenceVertexIndices.at(i) = sampleVertex.vertexIndex;
                } else {
                    filteredMesh.originalReferenceVertices.at(i) = sampleSetPRC.at(i).vertex;
                    filteredMesh.mappedReferenceVertexIndices.at(i) = 0xFFFFFFFF; // These are surface samples, not indexed vertices
                }
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

                    ShapeBench::FilterOutput output = filterInstanceMap.at(filterType)->apply(configuration, filteredMesh, dataset, filterRandomSeed);
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
                ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(
                        combinedMesh,
                        {filteredMesh.mappedReferenceVertices.size(), filteredMesh.mappedReferenceVertices.data()},
                        configuration,
                        radii,
                        pointCloudSamplingSeed,
                        sampleDescriptorGenerationSeed,
                        filteredDescriptors);

                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntries.at(i).included = filteredMesh.mappedVertexIncluded.at(i);
                    if(!resultsEntries.at(i).included) {
                        if(DescriptorMethod::getName() == "QUICCI" && enableIllustrationGenerationMode) {
                            illustrationImages.content[(sampleVertexIndex/illustrativeObjectStride) + i] = filteredDescriptors.at(i);
                        }
                        continue;
                    }

                    if(!enablePRCComparisonMode) {
                        resultsEntries.at(i).sourceVertex = sampleVerticesSet.at(sampleVertexIndex + i);
                    } else {
                        resultsEntries.at(i).sourceVertex = {sampleSetPRC.at(sampleVertexIndex + i).meshID, 0xFFFFFFFF};
                    }
                    resultsEntries.at(i).filteredDescriptorRank = 0;
                    resultsEntries.at(i).originalVertexLocation = filteredMesh.originalReferenceVertices.at(i);
                    resultsEntries.at(i).filteredVertexLocation = filteredMesh.mappedReferenceVertices.at(i);

                    if(enableIllustrationGenerationMode) {
                        if(DescriptorMethod::getName() == "QUICCI") {
                            illustrationImages.content[(sampleVertexIndex/illustrativeObjectStride) + i] = filteredDescriptors.at(i);
                        }
                        continue;
                    }

                    uint32_t imageIndex = ShapeBench::computeImageIndex<DescriptorMethod, DescriptorType>(cleanSampleDescriptors[sampleVertexIndex + i], filteredDescriptors.at(i), referenceDescriptors);

                    if(enablePRCComparisonMode) {
                        //TODO: compute PRC info struct
                        resultsEntries.at(i).prcMetadata = ShapeBench::computePRCInfo<DescriptorMethod, DescriptorType>(
                                filteredDescriptors.at(i),
                                referenceDescriptorsPRC,
                                sampleSetPRC.at(sampleVertexIndex),
                                filteredMesh.mappedReferenceVertices.at(i),
                                representativeSetPRC);
                    }

                    ShapeBench::AreaEstimate areaEstimate = ShapeBench::estimateAreaInSupportVolume<DescriptorMethod>(filteredMesh, resultsEntries.at(i).originalVertexLocation, resultsEntries.at(i).filteredVertexLocation, supportRadius, configuration, areaEstimationRandomSeed);

                    resultsEntries.at(i).filteredDescriptorRank = imageIndex;
                    resultsEntries.at(i).fractionAddedNoise = areaEstimate.addedAdrea;
                    resultsEntries.at(i).fractionSurfacePartiality = areaEstimate.subtractiveArea;
                }


                ShapeDescriptor::free(combinedMesh);

                if(!enableIllustrationGenerationMode) {
                    std::unique_lock<std::mutex> writeLock{resultWriteLock};
                    for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                        experimentResult.vertexResults.at(sampleVertexIndex + i) = resultsEntries.at(i);
                        if(resultsEntries.at(i).included) {
                            std::cout << fmt::format("Result: added area: {:<10} remaining area: {:<10} rank: {:<6}", resultsEntries.at(i).fractionAddedNoise, resultsEntries.at(i).fractionSurfacePartiality, resultsEntries.at(i).filteredDescriptorRank);
                            if(enablePRCComparisonMode) {
                                float nearestNeighbourDistance = resultsEntries.at(i).prcMetadata.distanceToNearestNeighbour;
                                float secondNearestNeighbourDistance = resultsEntries.at(i).prcMetadata.distanceToSecondNearestNeighbour;
                                float tao = secondNearestNeighbourDistance == 0 ? 0 : nearestNeighbourDistance / secondNearestNeighbourDistance;

                                float distanceToNearestNeighbourVertex = length(resultsEntries.at(i).prcMetadata.nearestNeighbourVertexModel - resultsEntries.at(i).prcMetadata.nearestNeighbourVertexScene);
                                bool isInRange = distanceToNearestNeighbourVertex <= (supportRadius/2.0);

                                uint32_t modelMeshID = resultsEntries.at(i).prcMetadata.modelPointMeshID;
                                uint32_t sceneMeshID = resultsEntries.at(i).prcMetadata.scenePointMeshID;
                                bool isSameObject = modelMeshID == sceneMeshID;
                                std::cout << fmt::format(" Tao: {:<6}/{:<6} = {:<10} in range: {:<6}<={:<6} -> {:<5} same object: {:<6} == {:<6} -> {:<5}",
                                                         nearestNeighbourDistance, secondNearestNeighbourDistance, tao,
                                                         distanceToNearestNeighbourVertex, supportRadius/2.0f, (isInRange ? "true" : "false"),
                                                         modelMeshID, sceneMeshID, (isSameObject ? "true" : "false"));
                            }
                            std::cout << std::endl;
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
                        writeExperimentResults(experimentResult, resultsDirectory, false);
                        for(uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
                            std::string filterName = experimentConfig.at("filters").at(filterStepIndex).at("type");
                            filterInstanceMap.at(filterName)->saveCaches(configuration);
                        }
                    }

                    // Some slight race condition here, but does not matter since it's only used for printing
                    threadActivity.at(omp_get_thread_num()) = 0;
                }
            }
        }

        if(!enableIllustrationGenerationMode) {
            std::cout << "Writing experiment results file.." << std::endl;
            writeExperimentResults(experimentResult, resultsDirectory, true);
            std::cout << "Experiment complete." << std::endl;
        } else {
            std::string fileName = "descriptors_" + DescriptorMethod::getName() + "_" + ShapeDescriptor::generateUniqueFilenameString() + "_" + std::string(experimentConfig.at("name")) + ".png";
            std::filesystem::path outputFilePath = illustrativeObjectOutputDirectory / fileName;
            if(DescriptorMethod::getName() == "QUICCI") {
                ShapeDescriptor::writeDescriptorImages({illustrationImages.length, reinterpret_cast<ShapeDescriptor::QUICCIDescriptor*>(illustrationImages.content)}, outputFilePath, false);
            }
        }

        std::cout << "Writing caches.." << std::endl;
        for (uint32_t filterStepIndex = 0; filterStepIndex < experimentConfig.at("filters").size(); filterStepIndex++) {
            std::string filterName = experimentConfig.at("filters").at(filterStepIndex).at("type");
            filterInstanceMap.at(filterName)->saveCaches(configuration);
            filterInstanceMap.at(filterName)->destroy();
        }
    }


    std::cout << "Cleaning up.." << std::endl;
    ShapeDescriptor::free(referenceDescriptors);
    ShapeDescriptor::free(cleanSampleDescriptors);
    if(enableIllustrationGenerationMode && DescriptorMethod::getName() == "QUICCI") {
        ShapeDescriptor::free(illustrationImages);
    }

    std::cout << "Experiments completed." << std::endl;
}

