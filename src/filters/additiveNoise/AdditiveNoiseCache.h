#pragma once

#include <unordered_map>
#include <cstdint>
#include <vector>
#include <mutex>
#include <fstream>
#include "utils/Orientation.h"
#include "nlohmann/json.hpp"
#include "AdditiveNoiseFilterSettings.h"

namespace ShapeBench {
    class AdditiveNoiseCache {
        std::unordered_map<uint64_t, uint32_t> startIndexMap;
        std::vector<ShapeBench::Orientation> objectOrientations;
        const std::string cacheFileName = "additiveNoiseSceneCache.bin";
        const std::string headerIdentifier = "ADDITIVENOISECACHE";
        uint32_t objectsPerEntry = 0;
        std::mutex cacheLock;
    public:
        bool contains(uint64_t randomSeed);
        void set(uint64_t randomSeed, const std::vector<ShapeBench::Orientation>& objectOrientations);
        std::vector<ShapeBench::Orientation> get(uint64_t randomSeed);

        // There's somehow a bunch of problems compiling regular functions called from a template with a config object as parameter
        // This is a workaround to avoid those problems, as ugly as it might be
        friend inline void loadAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config, bool loadEmpty);
        friend inline void saveAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config);

        long entryCount();
    };

    inline void reportNoiseCacheReadError(std::string reason, std::filesystem::path path) {
        throw std::runtime_error("Failed to read cache file at: " + path.string() + " - Reason: " + reason);
    }

    inline void loadAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config, bool loadEmpty) {
        std::unique_lock<std::mutex> lock {cache.cacheLock};
        std::filesystem::path cacheFilePath = std::filesystem::path(std::string(config.at("cacheDirectory"))) / cache.cacheFileName;
        uint32_t configuredObjectsPerEntry = uint32_t(config.at("filterSettings").at("additiveNoise").at("addedObjectCount")) + 1;

        // Objects per entry = 1 reference mesh + n added noise objects
        cache.objectsPerEntry = configuredObjectsPerEntry;

        if(std::filesystem::exists(cacheFilePath) && !loadEmpty) {
            std::ifstream inputFile(cacheFilePath, std::ios::binary);
            if(!inputFile) {
                reportNoiseCacheReadError("Failed to open file", cacheFilePath);
            }

            const nlohmann::json& additiveNoiseSettings = config.at("filterSettings").at("additiveNoise");
            AdditiveNoiseFilterSettings filterSettings = readAdditiveNoiseFilterSettings(config, additiveNoiseSettings);

            std::string readHeaderString;
            readHeaderString.resize(cache.headerIdentifier.size());
            inputFile.read(readHeaderString.data(), uint32_t(cache.headerIdentifier.size()));
            if(readHeaderString != cache.headerIdentifier) {
                reportNoiseCacheReadError("File header is invalid", cacheFilePath);
            }

            uint32_t headerVersion = 0;
            inputFile.read((char*) &headerVersion, sizeof(uint32_t));
            if(headerVersion != 1) {
                reportNoiseCacheReadError("Unsupported file format revision: " + std::to_string(headerVersion), cacheFilePath);
            }

            uint32_t entryCount = 0;
            inputFile.read((char*) &entryCount, sizeof(uint32_t));
            //std::cout << "    Cache has " << entryCount << " entries" << std::endl;

            cache.objectOrientations.resize(entryCount * cache.objectsPerEntry);
            cache.startIndexMap.reserve(entryCount);

            
            uint32_t parameter_addedClutterObjectCount = 0;
            inputFile.read((char*) &parameter_addedClutterObjectCount, sizeof(uint32_t));
            if(parameter_addedClutterObjectCount != filterSettings.addedClutterObjectCount) {
                reportNoiseCacheReadError("Simulator parameter \"addedClutterObjectCount\" does not match with the cached value (" + std::to_string(filterSettings.addedClutterObjectCount) + " versus " + std::to_string(parameter_addedClutterObjectCount), cacheFilePath);
            }
            uint32_t parameter_convexHullGenerationMaxVerticesPerHull = 0;
            inputFile.read((char*) &parameter_convexHullGenerationMaxVerticesPerHull, sizeof(uint32_t));
            if(parameter_convexHullGenerationMaxVerticesPerHull != filterSettings.convexHullGenerationMaxVerticesPerHull) {
                reportNoiseCacheReadError("Simulator parameter \"convexHullGenerationMaxVerticesPerHull\" does not match with the cached value (" + std::to_string(filterSettings.convexHullGenerationMaxVerticesPerHull) + " versus " + std::to_string(parameter_convexHullGenerationMaxVerticesPerHull), cacheFilePath);
            }
            uint32_t parameter_simulationFrameRate = 0;
            inputFile.read((char*) &parameter_simulationFrameRate, sizeof(uint32_t));
            if(parameter_simulationFrameRate != filterSettings.simulationFrameRate) {
                reportNoiseCacheReadError("Simulator parameter \"simulationFrameRate\" does not match with the cached value (" + std::to_string(filterSettings.simulationFrameRate) + " versus " + std::to_string(parameter_simulationFrameRate), cacheFilePath);
            }
            uint32_t parameter_convexHullGenerationRecursionDepth = 0;
            inputFile.read((char*) &parameter_convexHullGenerationRecursionDepth, sizeof(uint32_t));
            if(parameter_convexHullGenerationRecursionDepth != filterSettings.convexHullGenerationRecursionDepth) {
                reportNoiseCacheReadError("Simulator parameter \"convexHullGenerationRecursionDepth\" does not match with the cached value (" + std::to_string(filterSettings.convexHullGenerationRecursionDepth) + " versus " + std::to_string(parameter_convexHullGenerationRecursionDepth), cacheFilePath);
            }
            uint32_t parameter_convexHullGenerationResolution = 0;
            inputFile.read((char*) &parameter_convexHullGenerationResolution, sizeof(uint32_t));
            if(parameter_convexHullGenerationResolution != filterSettings.convexHullGenerationResolution) {
                reportNoiseCacheReadError("Simulator parameter \"convexHullGenerationResolution\" does not match with the cached value (" + std::to_string(filterSettings.convexHullGenerationResolution) + " versus " + std::to_string(parameter_convexHullGenerationResolution), cacheFilePath);
            }
            uint32_t parameter_maxConvexHulls = 0;
            inputFile.read((char*) &parameter_maxConvexHulls, sizeof(uint32_t));
            if(parameter_maxConvexHulls != filterSettings.maxConvexHulls) {
                reportNoiseCacheReadError("Simulator parameter \"maxConvexHulls\" does not match with the cached value (" + std::to_string(filterSettings.maxConvexHulls) + " versus " + std::to_string(parameter_maxConvexHulls), cacheFilePath);
            }
            uint32_t parameter_simulationStepLimit = 0;
            inputFile.read((char*) &parameter_simulationStepLimit, sizeof(uint32_t));
            if(parameter_simulationStepLimit != filterSettings.simulationStepLimit) {
                reportNoiseCacheReadError("Simulator parameter \"simulationStepLimit\" does not match with the cached value (" + std::to_string(filterSettings.simulationStepLimit) + " versus " + std::to_string(parameter_simulationStepLimit), cacheFilePath);
            }
            float parameter_floorFriction = 0;
            inputFile.read((char*) &parameter_floorFriction, sizeof(float));
            if(parameter_floorFriction != filterSettings.floorFriction) {
                reportNoiseCacheReadError("Simulator parameter \"floorFriction\" does not match with the cached value (" + std::to_string(filterSettings.floorFriction) + " versus " + std::to_string(parameter_floorFriction), cacheFilePath);
            }
            float parameter_initialObjectSeparation = 0;
            inputFile.read((char*) &parameter_initialObjectSeparation, sizeof(float));
            if(parameter_initialObjectSeparation != filterSettings.initialObjectSeparation) {
                reportNoiseCacheReadError("Simulator parameter \"initialObjectSeparation\" does not match with the cached value (" + std::to_string(filterSettings.initialObjectSeparation) + " versus " + std::to_string(parameter_initialObjectSeparation), cacheFilePath);
            }
            float parameter_objectAttractionForceMagnitude = 0;
            inputFile.read((char*) &parameter_objectAttractionForceMagnitude, sizeof(float));
            if(parameter_objectAttractionForceMagnitude != filterSettings.objectAttractionForceMagnitude) {
                reportNoiseCacheReadError("Simulator parameter \"objectAttractionForceMagnitude\" does not match with the cached value (" + std::to_string(filterSettings.objectAttractionForceMagnitude) + " versus " + std::to_string(parameter_objectAttractionForceMagnitude), cacheFilePath);
            }

            for(uint32_t i = 0; i < entryCount; i++) {
                uint64_t randomSeed = 0;
                inputFile.read((char*) &randomSeed, sizeof(uint64_t));
                uint32_t mappedIndex = 0;
                inputFile.read((char*) &mappedIndex, sizeof(uint32_t));

                cache.startIndexMap.insert({randomSeed, mappedIndex});
            }

            inputFile.read((char*) cache.objectOrientations.data(), cache.objectOrientations.size() * sizeof(ShapeBench::Orientation));
        }
    }

    inline void saveAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config) {
        std::unique_lock<std::mutex> lock {cache.cacheLock};
        uint32_t header_version = 1;
        uint32_t header_entryCount = cache.startIndexMap.size();

        const nlohmann::json& additiveNoiseSettings = config.at("filterSettings").at("additiveNoise");
        AdditiveNoiseFilterSettings filterSettings = readAdditiveNoiseFilterSettings(config, additiveNoiseSettings);

        std::filesystem::path cacheFilePath = std::filesystem::path(std::string(config.at("cacheDirectory"))) / cache.cacheFileName;
        std::ofstream outputFile(cacheFilePath, std::ios::binary);

        // Writing header
        outputFile.write(cache.headerIdentifier.c_str(), uint32_t(cache.headerIdentifier.size()));
        outputFile.write((const char*) &header_version, sizeof(uint32_t));
        outputFile.write((const char*) &header_entryCount, sizeof(uint32_t));

        // Writing settings used to generate results
        outputFile.write((const char*) (&filterSettings.addedClutterObjectCount), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.convexHullGenerationMaxVerticesPerHull), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.simulationFrameRate), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.convexHullGenerationRecursionDepth), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.convexHullGenerationResolution), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.maxConvexHulls), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.simulationStepLimit), sizeof(uint32_t));
        outputFile.write((const char*) (&filterSettings.floorFriction), sizeof(float));
        outputFile.write((const char*) (&filterSettings.initialObjectSeparation), sizeof(float));
        outputFile.write((const char*) (&filterSettings.objectAttractionForceMagnitude), sizeof(float));

        // Write start index map
        for(const std::pair<const uint64_t, uint32_t>& mappedIndices : cache.startIndexMap) {
            outputFile.write((const char*) &mappedIndices.first, sizeof(uint64_t));
            outputFile.write((const char*) &mappedIndices.second, sizeof(uint32_t));
        }

        // Write computed placements
        outputFile.write((const char*) cache.objectOrientations.data(), cache.objectOrientations.size() * sizeof(ShapeBench::Orientation));
    }
}

