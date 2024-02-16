#pragma once

#include <unordered_map>
#include <cstdint>
#include <vector>
#include "utils/Orientation.h"
#include "json.hpp"
#include "AdditiveNoiseFilterSettings.h"

namespace ShapeBench {
    class AdditiveNoiseCache {
        std::unordered_map<uint64_t, uint32_t> startIndexMap;
        std::vector<ShapeBench::Orientation> objectOrientations;
        const std::string cacheFileName = "additiveNoiseSceneCache.bin";
        uint32_t objectsPerEntry = 0;
    public:
        bool contains(uint64_t randomSeed);
        void set(uint64_t randomSeed, const std::vector<ShapeBench::Orientation>& objectOrientations);
        std::vector<ShapeBench::Orientation> get(uint64_t randomSeed);

        // There's somehow a bunch of problems compiling regular functions called from a template with a config object as parameter
        // This is a workaround to avoid those problems, as ugly as it might be
        friend inline void loadAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config);
        friend inline void saveAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config);
    };

    inline void loadAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config) {
        std::filesystem::path cacheFilePath = std::filesystem::path(config.at("cacheDirectory")) / cache.cacheFileName;

        if(std::filesystem::exists(cacheFilePath)) {

        } else {
            // Objects per entry = 1 reference mesh + n added noise objects
            cache.objectsPerEntry = uint32_t(config.at("filterSettings").at("additiveNoise").at("addedObjectCount")) + 1;
        }
    }

    inline void saveAdditiveNoiseCache(AdditiveNoiseCache& cache, const nlohmann::json& config) {
        std::string header_identifier = "ADDITIVENOISECACHE";
        int32_t header_version = 1;
        uint32_t header_entryCount = cache.startIndexMap.size();

        const nlohmann::json& additiveNoiseSettings = config.at("filterSettings").at("additiveNoise");

        AdditiveNoiseFilterSettings filterSettings = readAdditiveNoiseFilterSettings(config, additiveNoiseSettings);

        std::filesystem::path cacheFilePath = std::filesystem::path(config.at("cacheDirectory")) / cache.cacheFileName;
        std::ofstream outputFile(cacheFilePath, std::ios::binary);

        // Writing header
        outputFile.write(header_identifier.c_str(), uint32_t(header_identifier.size()));
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

