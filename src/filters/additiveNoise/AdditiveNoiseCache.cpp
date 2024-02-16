

#include "AdditiveNoiseCache.h"
#include "additiveNoiseFilter.h"

void ShapeBench::AdditiveNoiseCache::load(const nlohmann::json &config) {
    std::filesystem::path cacheFilePath = std::filesystem::path(config.at("cacheDirectory")) / cacheFileName;

    if(std::filesystem::exists(cacheFilePath)) {

    } else {
        // Objects per entry = 1 reference mesh + n added noise objects
        objectsPerEntry = uint32_t(config.at("filterSettings").at("additiveNoise").at("addedObjectCount")) + 1;
    }
}

void ShapeBench::AdditiveNoiseCache::save(const nlohmann::json &config) {
    std::string header_identifier = "ADDITIVENOISECACHE";
    int32_t header_version = 1;
    uint32_t header_entryCount = startIndexMap.size();

    const nlohmann::json& additiveNoiseSettings = config.at("filterSettings").at("additiveNoise");

    AdditiveNoiseFilterSettings filterSettings = readAdditiveNoiseFilterSettings(config, additiveNoiseSettings);

    std::filesystem::path cacheFilePath = std::filesystem::path(config.at("cacheDirectory")) / cacheFileName;
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
    for(const std::pair<const uint64_t, uint32_t>& mappedIndices : startIndexMap) {
        outputFile.write((const char*) &mappedIndices.first, sizeof(uint64_t));
        outputFile.write((const char*) &mappedIndices.second, sizeof(uint32_t));
    }

    // Write computed placements
    outputFile.write((const char*) objectOrientations.data(), objectOrientations.size() * sizeof(ShapeBench::Orientation));
}

bool ShapeBench::AdditiveNoiseCache::contains(uint64_t randomSeed) {
    return startIndexMap.contains(randomSeed);
}

void ShapeBench::AdditiveNoiseCache::set(uint64_t randomSeed, const std::vector<ShapeBench::Orientation> &orientations) {
    uint32_t startIndex = 0;
    if(!startIndexMap.contains(randomSeed)) {
        startIndex = objectOrientations.size();
    } else {
        startIndex = startIndexMap.at(randomSeed);
    }
    assert(orientations.size() == objectsPerEntry);

    for(uint32_t i = 0; i < orientations.size(); i++) {
        if(startIndex + i < objectOrientations.size()) {
            objectOrientations.at(startIndex + i) = orientations.at(i);
        } else {
            objectOrientations.push_back(orientations.at(i));
        }
    }
}

std::vector<ShapeBench::Orientation> ShapeBench::AdditiveNoiseCache::get(uint64_t randomSeed) {
    std::vector<ShapeBench::Orientation> orientations(objectsPerEntry);
    uint32_t startIndex = startIndexMap.at(randomSeed);
    for(uint32_t i = 0; i < objectsPerEntry; i++) {
        orientations.at(i) = objectOrientations.at(startIndex + i);
    }
    return orientations;
}
