#include "Dataset.h"
#include <fstream>
#include <random>
#include <iostream>
#include "json.hpp"

void Dataset::load(const std::filesystem::path &cacheFile) {
    std::ifstream inputStream{cacheFile};
    nlohmann::json cacheFileContents = nlohmann::json::parse(inputStream);

    assert(cacheFileContents.contains("files"));
    uint32_t fileCount = cacheFileContents.at("files").size();
    entries.reserve(fileCount);

    uint32_t excludedCount = 0;
    for(uint32_t i = 0; i < fileCount; i++) {
        nlohmann::json jsonEntry = cacheFileContents.at("files").at(i);
        bool isPointCloud = jsonEntry.at("isPointCloud");
        bool isNotEmpty = jsonEntry.at("vertexCount") > 0;
        bool parseFailed = jsonEntry.contains("parseFailed") && jsonEntry.at("parseFailed");
        // Exclude all point clouds, empty meshes, and meshes that failed to parse
        if(!isPointCloud && isNotEmpty && !parseFailed) {
            DatasetEntry entry;
            entry.vertexCount = jsonEntry.at("vertexCount");
            entry.id = jsonEntry.at("id");
            entry.computedObjectRadius = jsonEntry.at("boundingSphereRadius");
            entry.meshFile = std::string(jsonEntry.at("filePath"));
            entries.push_back(entry);
        } else {
            excludedCount++;
        }
    }
    std::cout << "    Dataset contains " << entries.size() << " meshes." << std::endl;
    std::cout << "    Excluded " << excludedCount << " meshes and/or point clouds." << std::endl;
    std::sort(entries.begin(), entries.end());
}

std::vector<VertexInDataset> Dataset::sampleVertices(uint64_t randomSeed, uint32_t count) const {
    std::vector<uint32_t> sampleHistogram(entries.size());
    std::vector<VertexInDataset> sampledEntries(count);

    std::mt19937_64 engine(randomSeed);
    std::uniform_int_distribution<uint32_t> distribution(0, entries.size()-1);

    for(uint32_t i = 0; i < count; i++) {
        uint32_t chosenMeshIndex = distribution(engine);
        sampleHistogram.at(chosenMeshIndex)++;
    }
    uint32_t nextIndex = 0;
    for(uint32_t i = 0; i < entries.size(); i++) {
        for(uint32_t j = 0; j < sampleHistogram.at(i); j++) {
            sampledEntries.at(nextIndex).meshID = i;
            std::uniform_int_distribution<uint32_t> vertexIndexDistribution(0, entries.at(i).vertexCount);
            sampledEntries.at(nextIndex).vertexIndex = vertexIndexDistribution(engine);
            nextIndex++;
        }
    }
    assert(nextIndex == count);
    return sampledEntries;
}

const DatasetEntry &Dataset::at(uint32_t meshID) const {
    return entries.at(meshID);
}

bool DatasetEntry::operator<(DatasetEntry &other) {
    return id < other.id;
}
