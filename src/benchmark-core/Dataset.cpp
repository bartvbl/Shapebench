#include "Dataset.h"
#include <fstream>
#include <random>
#include "json.hpp"

void Dataset::load(std::filesystem::path &cacheFile) {
    std::ifstream inputStream{cacheFile};
    nlohmann::json cacheFileContents = nlohmann::json::parse(inputStream);

    assert(cacheFileContents.contains("files"));
    uint32_t fileCount = cacheFileContents.at("files").size();
    entries.reserve(fileCount);

    for(uint32_t i = 0; i < fileCount; i++) {
        nlohmann::json jsonEntry = cacheFileContents.at("files").at(i);
        bool isPointCloud = jsonEntry.at("isPointCloud");
        bool isNotEmpty = jsonEntry.at("vertexCount") > 0;
        // Exclude all point clouds
        if(!isPointCloud && isNotEmpty) {
            DatasetEntry entry;
            entry.vertexCount = jsonEntry.at("vertexCount");
            entry.id = jsonEntry.at("id");
            entry.meshFile = std::string(jsonEntry.at("filePath"));
            entries.push_back(entry);
        }
    }
    std::sort(entries.begin(), entries.end());
}

std::vector<VertexInDataset> Dataset::sampleVertices(uint64_t randomSeed, uint32_t count) const {
    std::vector<uint32_t> sampleHistogram(entries.size());
    std::vector<VertexInDataset> sampledEntries(count);

    std::mt19937_64 engine(randomSeed);
    std::uniform_int_distribution<uint32_t> distribution(0, entries.size());

    for(uint32_t i = 0; i < count; i++) {
        uint32_t chosenMeshIndex = distribution(engine);
        sampleHistogram.at(chosenMeshIndex)++;
    }
    uint32_t nextIndex = 0;
    for(uint32_t i = 0; i < entries.size(); i++) {
        for(uint32_t j = 0; j < sampleHistogram.at(i); j++) {
            sampledEntries.at(nextIndex).meshFile = entries.at(i).meshFile;
            std::uniform_int_distribution<uint32_t> vertexIndexDistribution(0, entries.at(i).vertexCount);
            sampledEntries.at(nextIndex).vertexIndex = vertexIndexDistribution(engine);
            nextIndex++;
        }
    }
    assert(nextIndex == count);
    return sampledEntries;
}

bool DatasetEntry::operator<(DatasetEntry &other) {
    return id < other.id;
}
