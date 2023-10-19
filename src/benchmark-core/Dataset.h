#pragma once

#include <filesystem>
#include <vector>

struct VertexInDataset {
    uint32_t meshID = 0;
    uint32_t vertexIndex = 0;
};

struct DatasetEntry {
    std::filesystem::path meshFile;
    uint32_t vertexCount = 0;
    uint32_t id = 0xFFFFFFFF;
    float computedObjectRadius = 0;

    bool operator<(DatasetEntry& other);
};

class Dataset {
    std::vector<DatasetEntry> entries;
public:
    void load(const std::filesystem::path &cacheFile);
    std::vector<VertexInDataset> sampleVertices(uint64_t randomSeed, uint32_t count) const;
    const DatasetEntry& at(uint32_t meshID) const;
};