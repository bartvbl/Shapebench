#pragma once

#include <filesystem>
#include <vector>

struct VertexInDataset {
    std::filesystem::path meshFile;
    uint32_t vertexIndex = 0;
};

struct DatasetEntry {
    std::filesystem::path meshFile;
    uint32_t vertexCount = 0;
    uint32_t id = 0xFFFFFFFF;

    bool operator<(DatasetEntry& other);
};

class Dataset {
    std::vector<DatasetEntry> entries;
public:
    void load(const std::filesystem::path &cacheFile);
    std::vector<VertexInDataset> sampleVertices(uint64_t randomSeed, uint32_t count) const;
};