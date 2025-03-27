#pragma once

#include <filesystem>
#include <vector>
#include <shapeDescriptor/shapeDescriptor.h>
#include "nlohmann/json.hpp"
#include "LocalDatasetCache.h"

namespace ShapeBench {
    struct VertexInDataset {
        uint32_t meshID = 0;
        uint32_t vertexIndex = 0;
    };

    template<typename DescriptorType>
    struct DescriptorOfVertexInDataset {
        uint32_t meshID = 0;
        uint32_t vertexIndex = 0;
        DescriptorType descriptor;
        ShapeDescriptor::OrientedPoint vertex = {{0, 0, 0}, {0, 0, 0}};
    };

    struct DatasetEntry {
        std::filesystem::path meshFile;
        uint32_t vertexCount = 0;
        uint32_t id = 0xFFFFFFFF;
        double computedObjectRadius = 0;
        std::string meshIntegrityDigest = "NOT_SPECIFIED";
        std::string uncompressedMeshFileIntegrityDigest = "NOT_SPECIFIED";
        std::array<double, 3> computedObjectCentre = {0, 0, 0};

        bool operator<(DatasetEntry& other);
    };

    class Dataset {
        std::vector<DatasetEntry> entries;
    public:
        std::vector<VertexInDataset> sampleVertices(uint64_t randomSeed, uint32_t count, uint32_t verticesPerObject) const;
        const DatasetEntry& at(uint32_t meshID) const;
        void loadCache(const nlohmann::json& cacheJson);
    };
}
