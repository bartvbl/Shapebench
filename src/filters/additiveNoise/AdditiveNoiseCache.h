#pragma once

#include <unordered_map>
#include <cstdint>
#include <vector>
#include "utils/Orientation.h"
#include "json.hpp"

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
        void load(const nlohmann::json& config);
        void save(const nlohmann::json& config);
    };
}

