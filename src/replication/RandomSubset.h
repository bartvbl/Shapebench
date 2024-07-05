#pragma once

#include <cstdint>
#include <unordered_set>

namespace ShapeBench {
    class RandomSubset {
        std::unordered_set<uint32_t> subset;
        bool containsAll = false;
    public:
        RandomSubset(uint32_t start, uint32_t end, uint32_t count, uint64_t randomSeed);
        RandomSubset() = default;
        bool contains(uint32_t index);
    };
}
