#include <cassert>
#include <vector>
#include <random>
#include <algorithm>
#include "RandomSubset.h"
#include "benchmarkCore/randomEngine.h"

ShapeBench::RandomSubset::RandomSubset(uint32_t start, uint32_t end, uint32_t count, uint64_t randomSeed) {
    assert(end > start);
    uint32_t sequenceLength = end - start;
    assert(sequenceLength >= count);
    if(sequenceLength == count) {
        containsAll = true;
    }
    std::vector<uint32_t> sequence(sequenceLength);
    for(uint32_t i = start; i < end; i++) {
        sequence.at(i - start) = i;
    }
    ShapeBench::randomEngine engine(randomSeed);
    std::shuffle(sequence.begin(), sequence.end(), engine);
    sequence.resize(count);
    subset.reserve(count);
    for(uint32_t i : sequence) {
        subset.insert(i);
    }
}

bool ShapeBench::RandomSubset::contains(uint32_t index) {
    if(containsAll) {
        return true;
    }
    return subset.contains(index);
}
