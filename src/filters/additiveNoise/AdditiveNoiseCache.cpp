

#include "AdditiveNoiseCache.h"

bool ShapeBench::AdditiveNoiseCache::contains(uint64_t randomSeed) {
    std::unique_lock<std::mutex> lock {cacheLock};
    return startIndexMap.contains(randomSeed);
}

void ShapeBench::AdditiveNoiseCache::set(uint64_t randomSeed, const std::vector<ShapeBench::Orientation> &orientations) {
    std::unique_lock<std::mutex> lock {cacheLock};
    uint32_t startIndex = 0;
    if(!startIndexMap.contains(randomSeed)) {
        startIndex = objectOrientations.size();
        startIndexMap.insert({randomSeed, startIndex});
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
    std::unique_lock<std::mutex> lock {cacheLock};
    std::vector<ShapeBench::Orientation> orientations(objectsPerEntry);
    uint32_t startIndex = startIndexMap.at(randomSeed);
    for(uint32_t i = 0; i < objectsPerEntry; i++) {
        orientations.at(i) = objectOrientations.at(startIndex + i);
    }
    return orientations;
}

long ShapeBench::AdditiveNoiseCache::entryCount() {
    std::unique_lock<std::mutex> lock {cacheLock};
    return startIndexMap.size();
}
