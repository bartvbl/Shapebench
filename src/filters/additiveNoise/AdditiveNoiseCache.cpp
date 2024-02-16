

#include "AdditiveNoiseCache.h"
#include "additiveNoiseFilter.h"

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
