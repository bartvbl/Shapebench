#pragma once

#include <string>
#include <memory>
#include <vector>
#include "nlohmann/json.hpp"
#include "filters/FilteredMeshPair.h"
#include "benchmarkCore/randomEngine.h"
#include "results/ExperimentResult.h"
#include "filters/Filter.h"

namespace ShapeBench {
    void applyFilterSequence(
            ShapeBench::LocalDatasetCache *fileCache,
            const std::string modelFilterSequenceJSONEntryLabel,
            const std::string sceneFilterSequenceJSONEntryLabel,
            const nlohmann::json &configuration,
            const ShapeBench::Dataset &dataset,
            const std::unordered_map<std::string, std::unique_ptr<ShapeBench::Filter>> &filterInstanceMap,
            uint32_t verticesPerSampleObject,
            const nlohmann::json &experimentConfig,
            std::vector<ShapeBench::ExperimentResultsEntry>& resultsEntries,
            ShapeBench::randomEngine &experimentInstanceRandomEngine,
            ShapeBench::FilteredMeshPair &filteredModelObject,
            ShapeBench::FilteredMeshPair &filteredSceneObject);
}