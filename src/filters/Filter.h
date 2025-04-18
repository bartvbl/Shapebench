#pragma once

#include "nlohmann/json.hpp"
#include "FilteredMeshPair.h"
#include "dataset/Dataset.h"

namespace ShapeBench {
    struct FilterOutput {
        nlohmann::json metadata;
        nlohmann::json metadata_filterAppliedToBoth;
    };

    class Filter {

    public:
        virtual void init(const nlohmann::json& config, bool invalidateCaches) = 0;
        virtual void destroy() = 0;
        virtual void saveCaches(const nlohmann::json& config) = 0;

        virtual FilterOutput apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                   const Dataset &dataset, ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) = 0;
        // Only applies to filters that return true for mustBeAppliedOnBothMeshes()
        virtual FilterOutput applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                         ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                                         ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) = 0;

        virtual constexpr bool mustBeAppliedOnBothMeshes() = 0;
        virtual constexpr bool appliesNonrigidTransformation() = 0;
        virtual const std::string getFilterName() const = 0;

        virtual ~Filter() = default;
    };
}
