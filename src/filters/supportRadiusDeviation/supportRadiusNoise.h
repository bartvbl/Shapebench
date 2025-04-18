#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "nlohmann/json.hpp"
#include "filters/Filter.h"


namespace ShapeBench {
    struct SupportRadiusNoiseFilter : public ShapeBench::Filter {

    public:
        void init(const nlohmann::json& config, bool invalidateCaches) override;
        void destroy() override;
        void saveCaches(const nlohmann::json& config) override;

        FilterOutput apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                           ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) override;

        FilterOutput applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                           ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                           ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) override;

        constexpr bool mustBeAppliedOnBothMeshes() override {
            return false;
        }
        constexpr bool appliesNonrigidTransformation() override {
            return true;
        }
        const std::string getFilterName() const override;
    };

}
