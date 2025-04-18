#pragma once
#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "nlohmann/json.hpp"
#include "filters/Filter.h"

namespace ShapeBench {
    namespace internal {
        void calculateAverageEdgeLength(const ShapeDescriptor::cpu::Mesh& mesh, double &averageEdgeLength, uint32_t &edgeIndex);
    }

    class AlternateTriangulationFilter : public ShapeBench::Filter {

    public:

        virtual void init(const nlohmann::json& config, bool invalidateCaches);
        virtual void destroy();
        virtual void saveCaches(const nlohmann::json& config);

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
