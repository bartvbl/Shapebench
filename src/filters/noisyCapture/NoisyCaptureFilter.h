#pragma once

#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"
#include "filters/Filter.h"


namespace ShapeBench {
    class SimplificationFilter : public ShapeBench::Filter {
        void init(const nlohmann::json& config) override;
        void destroy() override;
        void saveCaches() override;

        FilterOutput apply(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed) override;
    };
}
