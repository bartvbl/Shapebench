#pragma once

#include "json.hpp"
#include "FilteredMeshPair.h"
#include "dataset/Dataset.h"

namespace ShapeBench {
    struct FilterOutput {
        nlohmann::json metadata;
    };

    class Filter {

    public:
        virtual void init(const nlohmann::json& config) = 0;
        virtual void destroy() = 0;
        virtual void saveCaches(const nlohmann::json& config) = 0;

        virtual FilterOutput apply(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed) = 0;
        virtual ~Filter() {}
    };
}
