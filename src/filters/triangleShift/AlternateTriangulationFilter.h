#pragma once
#include <shapeDescriptor/containerTypes.h>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"
#include "filters/Filter.h"

namespace ShapeBench {
    namespace internal {
        void calculateAverageEdgeLength(const ShapeDescriptor::cpu::Mesh& mesh, double &averageEdgeLength, uint32_t &edgeIndex);
    }

    class TriangleShiftFilter : public ShapeBench::Filter {
        RemeshingFilterOutput remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config);

        virtual void init(const nlohmann::json& config);
        virtual void destroy();
        virtual void saveCaches();

        virtual FilterOutput apply(const nlohmann::json& config, ShapeBench::FilteredMeshPair& scene, const Dataset& dataset, uint64_t randomSeed);

    };

}
