#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"

namespace ShapeBench {
    inline uint32_t computeSampleCount(double totalMeshArea, const nlohmann::json& config) {
        double sampleDensity = config.at("commonExperimentSettings").at("pointSampleDensity");
        uint32_t minPointSampleCount = config.at("limits").at("minPointSampleCount");
        uint32_t maxPointSampleCount = config.at("limits").at("maxPointSampleCount");
        uint32_t sampleCount = uint32_t(totalMeshArea * sampleDensity);
        sampleCount = std::max(sampleCount, minPointSampleCount);
        sampleCount = std::min(sampleCount, maxPointSampleCount);
        return sampleCount;
    }

    inline ShapeDescriptor::cpu::PointCloud computePointCloud(const ShapeDescriptor::cpu::Mesh& mesh, const nlohmann::json& config, uint32_t randomSeed) {
        double totalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(mesh);
        uint32_t sampleCount = computeSampleCount(totalMeshArea, config);
        return ShapeDescriptor::sampleMesh(mesh, sampleCount, randomSeed);
    }
}