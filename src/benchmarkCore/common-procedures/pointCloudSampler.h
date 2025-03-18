#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include "nlohmann/json.hpp"
#include "methods/Method.h"

namespace ShapeBench {
    inline uint32_t computeSampleCount(double totalMeshArea, float pointCountScaleFactor, const nlohmann::json& config) {
        double sampleDensity = config.at("commonExperimentSettings").at("pointSampleDensity");
        uint32_t minPointSampleCount = config.at("limits").at("minPointSampleCount");
        uint32_t maxPointSampleCount = config.at("limits").at("maxPointSampleCount");
        uint32_t sampleCount = uint32_t(totalMeshArea * pointCountScaleFactor * sampleDensity);
        sampleCount = std::max(sampleCount, minPointSampleCount);
        sampleCount = std::min(sampleCount, maxPointSampleCount);
        return sampleCount;
    }

    struct AreaEstimateSampleCounts {
        uint64_t originalMesh = 0;
        uint64_t filteredOriginalMesh = 0;
        uint64_t filteredAdditiveMesh = 0;
    };

    inline AreaEstimateSampleCounts computeAreaEstimateSampleCounts(const nlohmann::json& config, double areaOriginalMesh, double areaFilteredOriginalMesh, double areaFilteredAdditiveMesh) {
        double maxArea = std::max(areaOriginalMesh, areaFilteredOriginalMesh + areaFilteredAdditiveMesh);

        uint32_t baselineSampleCountMax = computeSampleCount(maxArea, 1.0f, config);

        // We ignore that in case of excessive additional geometry the original mesh can receive very little sample points
        // It's one or the other, really

        AreaEstimateSampleCounts sampleCounts;
        sampleCounts.originalMesh = uint32_t((areaOriginalMesh / maxArea) * double(baselineSampleCountMax));
        sampleCounts.filteredOriginalMesh = uint32_t((areaFilteredOriginalMesh / maxArea) * double(baselineSampleCountMax));
        sampleCounts.filteredAdditiveMesh = uint32_t((areaFilteredAdditiveMesh / maxArea) * double(baselineSampleCountMax));

        return sampleCounts;
    }

    template<typename DescriptorMethod>
    inline ShapeDescriptor::cpu::PointCloud computePointCloud(const ShapeDescriptor::cpu::Mesh& mesh, const nlohmann::json& config, float pointCountScaleFactor, uint32_t randomSeed) {

        double totalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(mesh);
        uint32_t sampleCount = computeSampleCount(totalMeshArea, pointCountScaleFactor, config);

        // Some methods are too expensive to compute at full resolution, so this is an override to still be able to include their results
        if(ShapeBench::hasConfigValue(config, DescriptorMethod::getName(), "pointDensityScaleFactor")) {
            double scaleFactor = ShapeBench::readDescriptorConfigValue<double>(config, DescriptorMethod::getName(), "pointDensityScaleFactor");
            sampleCount = uint32_t(scaleFactor * double(sampleCount));
        }

        return ShapeDescriptor::sampleMesh(mesh, sampleCount, randomSeed);
    }
}