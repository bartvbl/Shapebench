#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include <json.hpp>

namespace Shapebench {
    inline ShapeDescriptor::cpu::PointCloud computePointCloud(const ShapeDescriptor::cpu::Mesh& mesh, const nlohmann::json& config, uint32_t randomSeed) {
        double sampleDensity = config.at("pointSampleDensity");
        uint32_t minPointSampleCount = config.at("minPointSampleCount");
        double totalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(mesh);
        uint32_t sampleCount = uint32_t(totalMeshArea * sampleDensity);
        sampleCount = std::max(sampleCount, minPointSampleCount);
        return ShapeDescriptor::sampleMesh(mesh, sampleCount, randomSeed);

    }

    inline void computePointClouds(std::vector<ShapeDescriptor::cpu::Mesh> &meshes,
                                   std::vector<ShapeDescriptor::cpu::PointCloud> &clouds,
                                   const nlohmann::json& config,
                                   uint64_t randomSeed) {
        clouds.resize(meshes.size());
        #pragma omp parallel for schedule(dynamic) default(none) shared(meshes, config, clouds, randomSeed)
        for(uint32_t i = 0; i < meshes.size(); i++) {
            clouds.at(i) = computePointCloud(meshes.at(i), config, randomSeed);
        }
    }
}