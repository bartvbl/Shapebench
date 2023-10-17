#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "benchmark-core/Dataset.h"
#include "json.hpp"


template<typename DescriptorMethod>
bool shouldRunOnGPU(const nlohmann::json& config, uint32_t descriptorCount) {
    bool cudaSupportAvailable = ShapeDescriptor::isCUDASupportAvailable();
    bool methodHasGPUImplementation = DescriptorMethod::hasGPUKernels();
    const uint32_t defaultThreshold = 0; // Always pick GPU if available
    bool specifiesGPUThreshold = config.contains("limits") && config.at("limits").contains("computeDescriptorsOnGPUThreshold");
    uint32_t gpuThreshold = specifiesGPUThreshold
            ? uint32_t(config.at("limits").at("computeDescriptorsOnGPUThreshold")) : defaultThreshold;
    bool gpuDescriptorCountReached = descriptorCount > gpuThreshold;
    return cudaSupportAvailable && gpuDescriptorCountReached && methodHasGPUImplementation;
}

template<typename DescriptorMethod, typename DescriptorType>
ShapeDescriptor::cpu::array<DescriptorType> computeDescriptorsAndSaveToCPU(const ShapeDescriptor::cpu::Mesh& mesh,
                                                                           ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> origins,
                                                                           const nlohmann::json& config,
                                                                           float supportRadius,
                                                                           uint64_t randomSeed) {
    ShapeDescriptor::cpu::array<DescriptorType> descriptors;
    bool runOnGPU = shouldRunOnGPU<DescriptorMethod>(config, origins.length);
    bool methodRequiresPointCloud = DescriptorMethod::usesPointCloudInput();

    ShapeDescriptor::cpu::PointCloud pointCloud;
    if(methodRequiresPointCloud) {
        pointCloud = ShapeDescriptor::sampleMesh(mesh, )
    }

    if(runOnGPU) {


    } else {

        descriptors = DescriptorMethod::computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
        const nlohmann::json& config,
        float supportRadius)
    }
    return descriptors;
}

template<typename DescriptorMethod, typename DescriptorType>
ShapeDescriptor::gpu::array<DescriptorType> computeDescriptorsAndSaveToGPU(const ShapeDescriptor::gpu::Mesh& mesh,
                                                                           ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> origins,
                                                                           const nlohmann::json& config,
                                                                           float supportRadius,
                                                                           uint64_t randomSeed) {

}