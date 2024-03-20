#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "benchmarkCore/Dataset.h"
#include "json.hpp"
#include "benchmarkCore/Batch.h"
#include "supportRadiusEstimation/SupportRadiusEstimation.h"
#include "pointCloudSampler.h"

namespace ShapeBench {
    template<typename DescriptorMethod, typename DescriptorType>
    DescriptorType computeDescriptors(
            const ShapeDescriptor::cpu::Mesh& mesh,
            const ShapeDescriptor::cpu::PointCloud& pointCloud,
            const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
            const nlohmann::json &config,
            const std::vector<float>& supportRadii,
            uint64_t randomSeed,
            std::vector<DescriptorType>& outputDescriptors) {
        ShapeDescriptor::cpu::array<DescriptorType> descriptors;
        if (DescriptorMethod::usesPointCloudInput()) {
            if(DescriptorMethod::shouldUseGPUKernel()) {
                ShapeDescriptor::gpu::PointCloud gpuCloud = ShapeDescriptor::copyToGPU(pointCloud);
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins = ShapeDescriptor::copyToGPU(descriptorOrigins);
                ShapeDescriptor::gpu::array<DescriptorType> gpuDescriptors = DescriptorMethod::computeDescriptors(gpuCloud, gpuOrigins, config, supportRadii, randomSeed);
                descriptors = ShapeDescriptor::copyToCPU(gpuDescriptors);
                ShapeDescriptor::free(gpuDescriptors);
                ShapeDescriptor::free(gpuOrigins);
                ShapeDescriptor::free(gpuCloud);
            } else {
                descriptors = DescriptorMethod::computeDescriptors(pointCloud, descriptorOrigins, config, supportRadii, randomSeed);
            }
        } else {
            if(DescriptorMethod::shouldUseGPUKernel()) {
                ShapeDescriptor::gpu::Mesh gpuMesh = ShapeDescriptor::copyToGPU(mesh);
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins = ShapeDescriptor::copyToGPU(descriptorOrigins);
                ShapeDescriptor::gpu::array<DescriptorType> gpuDescriptors = DescriptorMethod::computeDescriptors(gpuMesh, gpuOrigins, config, supportRadii, randomSeed);
                descriptors = ShapeDescriptor::copyToCPU(gpuDescriptors);
                ShapeDescriptor::free(gpuDescriptors);
                ShapeDescriptor::free(gpuOrigins);
                ShapeDescriptor::free(gpuMesh);
            } else {
                descriptors = DescriptorMethod::computeDescriptors(mesh, descriptorOrigins, config, supportRadii, randomSeed);
            }
        }

        DescriptorType descriptor = descriptors.content[0];

        ShapeDescriptor::free(descriptors);

        return descriptor;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    void computeDescriptors(
            const ShapeDescriptor::cpu::Mesh& mesh,
            const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
            const nlohmann::json &config,
            const std::vector<float>& supportRadii,
            uint64_t pointCloudSamplingSeed,
            uint64_t descriptorRandomSeed,
            std::vector<DescriptorType>& outputDescriptors) {

        ShapeDescriptor::cpu::PointCloud pointCloud;
        if (DescriptorMethod::usesPointCloudInput()) {
            pointCloud = computePointCloud<DescriptorMethod>(mesh, config, pointCloudSamplingSeed);
        }

        computeDescriptors<DescriptorMethod, DescriptorType>(mesh, pointCloud, descriptorOrigins, config, supportRadii, descriptorRandomSeed, outputDescriptors);

        if(DescriptorMethod::usesPointCloudInput()) {
            ShapeDescriptor::free(pointCloud);
        }
    }
}