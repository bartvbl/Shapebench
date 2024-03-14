#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "benchmarkCore/Dataset.h"
#include "json.hpp"
#include "benchmarkCore/Batch.h"
#include "supportRadiusEstimation/SupportRadiusEstimation.h"
#include "pointCloudSampler.h"

namespace ShapeBench {
    template<typename DescriptorMethod, typename DescriptorType>
    DescriptorType computeSingleDescriptor(const ShapeDescriptor::cpu::Mesh& mesh,
                                 const ShapeDescriptor::cpu::PointCloud& pointCloud,
                                 ShapeDescriptor::OrientedPoint descriptorOrigin,
                                 const nlohmann::json &config,
                                 float supportRadius,
                                 uint64_t randomSeed) {
        ShapeDescriptor::cpu::array<DescriptorType> descriptors;
        if (DescriptorMethod::usesPointCloudInput()) {
            descriptors = DescriptorMethod::computeDescriptors(pointCloud, {1, &descriptorOrigin}, config, supportRadius, randomSeed);
        } else {
            descriptors = DescriptorMethod::computeDescriptors(mesh, {1, &descriptorOrigin}, config, supportRadius, randomSeed);
        }

        DescriptorType descriptor = descriptors.content[0];

        ShapeDescriptor::free(descriptors);

        return descriptor;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    DescriptorType computeSingleDescriptor(const ShapeDescriptor::cpu::Mesh& mesh,
                                           ShapeDescriptor::OrientedPoint descriptorOrigin,
                                           const nlohmann::json &config,
                                           float supportRadius,
                                           uint64_t pointCloudSamplingSeed,
                                           uint64_t descriptorRandomSeed) {
        ShapeDescriptor::cpu::PointCloud pointCloud;
        if (DescriptorMethod::usesPointCloudInput()) {
            pointCloud = computePointCloud(mesh, config, pointCloudSamplingSeed);
        }

        DescriptorType descriptor = computeSingleDescriptor<DescriptorMethod, DescriptorType>(mesh, pointCloud, descriptorOrigin, config, supportRadius, descriptorRandomSeed);

        if(DescriptorMethod::usesPointCloudInput()) {
            ShapeDescriptor::free(pointCloud);
        }

        return descriptor;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    void computeDescriptorsForEachSupportRadii(
            VertexInDataset vertexToRender,
            const ShapeDescriptor::cpu::Mesh& mesh,
            const ShapeDescriptor::cpu::PointCloud& pointCloud,
            const nlohmann::json &config,
            uint64_t randomSeed,
            const std::vector<float>& supportRadii,
            std::vector<DescriptorType>& outputDescriptors) {
        assert(supportRadii.size() == outputDescriptors.size());

        for (uint32_t radiusIndex = 0; radiusIndex < supportRadii.size(); radiusIndex++) {
            uint32_t vertexIndex = vertexToRender.vertexIndex;
            ShapeDescriptor::OrientedPoint originPoint = {mesh.vertices[vertexIndex], mesh.normals[vertexIndex]};
            outputDescriptors.at(radiusIndex) = computeSingleDescriptor<DescriptorMethod, DescriptorType>(mesh, pointCloud, originPoint, config, supportRadii.at(radiusIndex), randomSeed);
        }
    }
}