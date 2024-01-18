#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "DescriptorGenerator.h"
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"
#include "support-radius-estimation/SupportRadiusEstimation.h"
#include "PointCloudSampler.h"

namespace Shapebench {
    template<typename DescriptorMethod, typename DescriptorType>
    void computeDescriptorsForEachSupportRadii(
            VertexInDataset vertexToRender,
            const ShapeDescriptor::cpu::Mesh& mesh,
            const ShapeDescriptor::cpu::PointCloud& pointCloud,
            const nlohmann::json &config,
            uint32_t randomSeed,
            const std::vector<float>& supportRadii,
            std::vector<DescriptorType>& outputDescriptors) {
        assert(supportRadii.size() == outputDescriptors.size());

        for (uint32_t radiusIndex = 0; radiusIndex < supportRadii.size(); radiusIndex++) {
            uint32_t vertexIndex = vertexToRender.vertexIndex;
            ShapeDescriptor::OrientedPoint originPoint = {mesh.vertices[vertexIndex], mesh.normals[vertexIndex]};

            ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> convertedOriginArray = {1, &originPoint};

            ShapeDescriptor::cpu::array<DescriptorType> descriptors;
            if (DescriptorMethod::usesPointCloudInput()) {
                descriptors = DescriptorMethod::computeDescriptors(pointCloud, convertedOriginArray, config, supportRadii.at(radiusIndex));
            } else {
                descriptors = DescriptorMethod::computeDescriptors(mesh, convertedOriginArray, config, supportRadii.at(radiusIndex));
            }

            outputDescriptors.at(radiusIndex) = descriptors.content[0];

            ShapeDescriptor::free(descriptors);
        }
    }
}