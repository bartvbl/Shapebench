#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"

namespace Shapebench {


    template<typename DescriptorMethod, typename DescriptorType, int supportRadiusCount>
    std::array<ShapeDescriptor::gpu::array<DescriptorType>, supportRadiusCount> computeReferenceDescriptors(
            const std::vector<VertexInDataset> &verticesToRender,
            const nlohmann::json &config,
            std::array<float, supportRadiusCount> supportRadius,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {

    }

    template<typename DescriptorMethod, typename DescriptorType>
    ShapeDescriptor::gpu::array<DescriptorType> computeReferenceDescriptors(
            const std::vector<VertexInDataset> &verticesToRender,
            const nlohmann::json &config,
            float supportRadius,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {
        return computeReferenceDescriptors<DescriptorMethod, DescriptorType, 1>(verticesToRender, config, {supportRadius}, startIndex, endIndex).at(0);
    }
}