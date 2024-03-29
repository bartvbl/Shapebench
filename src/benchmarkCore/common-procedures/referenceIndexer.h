#pragma once

#include <cstdint>
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    template<typename DescriptorMethod, typename DescriptorType>
    uint32_t computeImageIndex(const DescriptorType& cleanDescriptor, const DescriptorType& filteredDescriptor, ShapeDescriptor::cpu::array<DescriptorType> referenceSet) {
        float sampleDescriptorDistance = DescriptorMethod::computeDescriptorDistance(filteredDescriptor, cleanDescriptor);
        //std::cout << "Distance to beat: " << sampleDescriptorDistance << std::endl;
        uint32_t filteredDescriptorRank = 0;
        for(uint32_t i = 0; i < referenceSet.length; i++) {
            float referenceDescriptorDistance = DescriptorMethod::computeDescriptorDistance(filteredDescriptor, referenceSet[i]);
            if(referenceDescriptorDistance < sampleDescriptorDistance) {
                //std::cout << "Distance to beat: " + std::to_string(referenceDescriptorDistance) + ", beat distance: " + std::to_string(sampleDescriptorDistance) + "\n" << std::flush;
                filteredDescriptorRank++;
            }
        }
        return filteredDescriptorRank;
    }
}