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

    template<typename DescriptorMethod, typename DescriptorType>
    ShapeBench::PRCInfo computePRCInfo(const DescriptorType& filteredDescriptor,
                                       ShapeDescriptor::cpu::array<DescriptorType> referenceSet,
                                       uint32_t sampleMeshID,
                                       ShapeDescriptor::cpu::float3 filteredSampleSurfacePoint,
                                       const std::vector<ShapeBench::ChosenVertexPRC>& representativeSetPRC) {
        ShapeBench::PRCInfo outputMetadata;
        /*
         * For the ShapeBench approach: use sample descriptors from 1000 objects, 1000 points each. Reference set is the same as other experiments
         * For the PRC approach: use same sample descriptors, but reference set is the same 1000 sample objects but different surface points
         *
         */

        // Compute nearest and second nearest neighbours
        float nearestNeighbourDistance = std::numeric_limits<float>::max();
        float secondNearestNeighbourDistance = std::numeric_limits<float>::max();
        uint32_t nearestNeighbourVertexIndex = 0xFFFFFFFF;
        uint32_t secondNearestNeighbourVertexIndex = 0xFFFFFFFF;
        for(uint32_t i = 0; i < referenceSet.length; i++) {
            float referenceDescriptorDistance = DescriptorMethod::computeDescriptorDistance(filteredDescriptor, referenceSet[i]);
            if(referenceDescriptorDistance < nearestNeighbourDistance) {
                secondNearestNeighbourDistance = nearestNeighbourDistance;
                secondNearestNeighbourVertexIndex = nearestNeighbourVertexIndex;
                nearestNeighbourDistance = referenceDescriptorDistance;
                nearestNeighbourVertexIndex = i;
            }
        }

        // Determine mesh ID of nearest neighbour and filtered descriptor
        outputMetadata.scenePointMeshID = sampleMeshID;
        outputMetadata.modelPointMeshID = representativeSetPRC.at(nearestNeighbourVertexIndex).meshID;

        // Determine coordinates of nearest neighbour and filtered descriptor
        outputMetadata.nearestNeighbourVertexScene = filteredSampleSurfacePoint;
        outputMetadata.nearestNeighbourVertexModel = representativeSetPRC.at(nearestNeighbourVertexIndex).vertex.vertex;

        return outputMetadata;
    }
}