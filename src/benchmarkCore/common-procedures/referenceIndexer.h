#pragma once

#include <cstdint>
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    template<typename DescriptorMethod, typename DescriptorType>
    uint32_t computeImageIndex(const ShapeBench::DescriptorOfVertexInDataset<DescriptorType>& cleanDescriptor, const DescriptorType& filteredDescriptor, const std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>& referenceSet) {
        float sampleDescriptorDistance = DescriptorMethod::computeDescriptorDistance(filteredDescriptor, cleanDescriptor.descriptor);
        //std::cout << "Distance to beat: " << sampleDescriptorDistance << std::endl;
        uint32_t filteredDescriptorRank = 0;
        for(uint32_t i = 0; i < referenceSet.size(); i++) {
            float referenceDescriptorDistance = DescriptorMethod::computeDescriptorDistance(filteredDescriptor, referenceSet.at(i).descriptor);
            if(referenceDescriptorDistance < sampleDescriptorDistance) {
                //std::cout << "Distance to beat: " + std::to_string(referenceDescriptorDistance) + ", beat distance: " + std::to_string(sampleDescriptorDistance) + "\n" << std::flush;
                filteredDescriptorRank++;
            }
        }
        return filteredDescriptorRank;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    ShapeBench::PRCInfo computePRCInfo(const ShapeBench::DescriptorOfVertexInDataset<DescriptorType>& filteredDescriptor,
                                       const ShapeBench::DescriptorOfVertexInDataset<DescriptorType>& cleanDescriptor,
                                       const std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>& referenceSet,
                                       const std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>& sampleDescriptors,
                                       bool useReferenceDescriptors) {
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

        const std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>& setToProcess = useReferenceDescriptors ? referenceSet : sampleDescriptors;

        // "add" the correct clean descriptor to the list such that it can be found
        int extraIteration = useReferenceDescriptors ? 1 : 0;

        for(uint32_t i = 0; i < setToProcess.size() + extraIteration; i++) {
            const DescriptorType& descriptor = i < setToProcess.size() ? setToProcess.at(i).descriptor : cleanDescriptor.descriptor;
            float referenceDescriptorDistance = DescriptorMethod::computeDescriptorDistance(filteredDescriptor.descriptor, descriptor);
            if(referenceDescriptorDistance < nearestNeighbourDistance) {
                secondNearestNeighbourDistance = nearestNeighbourDistance;
                nearestNeighbourDistance = referenceDescriptorDistance;
                nearestNeighbourVertexIndex = i;
            }
        }

        outputMetadata.distanceToNearestNeighbour = nearestNeighbourDistance;
        outputMetadata.distanceToSecondNearestNeighbour = secondNearestNeighbourDistance;

        // Determine mesh ID of nearest neighbour and filtered descriptor
        outputMetadata.scenePointMeshID = filteredDescriptor.meshID;
        outputMetadata.modelPointMeshID = nearestNeighbourVertexIndex != setToProcess.size() ? referenceSet.at(nearestNeighbourVertexIndex).meshID : cleanDescriptor.meshID;

        // Determine coordinates of nearest neighbour and filtered descriptor
        outputMetadata.nearestNeighbourVertexScene = filteredDescriptor.vertex.vertex;
        outputMetadata.nearestNeighbourVertexModel = nearestNeighbourVertexIndex != setToProcess.size() ? referenceSet.at(nearestNeighbourVertexIndex).vertex.vertex : cleanDescriptor.vertex.vertex;

        return outputMetadata;
    }
}