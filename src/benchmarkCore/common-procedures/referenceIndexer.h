#pragma once

#include <cstdint>
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {

    template<typename DescriptorMethod, typename DescriptorType>
    MatchingPerformanceMeasure computeAchievedMatchingPerformance(const ShapeBench::DescriptorOfVertexInDataset<DescriptorType>& modelDescriptor,
                                                                  const ShapeBench::DescriptorOfVertexInDataset<DescriptorType>& sceneDescriptor,
                                                                  const std::vector<ShapeBench::DescriptorOfVertexInDataset<DescriptorType>>& referenceSet,
                                                                  bool enablePRC) {


        // Used for computing the DDI
        float sampleDescriptorDistance = DescriptorMethod::computeDescriptorDistance(sceneDescriptor.descriptor, modelDescriptor.descriptor, std::numeric_limits<float>::max());
        uint32_t filteredDescriptorRank = 0;

        // Used for computing the PRC
        // Compute nearest and second nearest neighbours
        float nearestNeighbourDistance = enablePRC ? std::numeric_limits<float>::max() : 0;
        float secondNearestNeighbourDistance = std::numeric_limits<float>::max();
        uint32_t nearestNeighbourVertexIndex = 0xFFFFFFFF;

        // "add" the correct clean descriptor to the list such that it can be found
        int extraIteration = 1;

        for(uint32_t i = 0; i < referenceSet.size() + extraIteration; i++) {
            float earlyExitThreshold = std::max(sampleDescriptorDistance, nearestNeighbourDistance);

            const DescriptorType& descriptor = i < referenceSet.size() ? referenceSet.at(i).descriptor : modelDescriptor.descriptor;
            float referenceDescriptorDistance = DescriptorMethod::computeDescriptorDistance(sceneDescriptor.descriptor, descriptor, earlyExitThreshold);

            // DDI
            if(referenceDescriptorDistance < sampleDescriptorDistance && i < referenceSet.size()) {
                filteredDescriptorRank++;
            }

            // PRC
            if(enablePRC && referenceDescriptorDistance < nearestNeighbourDistance) {
                secondNearestNeighbourDistance = nearestNeighbourDistance;
                nearestNeighbourDistance = referenceDescriptorDistance;
                nearestNeighbourVertexIndex = i;
            }
        }

        MatchingPerformanceMeasure measure;
        ShapeBench::PRCMeasure outputMetadata;

        if(enablePRC) {
            outputMetadata.distanceToNearestNeighbour = nearestNeighbourDistance;
            outputMetadata.distanceToSecondNearestNeighbour = secondNearestNeighbourDistance;

            // Determine mesh ID of nearest neighbour and filtered descriptor
            outputMetadata.scenePointMeshID = sceneDescriptor.meshID;
            outputMetadata.modelPointMeshID = nearestNeighbourVertexIndex != referenceSet.size() ? referenceSet.at(nearestNeighbourVertexIndex).meshID : modelDescriptor.meshID;

            // Determine coordinates of nearest neighbour and filtered descriptor
            outputMetadata.nearestNeighbourVertexScene = sceneDescriptor.vertex.vertex;
            outputMetadata.nearestNeighbourVertexModel = nearestNeighbourVertexIndex != referenceSet.size() ? referenceSet.at(nearestNeighbourVertexIndex).vertex.vertex : modelDescriptor.vertex.vertex;

            measure.prcMeasure = outputMetadata;
        }

        measure.descriptorDistanceIndex = filteredDescriptorRank;

        return measure;
    }
}