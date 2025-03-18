#pragma once

#include "methods/Method.h"

namespace ShapeBench {
    template<typename DescriptorType, typename contentsType>
    inline float computeEuclideanDistance(const DescriptorType& descriptor1, const DescriptorType& descriptor2, float earlyExitThreshold) {
        constexpr uint32_t elementsInDescriptor = sizeof(DescriptorType) / sizeof(contentsType);
        contentsType combinedSquaredDistance = 0;
        float distanceThreshold = earlyExitThreshold * earlyExitThreshold;
        for (uint32_t binIndex = 0; binIndex < elementsInDescriptor; binIndex++) {
            contentsType needleBinValue = descriptor1.contents[binIndex];
            contentsType haystackBinValue = descriptor2.contents[binIndex];
            contentsType binDelta = needleBinValue - haystackBinValue;
            combinedSquaredDistance += binDelta * binDelta;
            if(combinedSquaredDistance > distanceThreshold) {
                break;
            }
        }

        if(combinedSquaredDistance == 0) {
            return 0;
        }

        float distance = std::sqrt(combinedSquaredDistance);
        if(std::isnan(distance)) {
            throw std::runtime_error("Found a NaN!");
        }

        return distance;
    }

    template<typename DescriptorType, typename contentsType>
    __device__ __inline__ float computeEuclideanDistanceGPU(const DescriptorType& descriptor1, const DescriptorType& descriptor2, float earlyExitThreshold) {
        constexpr uint32_t elementsInDescriptor = sizeof(DescriptorType) / sizeof(contentsType);

        contentsType threadSquaredDistance = 0;
        for (short binIndex = threadIdx.x; binIndex < elementsInDescriptor; binIndex += blockDim.x) {
            contentsType needleBinValue = descriptor1.contents[binIndex];
            contentsType haystackBinValue = descriptor2.contents[binIndex];
            contentsType binDelta = needleBinValue - haystackBinValue;
            threadSquaredDistance += binDelta * binDelta;
        }

        contentsType combinedSquaredDistance = ShapeDescriptor::warpAllReduceSum(threadSquaredDistance);
        if(combinedSquaredDistance == 0) {
            return 0;
        }
        return float(std::sqrt(float(combinedSquaredDistance)));
    }
}
