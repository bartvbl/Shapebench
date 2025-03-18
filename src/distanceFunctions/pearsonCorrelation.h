#pragma once

#include "methods/Method.h"
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    template<typename DescriptorType, typename contentsType>
        static inline float computePearsonCorrelation(
                const DescriptorType& descriptor,
                const DescriptorType& otherDescriptor) {

            float average1 = 0;
            float average2 = 0;

            constexpr uint32_t elementsInDescriptor = sizeof(DescriptorType) / sizeof(contentsType);

            for (int index = 0; index < elementsInDescriptor; index++) {
                contentsType pixelValue1 = descriptor.contents[index];
                contentsType pixelValue2 = otherDescriptor.contents[index];

                average1 += (float(pixelValue1) - average1) / float(index + 1);
                average2 += (float(pixelValue2) - average2) / float(index + 1);
            }

            float multiplicativeSum = 0;
            float deviationSquaredSumX = 0;
            float deviationSquaredSumY = 0;

            for (int index = 0; index < elementsInDescriptor; index++) {
                contentsType pixelValue1 = descriptor.contents[index];
                contentsType pixelValue2 = otherDescriptor.contents[index];

                float deviation1 = float(pixelValue1) - average1;
                float deviation2 = float(pixelValue2) - average2;

                deviationSquaredSumX += deviation1 * deviation1;
                deviationSquaredSumY += deviation2 * deviation2;
                multiplicativeSum += deviation1 * deviation2;
            }

            float correlation = multiplicativeSum / (sqrt(deviationSquaredSumX) * sqrt(deviationSquaredSumY));

            if(std::isnan(correlation)) {
                correlation = 0;
            }

            return correlation;
        }

    template<typename DescriptorType, typename contentsType>
    __device__ __inline__ float computePearsonCorrelationGPU(DescriptorType descriptor1, DescriptorType descriptor2) {
        constexpr uint32_t elementsInDescriptor = sizeof(DescriptorType) / sizeof(contentsType);
        // Average calculation would be wrong if this is false
        static_assert(elementsInDescriptor % 32 == 0);

        float average1 = 0;
        float average2 = 0;

        for (int index = threadIdx.x; index < elementsInDescriptor; index += warpSize) {
            contentsType pixelValue1 = descriptor1.contents[index];
            contentsType pixelValue2 = descriptor2.contents[index];

            average1 += (float(pixelValue1) - average1) / float(index + 1);
            average2 += (float(pixelValue2) - average2) / float(index + 1);
        }

        float threadMultiplicativeSum = 0;
        float threadDeviationSquaredSumX = 0;
        float threadDeviationSquaredSumY = 0;

        float sumAveragesX = ShapeDescriptor::warpAllReduceSum(average1);
        float sumAveragesY = ShapeDescriptor::warpAllReduceSum(average2);

        average1 = sumAveragesX / float(warpSize);
        average2 = sumAveragesY / float(warpSize);

        for (int index = threadIdx.x; index < elementsInDescriptor; index += warpSize) {
            contentsType pixelValue1 = descriptor1.contents[index];
            contentsType pixelValue2 = descriptor2.contents[index];

            float deviation1 = float(pixelValue1) - average1;
            float deviation2 = float(pixelValue2) - average2;

            threadDeviationSquaredSumX += deviation1 * deviation1;
            threadDeviationSquaredSumY += deviation2 * deviation2;
            threadMultiplicativeSum += deviation1 * deviation2;
        }

        float deviationSquaredSumX = ShapeDescriptor::warpAllReduceSum(threadDeviationSquaredSumX);
        float deviationSquaredSumY = ShapeDescriptor::warpAllReduceSum(threadDeviationSquaredSumY);
        float multiplicativeSum = ShapeDescriptor::warpAllReduceSum(threadMultiplicativeSum);

        float correlation = multiplicativeSum / (sqrt(deviationSquaredSumX) * sqrt(deviationSquaredSumY));

        if(isnan(correlation)) {
            return 0;
        }

        return correlation;
    }
}