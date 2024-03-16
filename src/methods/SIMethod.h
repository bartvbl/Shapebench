#pragma once

#include "Method.h"
#include "json.hpp"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>

namespace ShapeBench {
    struct SIMethod : public ShapeBench::Method<ShapeDescriptor::SpinImageDescriptor> {
        __host__ __device__ static __inline__ float computePearsonCorrelation(
                const ShapeDescriptor::SpinImageDescriptor& descriptor,
                const ShapeDescriptor::SpinImageDescriptor& otherDescriptor) {
#ifdef __CUDA_ARCH__
            float threadSumX = 0;
            float threadSumY = 0;

            // Move input values closer to 0 for better numerical stability
            const float numericalStabilityFactor = 1000.0f;

            const uint32_t count = spinImageWidthPixels * spinImageWidthPixels;

            for (int index = threadIdx.x; index < count; index += warpSize) {
                spinImagePixelType pixelValueX = descriptor.contents[index] / numericalStabilityFactor;
                spinImagePixelType pixelValueY = otherDescriptor.contents[index] / numericalStabilityFactor;

                threadSumX += pixelValueX;
                threadSumY += pixelValueY;
            }

            float threadMultiplicativeSum = 0;
            float threadDeviationSquaredSumX = 0;
            float threadDeviationSquaredSumY = 0;

            float sumX = ShapeDescriptor::warpAllReduceSum(threadSumX);
            float sumY = ShapeDescriptor::warpAllReduceSum(threadSumY);

            if(sumX == 0 && sumY == 0) {
                return -1;
            }

            float averageX = sumX / float(count);
            float averageY = sumY / float(count);

            for (int index = threadIdx.x; index < count; index += warpSize) {
                spinImagePixelType pixelValueX = descriptor.contents[index] / numericalStabilityFactor;
                spinImagePixelType pixelValueY = otherDescriptor.contents[index] / numericalStabilityFactor;

                float deviationX = pixelValueX - averageX;
                float deviationY = pixelValueY - averageY;

                threadDeviationSquaredSumX += deviationX * deviationX;
                threadDeviationSquaredSumY += deviationY * deviationY;
                threadMultiplicativeSum += deviationX * deviationY;
            }

            float deviationSquaredSumX = ShapeDescriptor::warpAllReduceSum(threadDeviationSquaredSumX);
            float deviationSquaredSumY = ShapeDescriptor::warpAllReduceSum(threadDeviationSquaredSumY);
            float multiplicativeSum = ShapeDescriptor::warpAllReduceSum(threadMultiplicativeSum);

            float correlation = multiplicativeSum / (sqrt(deviationSquaredSumX) * sqrt(deviationSquaredSumY));

            if(isnan(correlation)) {
                return 0;
            }

            return correlation;
#else

            float sumX = 0;
            float sumY = 0;

            // Move input values closer to 0 for better numerical stability
            const float numericalStabilityFactor = 1000.0f;

            const uint32_t count = spinImageWidthPixels * spinImageWidthPixels;

            for (int index = 0; index < count; index++) {
                spinImagePixelType pixelValueX = descriptor.contents[index] / numericalStabilityFactor;
                spinImagePixelType pixelValueY = otherDescriptor.contents[index] / numericalStabilityFactor;

                sumX += pixelValueX;
                sumY += pixelValueY;
            }

            float multiplicativeSum = 0;
            float deviationSquaredSumX = 0;
            float deviationSquaredSumY = 0;

            float averageX = sumX / float(count);
            float averageY = sumY / float(count);

            for (int index = 0; index < count; index++) {
                spinImagePixelType pixelValueX = descriptor.contents[index] / numericalStabilityFactor;
                spinImagePixelType pixelValueY = otherDescriptor.contents[index] / numericalStabilityFactor;

                float deviationX = pixelValueX - averageX;
                float deviationY = pixelValueY - averageY;

                deviationSquaredSumX += deviationX * deviationX;
                deviationSquaredSumY += deviationY * deviationY;
                multiplicativeSum += deviationX * deviationY;
            }

            float correlation = multiplicativeSum / (sqrt(deviationSquaredSumX) * sqrt(deviationSquaredSumY));

            if(std::isnan(correlation)) {
                correlation = 0;
            }

            return correlation;
#endif
        }

        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::SpinImageDescriptor& descriptor,
                const ShapeDescriptor::SpinImageDescriptor& otherDescriptor) {
            // Adapter such that the distance function satisfies the "higher distance is worse" criterion
            float correlation = computePearsonCorrelation(descriptor, otherDescriptor);

            // Pearson correlation varies between -1 and 1
            // This makes it such that 0 = best and 2 = worst
            float adjustedCorrelation = 1 - correlation;
            return adjustedCorrelation;
        }

        static bool usesMeshInput() {
            return false;
        }
        static bool usesPointCloudInput() {
            return true;
        }
        static bool hasGPUKernels() {
            return true;
        }
        static bool shouldUseGPUKernel() {
            return false;
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            float supportAngleDegrees = readDescriptorConfigValue<float>(config, "Spin Image", "supportAngle");
            return ShapeDescriptor::generateSpinImages(cloud, descriptorOrigins, supportRadius, supportAngleDegrees);
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            float supportAngleDegrees = readDescriptorConfigValue<float>(config, "Spin Image", "supportAngle");
            return ShapeDescriptor::generateSpinImages(cloud, descriptorOrigins, supportRadius, supportAngleDegrees);
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return isPointInCylindricalVolume(descriptorOrigin, supportRadius, supportRadius, samplePoint);
        }

        static std::string getName() {
            return "SI";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["resolution"] = spinImageWidthPixels;
            metadata["pixelType"] = std::string(typeid(spinImagePixelType).name());
            metadata["distanceFunction"] = "Pearson Correlation";
            return metadata;
        }
    };
}