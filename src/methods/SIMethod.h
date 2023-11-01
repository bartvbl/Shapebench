#pragma once

#include "Method.h"
#include "json.hpp"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>

namespace Shapebench {
    struct SIMethod : public Shapebench::Method<ShapeDescriptor::SpinImageDescriptor> {
        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::SpinImageDescriptor& descriptor,
                const ShapeDescriptor::SpinImageDescriptor& otherDescriptor) {

#ifdef __CUDA_ARCH__
            float threadSumX = 0;
            float threadSumY = 0;
            float threadSquaredSumX = 0;
            float threadSquaredSumY = 0;
            float threadMultiplicativeSum = 0;

            // Move input values closer to 0 for better numerical stability
            const float numericalStabilityFactor = 1000.0f;

            const uint32_t count = spinImageWidthPixels * spinImageWidthPixels;

            for (int index = threadIdx.x; index < count; index += warpSize) {
                spinImagePixelType pixelValueX = descriptor.contents[index] / numericalStabilityFactor;
                spinImagePixelType pixelValueY = otherDescriptor.contents[index] / numericalStabilityFactor;

                threadSumX += pixelValueX;
                threadSumY += pixelValueY;
                threadSquaredSumX += pixelValueX * pixelValueX;
                threadSquaredSumY += pixelValueY * pixelValueY;
                threadMultiplicativeSum += pixelValueX * pixelValueY;
            }

            float sumX = ShapeDescriptor::warpAllReduceSum(threadSumX);
            float sumY = ShapeDescriptor::warpAllReduceSum(threadSumY);
            float squaredSumX = ShapeDescriptor::warpAllReduceSum(threadSquaredSumX);
            float squaredSumY = ShapeDescriptor::warpAllReduceSum(threadSquaredSumY);
            float multiplicativeSum = ShapeDescriptor::warpAllReduceSum(threadMultiplicativeSum);

            float correlation = ((float(count) * multiplicativeSum) + (sumX * sumY))
                              / (sqrt((float(count) * squaredSumX) - (sumX * sumX)) * sqrt((float(count) * squaredSumY) - (sumY * sumY)));

            return correlation;
#else

            float sumX = 0;
            float sumY = 0;
            float squaredSumX = 0;
            float squaredSumY = 0;
            float multiplicativeSum = 0;

            // Move input values closer to 0 for better numerical stability
            const float numericalStabilityFactor = 1000.0f;

            const uint32_t count = spinImageWidthPixels * spinImageWidthPixels;

            for (int index = 0; index < count; index++) {
                spinImagePixelType pixelValueX = descriptor.contents[index] / numericalStabilityFactor;
                spinImagePixelType pixelValueY = otherDescriptor.contents[index] / numericalStabilityFactor;

                sumX += pixelValueX;
                sumY += pixelValueY;
                squaredSumX += pixelValueX * pixelValueX;
                squaredSumY += pixelValueY * pixelValueY;
                multiplicativeSum += pixelValueX * pixelValueY;
            }

            float correlation = ((float(count) * multiplicativeSum) + (sumX * sumY))
                                / (sqrt((float(count) * squaredSumX) - (sumX * sumX)) * sqrt((float(count) * squaredSumY) - (sumY * sumY)));

            return correlation;
#endif
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
        static ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            float supportAngleDegrees = readDescriptorConfigValue<float>(config, "Spin Image", "supportAngle");
            return ShapeDescriptor::generateSpinImages(cloud, descriptorOrigins, supportRadius, supportAngleDegrees);
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            float supportAngleDegrees = readDescriptorConfigValue<float>(config, "Spin Image", "supportAngle");
            return ShapeDescriptor::generateSpinImages(cloud, descriptorOrigins, supportRadius, supportAngleDegrees);
        }
        static std::string getName() {
            return "QUICCI";
        }

        static ShapeDescriptor::cpu::array<uint32_t> computeDescriptorRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> haystackDescriptors) {
            return ShapeDescriptor::computeSpinImageSearchResultRanks(needleDescriptors, haystackDescriptors);
        }
    };
}