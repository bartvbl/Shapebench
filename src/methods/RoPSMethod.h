#pragma once

#include "Method.h"
#include "json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>

namespace ShapeBench {
    static float RoPSPointCloudSamplingDensity;
    static uint32_t RoPSPointCloudSampleLimit;

    struct RoPSMethod : public ShapeBench::Method<ShapeDescriptor::RoPSDescriptor> {

        static constexpr int elementsPerRoPSDescriptor = ROPS_NUM_ROTATIONS * 3 * 3 * 5;

        static void init(const nlohmann::json& config) {
            RoPSPointCloudSamplingDensity = readDescriptorConfigValue<float>(config, "RoPS", "pointSamplingDensity");
            RoPSPointCloudSampleLimit = readDescriptorConfigValue<uint32_t>(config, "RoPS", "pointSampleLimit");

        }

        __host__ __device__ static __inline__ float computeEuclideanDistance(
                const ShapeDescriptor::RoPSDescriptor& descriptor,
                const ShapeDescriptor::RoPSDescriptor& otherDescriptor) {
#ifdef __CUDA_ARCH__
            float threadSquaredDistance = 0;
            for (short binIndex = threadIdx.x; binIndex < elementsPerRoPSDescriptor; binIndex += blockDim.x) {
                float needleBinValue = descriptor.contents[binIndex];
                float haystackBinValue = otherDescriptor.contents[binIndex];
                float binDelta = needleBinValue - haystackBinValue;
                threadSquaredDistance += binDelta * binDelta;
            }

            float totalSquaredDistance = ShapeDescriptor::warpAllReduceSum(threadSquaredDistance);
            return sqrt(totalSquaredDistance);
#else
            float combinedSquaredDistance = 0;
            for (short binIndex = 0; binIndex < elementsPerRoPSDescriptor; binIndex++) {
                float needleBinValue = descriptor.contents[binIndex];
                float haystackBinValue = otherDescriptor.contents[binIndex];
                float binDelta = needleBinValue - haystackBinValue;
                combinedSquaredDistance += binDelta * binDelta;
            }

            return std::sqrt(combinedSquaredDistance);
#endif
        }

        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::RoPSDescriptor& descriptor,
                const ShapeDescriptor::RoPSDescriptor& otherDescriptor) {
            return computeEuclideanDistance(descriptor, otherDescriptor);
        }

        static bool usesMeshInput() {
            return true;
        }

        static bool usesPointCloudInput() {
            return false;
        }

        static bool hasGPUKernels() {
            return false;
        }

        static bool shouldUseGPUKernel() {
            return false;
        }

        static ShapeDescriptor::gpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            throwIncompatibleException();
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            return ShapeDescriptor::generateRoPSDescriptors(mesh,
                    descriptorOrigins,
                    supportRadius,
                    RoPSPointCloudSamplingDensity,
                    randomSeed, RoPSPointCloudSampleLimit);
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius,
                uint64_t randomSeed) {
            throwIncompatibleException();
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return ShapeBench::isPointInSphericalVolume(descriptorOrigin, supportRadius, samplePoint);
        }

        static std::string getName() {
            return "RoPS";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["rotationCount"] = ROPS_NUM_ROTATIONS;
            metadata["binCount"] = ROPS_HISTOGRAM_BINS;
            metadata["pointCloudSamplingDensity"] = RoPSPointCloudSamplingDensity;
            metadata["pointCloudSampleLimit"] = RoPSPointCloudSampleLimit;
            metadata["distanceFunction"] = "Euclidean distance";
            return metadata;
        }
    };
}