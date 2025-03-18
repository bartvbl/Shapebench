#pragma once

#include "Method.h"
#include "json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>

namespace ShapeBench {
    static float FPFHPointSamplingDensity;

    struct FPFHMethod : public ShapeBench::Method<ShapeDescriptor::FPFHDescriptor> {

        static constexpr uint32_t elementsPerFPFHDescriptor = 3 * FPFH_BINS_PER_FEATURE;

        static void init(const nlohmann::json& config) {
            FPFHPointSamplingDensity = readDescriptorConfigValue<float>(config, "FPFH", "pointSamplingDensity");

        }

        __host__ __device__ static __inline__ float computeEuclideanDistance(
                const ShapeDescriptor::FPFHDescriptor& descriptor,
                const ShapeDescriptor::FPFHDescriptor& otherDescriptor) {
#ifdef __CUDA_ARCH__
            float threadSquaredDistance = 0;
            for (short binIndex = threadIdx.x; binIndex < elementsPerFPFHDescriptor; binIndex += blockDim.x) {
                float needleBinValue = descriptor.contents[binIndex];
                float haystackBinValue = otherDescriptor.contents[binIndex];
                float binDelta = needleBinValue - haystackBinValue;
                threadSquaredDistance += binDelta * binDelta;
            }

            float totalSquaredDistance = ShapeDescriptor::warpAllReduceSum(threadSquaredDistance);
            return sqrt(totalSquaredDistance);
#else
            float combinedSquaredDistance = 0;
            for (short binIndex = 0; binIndex < elementsPerFPFHDescriptor; binIndex++) {
                float needleBinValue = descriptor.contents[binIndex];
                float haystackBinValue = otherDescriptor.contents[binIndex];
                float binDelta = needleBinValue - haystackBinValue;
                combinedSquaredDistance += binDelta * binDelta;
            }

            return std::sqrt(combinedSquaredDistance);
#endif
        }

        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::FPFHDescriptor& descriptor,
                const ShapeDescriptor::FPFHDescriptor& otherDescriptor) {
            return computeEuclideanDistance(descriptor, otherDescriptor);
        }

        static bool usesMeshInput() {
            return false;
        }

        static bool usesPointCloudInput() {
            return true;
        }

        static bool hasGPUKernels() {
            return false;
        }

        static bool shouldUseGPUKernel() {
            return false;
        }

        static ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::FPFHDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<ShapeDescriptor::FPFHDescriptor> outDescriptors(descriptorOrigins.length);



            return outDescriptors;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return ShapeBench::isPointInSphericalVolume(descriptorOrigin, supportRadius, samplePoint);
        }

        static std::string getName() {
            return "FPFH";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["distanceFunction"] = "Euclidean distance";
            return metadata;
        }
    };
}