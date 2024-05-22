#pragma once

#include "Method.h"
#include "json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>
#include <shapeDescriptor/descriptors/SHOTGenerator.h>

namespace ShapeBench {
    template<typename SHOTDescriptor = ShapeDescriptor::SHOTDescriptor<>>
    struct SHOTMethod : public ShapeBench::Method<SHOTDescriptor> {


        static void init(const nlohmann::json& config) {

        }

        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const SHOTDescriptor& descriptor,
                const SHOTDescriptor& otherDescriptor) {

#ifdef __CUDA_ARCH__
            float threadSquaredDistance = 0;
            for (short binIndex = threadIdx.x; binIndex < SHOTDescriptor::totalBinCount; binIndex += blockDim.x) {
                float needleBinValue = descriptor.contents[binIndex];
                float haystackBinValue = otherDescriptor.contents[binIndex];
                float binDelta = needleBinValue - haystackBinValue;
                threadSquaredDistance += binDelta * binDelta;
            }

            float combinedSquaredDistance = ShapeDescriptor::warpAllReduceSum(threadSquaredDistance);
            if(combinedSquaredDistance == 0) {
                return 0;
            }
            return std::sqrt(combinedSquaredDistance);
#else
            float combinedSquaredDistance = 0;
            for (uint32_t binIndex = 0; binIndex < descriptor.totalBinCount; binIndex++) {
                float needleBinValue = descriptor.contents[binIndex];
                float haystackBinValue = otherDescriptor.contents[binIndex];
                float binDelta = needleBinValue - haystackBinValue;
                combinedSquaredDistance += binDelta * binDelta;
            }

            if(combinedSquaredDistance == 0) {
                return 0;
            }

            float distance = std::sqrt(combinedSquaredDistance);
            if(std::isnan(distance)) {
                throw std::runtime_error("Found a NaN!");
            }

            return distance;

#endif
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

        static ShapeDescriptor::gpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method<SHOTDescriptor>::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method<SHOTDescriptor>::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method<SHOTDescriptor>::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            return ShapeDescriptor::generateSHOTDescriptorsMultiRadius<SHOTDescriptor>(cloud, descriptorOrigins, supportRadii);
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return ShapeBench::isPointInSphericalVolume(descriptorOrigin, supportRadius, samplePoint);
        }

        static std::string getName() {
            return "SHOT";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["SHOTElevationDivisions"] = SHOTDescriptor::elevationDivisions;
            metadata["SHOTRadialDivisions"] = SHOTDescriptor::radialDivisions;
            metadata["SHOTAzimuthDivisions"] = SHOTDescriptor::azimuthDivisions;
            metadata["SHOTInternalHistogramBins"] = SHOTDescriptor::internalHistogramBins;
            metadata["distanceFunction"] = "Euclidean distance";
            return metadata;
        }
    };
}
