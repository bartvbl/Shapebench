#pragma once

#include "Method.h"
#include "json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>

namespace ShapeBench {
    static float minSupportRadiusFactor;
    static float pointDensityRadius;

    struct ShapeContextMethod : public ShapeBench::Method<ShapeDescriptor::ShapeContextDescriptor> {

        static constexpr uint32_t elementsPerShapeContextDescriptor
            = SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

        static void init(const nlohmann::json& config) {
            minSupportRadiusFactor = readDescriptorConfigValue<float>(config, "3DSC", "minSupportRadiusFactor");
            pointDensityRadius = readDescriptorConfigValue<float>(config, "3DSC", "pointDensityRadius");
        }

        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::ShapeContextDescriptor& descriptor,
                const ShapeDescriptor::ShapeContextDescriptor& otherDescriptor) {

#ifdef __CUDA_ARCH__
            float lowestSquareSum;
            for(int sliceOffset = 0; sliceOffset < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT; sliceOffset++) {
                float threadSquaredDistance = 0;
                for (short binIndex = threadIdx.x; binIndex < elementsPerShapeContextDescriptor; binIndex += blockDim.x) {
                    float needleBinValue = descriptor.contents[binIndex];
                    short haystackBinIndex = (binIndex + (sliceOffset * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT));
                    // Simple modulo that I think is less expensive
                    if (haystackBinIndex >= elementsPerShapeContextDescriptor) {
                        haystackBinIndex -= elementsPerShapeContextDescriptor;
                    }
                    float haystackBinValue = otherDescriptor.contents[haystackBinIndex];
                    float binDelta = needleBinValue - haystackBinValue;
                    threadSquaredDistance += binDelta * binDelta;
                }

                float combinedSquaredDistance = ShapeDescriptor::warpAllReduceSum(threadSquaredDistance);

                if (threadIdx.x == 0 && (sliceOffset == 0 || combinedSquaredDistance < lowestSquareSum)) {
                    lowestSquareSum = combinedSquaredDistance;
                }
            }

            float lowestDistance = std::sqrt(lowestSquareSum);
            return lowestDistance;
#else
            float squaredSum = 1;
            for(int sliceOffset = 0; sliceOffset < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT; sliceOffset++) {
                float combinedSquaredDistance = 0;
                for (short binIndex = 0; (binIndex < elementsPerShapeContextDescriptor) && (combinedSquaredDistance < squaredSum); binIndex++) {
                    float needleBinValue = descriptor.contents[binIndex];
                    uint32_t haystackBinIndex = (binIndex + (sliceOffset * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT));
                    // Simple modulo that I think is less expensive
                    if (haystackBinIndex >= elementsPerShapeContextDescriptor) {
                        haystackBinIndex -= elementsPerShapeContextDescriptor;
                    }
                    float haystackBinValue = otherDescriptor.contents[haystackBinIndex];
                    float binDelta = needleBinValue - haystackBinValue;
                    combinedSquaredDistance += binDelta * binDelta;
                }

                if(sliceOffset == 0 || combinedSquaredDistance < squaredSum) {
                    squaredSum = combinedSquaredDistance;
                }
            }

            float lowestDistance = std::sqrt(squaredSum);
            if(std::isnan(lowestDistance)) {
                throw std::runtime_error("Found a NaN!");
            }

            return lowestDistance;

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

        static ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            std::vector<float> minSupportRadii(descriptorOrigins.length);
            for(uint32_t i = 0; i < supportRadii.size(); i++) {
                minSupportRadii.at(i) = minSupportRadiusFactor * supportRadii.at(i);
            }
            ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors
                = ShapeDescriptor::generate3DSCDescriptorsMultiRadius(cloud, descriptorOrigins, pointDensityRadius, minSupportRadii, supportRadii);

            return descriptors;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            bool isWithinOuterRadius = ShapeBench::isPointInSphericalVolume(descriptorOrigin, supportRadius, samplePoint);
            bool isWithinInnerRadius = ShapeBench::isPointInSphericalVolume(descriptorOrigin, minSupportRadiusFactor * supportRadius, samplePoint);
            return isWithinOuterRadius && !isWithinInnerRadius;
        }

        static std::string getName() {
            return "3DSC";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["horizontalSliceCount"] = SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT;
            metadata["verticalSliceCount"] = SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;
            metadata["layerCount"] = SHAPE_CONTEXT_LAYER_COUNT;
            metadata["minSupportRadiusScaleFactor"] = minSupportRadiusFactor;
            metadata["pointDensityRadius"] = pointDensityRadius;
            metadata["distanceFunction"] = "Euclidean distance";
            return metadata;
        }
    };
}
