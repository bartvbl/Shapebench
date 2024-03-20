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

        __host__ __device__ static __inline__ float computeEuclideanDistance(
                const ShapeDescriptor::ShapeContextDescriptor& descriptor,
                const ShapeDescriptor::ShapeContextDescriptor& otherDescriptor) {
#ifdef __CUDA_ARCH__
            __shared__ float squaredSums[SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT];
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

                if (threadIdx.x == 0) {
                    squaredSums[sliceOffset] = combinedSquaredDistance;
                }
            }

            // An entire warp must participate in the reduction, so we give the excess threads
            // the highest possible value so that any other value will be lower
            float threadValue = threadIdx.x < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT ?
                                squaredSums[threadIdx.x] : FLT_MAX;
            float lowestDistance = std::sqrt(ShapeDescriptor::warpAllReduceMin(threadValue));

            return lowestDistance;
#else
            std::array<float, SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT> squaredSums;
            for(int sliceOffset = 0; sliceOffset < SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT; sliceOffset++) {
                float combinedSquaredDistance = 0;
                for (short binIndex = 0; binIndex < elementsPerShapeContextDescriptor; binIndex++) {
                    float needleBinValue = descriptor.contents[binIndex];
                    short haystackBinIndex = (binIndex + (sliceOffset * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_LAYER_COUNT));
                    // Simple modulo that I think is less expensive
                    if (haystackBinIndex >= elementsPerShapeContextDescriptor) {
                        haystackBinIndex -= elementsPerShapeContextDescriptor;
                    }
                    float haystackBinValue = otherDescriptor.contents[haystackBinIndex];
                    float binDelta = needleBinValue - haystackBinValue;
                    combinedSquaredDistance += binDelta * binDelta;
                }

                squaredSums[sliceOffset] = combinedSquaredDistance;
            }

            // An entire warp must participate in the reduction, so we give the excess threads
            // the highest possible value so that any other value will be lower

            float lowestDistance = squaredSums.at(0);
            for(int i = 1; i < squaredSums.size(); i++) {
                lowestDistance = std::min(lowestDistance, squaredSums.at(i));
            }
            lowestDistance = std::sqrt(lowestDistance);

            return lowestDistance;

#endif
        }

        __host__ __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::ShapeContextDescriptor& descriptor,
                const ShapeDescriptor::ShapeContextDescriptor& otherDescriptor) {
            // Adapter such that the distance function satisfies the "higher distance is worse" criterion
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
            return ShapeDescriptor::generate3DSCDescriptorsMultiRadius(cloud, descriptorOrigins, pointDensityRadius, minSupportRadii, supportRadii);
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