#pragma once

#include "Method.h"
#include "nlohmann/json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include "distanceFunctions/euclideanDistance.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>

namespace ShapeBench {
    static float RoPSPointCloudSamplingDensity;
    static uint32_t RoPSPointCloudSampleLimit;

    struct RoPSMethod {

        static void init(const nlohmann::json& config) {
            RoPSPointCloudSamplingDensity = readDescriptorConfigValue<float>(config, "RoPS", "pointSamplingDensity");
            RoPSPointCloudSampleLimit = readDescriptorConfigValue<uint32_t>(config, "RoPS", "pointSampleLimit");
        }

        static void destroy() {

        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeDescriptor::RoPSDescriptor& descriptor,
                const ShapeDescriptor::RoPSDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistanceGPU<ShapeDescriptor::RoPSDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static inline float computeDescriptorDistance(
                const ShapeDescriptor::RoPSDescriptor& descriptor,
                const ShapeDescriptor::RoPSDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistance<ShapeDescriptor::RoPSDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
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

        static ShapeDescriptor::gpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> outDescriptors(descriptorOrigins.length);
            for(uint32_t i = 0; i < descriptorOrigins.length; i++) {
                ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> descriptor =
                        ShapeDescriptor::generateRoPSDescriptors(
                                 mesh,
                                 {1, &descriptorOrigins.content[i]},
                                 supportRadii.at(i),
                                 RoPSPointCloudSamplingDensity,
                                 randomSeed, RoPSPointCloudSampleLimit);
                outDescriptors[i] = descriptor[0];
                ShapeDescriptor::free(descriptor);
            }
            return outDescriptors;
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::RoPSDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::FAST_SPHERICAL;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return ShapeBench::isPointInSphericalVolume(descriptorOrigin, supportRadius, samplePoint);
        }

        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            Method::throwUnimplementedException();
            return 0;
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