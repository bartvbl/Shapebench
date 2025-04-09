#pragma once

#include "Method.h"
#include "nlohmann/json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include "distanceFunctions/euclideanDistance.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>
#include <shapeDescriptor/descriptors/SHOTGenerator.h>

namespace ShapeBench {
    template<typename SHOTDescriptor = ShapeDescriptor::SHOTDescriptor<>>
    struct SHOTMethod {

        static void init(const nlohmann::json& config) {

        }

        static void destroy() {

        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const SHOTDescriptor& descriptor,
                const SHOTDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistanceGPU<SHOTDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static inline float computeDescriptorDistance(
                const SHOTDescriptor& descriptor,
                const SHOTDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistance<SHOTDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
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

        static ShapeDescriptor::gpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<SHOTDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            return ShapeDescriptor::generateSHOTDescriptorsMultiRadius(cloud, descriptorOrigins, supportRadii);
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
