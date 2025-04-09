#pragma once

#include "Method.h"
#include "nlohmann/json.hpp"
#include "distanceFunctions/pearsonCorrelation.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>

namespace ShapeBench {
    static float supportAngleDegrees;

    struct SIMethod {


        static void init(const nlohmann::json& config) {
            supportAngleDegrees = readDescriptorConfigValue<float>(config, "SI", "supportAngle");
        }

        static void destroy() {

        }



        static inline float computeDescriptorDistance(
                const ShapeDescriptor::SpinImageDescriptor& descriptor,
                const ShapeDescriptor::SpinImageDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            // Adapter such that the distance function satisfies the "higher distance is worse" criterion
            float correlation = ShapeBench::computePearsonCorrelation<ShapeDescriptor::SpinImageDescriptor, float>(descriptor, otherDescriptor);

            // Pearson correlation varies between -1 and 1
            // This makes it such that 0 = best and 2 = worst
            float adjustedCorrelation = 1 - correlation;
            return adjustedCorrelation;
        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeDescriptor::SpinImageDescriptor& descriptor,
                const ShapeDescriptor::SpinImageDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            // Adapter such that the distance function satisfies the "higher distance is worse" criterion
            float correlation = ShapeBench::computePearsonCorrelationGPU<ShapeDescriptor::SpinImageDescriptor, float>(descriptor, otherDescriptor);

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
            return false;
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::SpinImageDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {

            return ShapeDescriptor::generateSpinImagesMultiRadius(cloud, descriptorOrigins, supportRadii, supportAngleDegrees);
        }

        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::FAST_SPHERICAL;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return isPointInCylindricalVolume(descriptorOrigin, supportRadius, supportRadius, samplePoint);
        }

        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            Method::throwUnimplementedException();
            return 0;
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
