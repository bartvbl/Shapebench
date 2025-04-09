#pragma once

#include "Method.h"
#include "nlohmann/json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include "distanceFunctions/euclideanDistance.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>

namespace ShapeBench {
    static float minSupportRadiusFactor;
    static float pointDensityRadius;

    struct ShapeContextMethod {

        static constexpr uint32_t elementsPerShapeContextDescriptor
            = SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

        static void init(const nlohmann::json& config) {
            minSupportRadiusFactor = readDescriptorConfigValue<float>(config, "3DSC", "minSupportRadiusFactor");
            pointDensityRadius = readDescriptorConfigValue<float>(config, "3DSC", "pointDensityRadius");
        }

        static void destroy() {

        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeDescriptor::ShapeContextDescriptor& descriptor,
                const ShapeDescriptor::ShapeContextDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistanceGPU<ShapeDescriptor::ShapeContextDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static inline float computeDescriptorDistance(
                const ShapeDescriptor::ShapeContextDescriptor& descriptor,
                const ShapeDescriptor::ShapeContextDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistance<ShapeDescriptor::ShapeContextDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
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

        static ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            std::vector<float> minSupportRadii(1);
            std::vector<float> singleRadius(1);
            ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptors(supportRadii.size());
            for(uint32_t i = 0; i < supportRadii.size(); i++) {
                minSupportRadii.at(0) = minSupportRadiusFactor * supportRadii.at(i);
                singleRadius.at(0) = supportRadii.at(i);
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> singlePoint{1, descriptorOrigins.content + i};
                float density = pointDensityRadius * supportRadii.at(i);
                ShapeDescriptor::cpu::array<ShapeDescriptor::ShapeContextDescriptor> descriptor = ShapeDescriptor::generate3DSCDescriptorsMultiRadius(cloud, singlePoint, density, minSupportRadii, singleRadius);
                descriptors[i] = descriptor[0];
                ShapeDescriptor::free(descriptor);
            }
            return descriptors;
        }

        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::CUSTOM;
        }

        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            // ONLY needed when getIntersectingAreaEstimationStrategy() indicates a custom method will be used
            // Refer to the documentation for useful utility functions that may help you
            double outerArea = ShapeBench::computeAreaInSphericalSupportVolume(parameters.mesh, parameters.descriptorOrigin, parameters.supportRadius);
            double innerArea = ShapeBench::computeAreaInSphericalSupportVolume(parameters.mesh, parameters.descriptorOrigin, minSupportRadiusFactor * parameters.supportRadius);
            return outerArea - innerArea;
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
