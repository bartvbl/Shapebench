#pragma once

#include "Method.h"
#include "nlohmann/json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>
#include <cfloat>

namespace ShapeBench {

    struct USCMethod {
        inline static float minSupportRadiusFactor = 0;
        inline static float pointDensityRadius = 0;

        static constexpr uint32_t elementsPerShapeContextDescriptor
            = SHAPE_CONTEXT_HORIZONTAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT * SHAPE_CONTEXT_VERTICAL_SLICE_COUNT;

        static void init(const nlohmann::json& config) {
            minSupportRadiusFactor = readDescriptorConfigValue<float>(config, "USC", "minSupportRadiusFactor");
            pointDensityRadius = readDescriptorConfigValue<float>(config, "USC", "pointDensityRadius");
        }

        static void destroy() {

        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeDescriptor::UniqueShapeContextDescriptor& descriptor,
                const ShapeDescriptor::UniqueShapeContextDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistanceGPU<ShapeDescriptor::UniqueShapeContextDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
        }

        static inline float computeDescriptorDistance(
                const ShapeDescriptor::UniqueShapeContextDescriptor& descriptor,
                const ShapeDescriptor::UniqueShapeContextDescriptor& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistance<ShapeDescriptor::UniqueShapeContextDescriptor, float>(descriptor, otherDescriptor, earlyExitThreshold);
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

        static ShapeDescriptor::gpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
#define recomputeDensity true
#if !recomputeDensity
            std::vector<float> minSupportRadii(supportRadii.size());
            for(uint32_t i = 0; i < supportRadii.size(); i++) {
                minSupportRadii.at(i) = minSupportRadiusFactor * supportRadii.at(i);
            }

            ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> descriptors = ShapeDescriptor::generateUniqueShapeContextDescriptorsMultiRadius(
                    cloud, descriptorOrigins, pointDensityRadius, minSupportRadii, supportRadii);
#else
            std::vector<float> minSupportRadii(1);
            std::vector<float> singleRadius(1);
            ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> descriptors(supportRadii.size());
            for(uint32_t i = 0; i < supportRadii.size(); i++) {
                minSupportRadii.at(0) = minSupportRadiusFactor * supportRadii.at(i);
                singleRadius.at(0) = supportRadii.at(i);
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> singlePoint{1, descriptorOrigins.content + i};
                float density = pointDensityRadius * supportRadii.at(i);
                ShapeDescriptor::cpu::array<ShapeDescriptor::UniqueShapeContextDescriptor> descriptor = ShapeDescriptor::generateUniqueShapeContextDescriptorsMultiRadius(cloud, singlePoint, density, minSupportRadii, singleRadius);
                descriptors[i] = descriptor[0];
                ShapeDescriptor::free(descriptor);
            }
#endif
            return descriptors;
        }

        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::CUSTOM;
        }

        static double computeIntersectingAreaCustom(ShapeBench::IntersectionAreaParameters& parameters) {
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
            return "USC";
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["horizontalSliceCount"] = ShapeDescriptor::UniqueShapeContextDescriptor::horizontalSliceCount;
            metadata["verticalSliceCount"] = ShapeDescriptor::UniqueShapeContextDescriptor::verticalSliceCount;
            metadata["layerCount"] = ShapeDescriptor::UniqueShapeContextDescriptor::layerCount;
            metadata["minSupportRadiusScaleFactor"] = minSupportRadiusFactor;
            metadata["pointDensityRadius"] = pointDensityRadius;
            metadata["distanceFunction"] = "Euclidean distance";
            return metadata;
        }
    };
}
