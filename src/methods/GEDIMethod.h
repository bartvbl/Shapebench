#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <nlohmann/json.hpp>
#include "Method.h"
#include "types/IntersectingAreaParameters.h"
#include "types/IntersectingAreaEstimationStrategy.h"
#include "methods/pythonadapter/PythonAdapter.h"
#include "methods/pythonadapter/PythonDescriptor.h"
#include "distanceFunctions/euclideanDistance.h"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"

namespace ShapeBench {
    struct GEDIMethod {

        using DescriptorType = ShapeBench::PythonDescriptor<32>;

        static void init(const nlohmann::json &config) {
            ShapeBench::PythonAdapter<DescriptorType>::init(getName(), config);
        }

        static void destroy() {
            ShapeBench::PythonAdapter<DescriptorType>::destroy();
        }

        static bool usesMeshInput() {
            return false;
        }

        static bool usesPointCloudInput() {
            return true;
        }

        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh &mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwUnimplementedException();
            return {};
        }

        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud &cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<DescriptorType> descriptors = ShapeBench::PythonAdapter<DescriptorType>::computeDescriptors(cloud, descriptorOrigins, config, supportRadii, randomSeed);
            return descriptors;
        }

        static __inline__ float computeDescriptorDistance(
                const DescriptorType& descriptor,
                const DescriptorType& otherDescriptor,
                float earlyExitThreshold) {
            return ShapeBench::computeEuclideanDistance<DescriptorType, float>(descriptor, otherDescriptor, earlyExitThreshold);
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
            return "GEDI";
        }

        static nlohmann::json getMetadata() {
            return PythonAdapter<DescriptorType>::getMetadata();
        }




        // ----- GPU functions, not supported by Python -----
        // They are mandatory for the benchmark, so we just put them here

        static bool hasGPUKernels() {
            return false;
        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const DescriptorType& descriptor,
                const DescriptorType& otherDescriptor,
                float earlyExitThreshold) {
            return 0;
        }

        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh &mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> &device_descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud &cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> &device_descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
    };
}



