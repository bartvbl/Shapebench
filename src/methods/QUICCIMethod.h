#pragma once

#include "Method.h"
#include <shapeDescriptor/shapeDescriptor.h>

namespace Shapebench {
    struct QUICCIMethod : public Shapebench::Method<ShapeDescriptor::QUICCIDescriptor> {
        __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::QUICCIDescriptor& descriptor,
                const ShapeDescriptor::QUICCIDescriptor& otherDescriptor) {
            return 0.5;
        }

        static bool usesMeshInput() {
            return true;
        }
        static bool usesPointCloudInput() {
            return false;
        }
        static bool hasGPUKernels() {
            return true;
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            return ShapeDescriptor::generateQUICCImages(mesh, descriptorOrigins, supportRadius);
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            return ShapeDescriptor::generateQUICCImages(mesh, descriptorOrigins, supportRadius);
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwIncompatibleException();
            return {};
        }
        static std::string getName() {
            return "QUICCI";
        }

        static ShapeDescriptor::cpu::array<uint32_t> computeDescriptorRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors) {
            return ShapeDescriptor::computeQUICCImageSearchResultRanks(needleDescriptors, haystackDescriptors);
        }
    };
}