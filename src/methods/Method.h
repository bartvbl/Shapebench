#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace Shapebench {
    template<typename DescriptorType>
    struct Method {
    private:
        static void throwUnimplementedException() {
            throw std::logic_error("This method is required but has not been implemented.");
        }
    public:
        static bool usesMeshInput() {
            throwUnimplementedException();
            return false;
        }
        static bool usesPointCloudInput() {
            throwUnimplementedException();
            return false;
        }
        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                float supportRadius) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                float supportRadius) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::cpu::array<uint32_t> computeDescriptorRanks(
                ShapeDescriptor::gpu::array<DescriptorType> needleDescriptors,
                ShapeDescriptor::gpu::array<DescriptorType> haystackDescriptors
                ) {
            throwUnimplementedException();
            return {};
        }
        static std::string getName() {
            return "METHOD NAME NOT SPECIFIED";
        }
    };




}
