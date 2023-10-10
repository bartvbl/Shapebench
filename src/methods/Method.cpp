#include <stdexcept>
#include "Method.h"

void throwUnimplementedException() {
    throw std::logic_error("This method is required but has not been implemented.");
}

template<typename DescriptorType>
ShapeDescriptor::gpu::array<DescriptorType>
Shapebench::Method<DescriptorType>::computeDescriptors(const ShapeDescriptor::gpu::PointCloud mesh,
                                           const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                                           float supportRadius) {
    throwUnimplementedException();
    return {};
}

template<typename DescriptorType>
ShapeDescriptor::gpu::array<DescriptorType>
Shapebench::Method<DescriptorType>::computeDescriptors(const ShapeDescriptor::gpu::Mesh mesh,
                                           const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                                           float supportRadius) {
    throwUnimplementedException();
    return {};
}

template<typename DescriptorType>
bool Shapebench::Method<DescriptorType>::usesPointCloudInput() {
    throwUnimplementedException();
    return false;
}

template<typename DescriptorType>
bool Shapebench::Method<DescriptorType>::usesMeshInput() {
    throwUnimplementedException();
    return false;
}