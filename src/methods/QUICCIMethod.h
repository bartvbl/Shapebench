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

        static ShapeDescriptor::cpu::array<uint32_t> computeDescriptorRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors) {
            return ShapeDescriptor::computeQUICCImageSearchResultRanks(needleDescriptors, haystackDescriptors);
        }
    };
}