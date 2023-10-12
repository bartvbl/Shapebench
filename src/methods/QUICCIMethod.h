#pragma once

#include "Method.h"
#include "shapeDescriptor/common/types/methods/QUICCIDescriptor.h"

namespace Shapebench {
    struct QUICCIMethod : public Shapebench::Method<ShapeDescriptor::QUICCIDescriptor> {
        __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::QUICCIDescriptor& descriptor,
                const ShapeDescriptor::QUICCIDescriptor& otherDescriptor) {
            return 0.5;
        }
    };
}