#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    struct Orientation {
        ShapeDescriptor::cpu::float3 position {0, 0, 0};
        ShapeDescriptor::cpu::float4 rotation {0, 0, 0, 0};
    };
}