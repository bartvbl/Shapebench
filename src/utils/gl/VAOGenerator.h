#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "GeometryBuffer.h"

namespace ShapeBench {
    GeometryBuffer generateVertexArray(
            ShapeDescriptor::cpu::float3* vertices,
            ShapeDescriptor::cpu::float3* normals,
            ShapeDescriptor::cpu::float3* colours,
            unsigned int vertexCount);
    void destroyVertexArray(GeometryBuffer buffer);
}
