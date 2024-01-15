#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

struct BufferObject {
    unsigned int VAOID = 0;
    unsigned int vertexBufferID = 0;
    unsigned int normalBufferID = 0;
    unsigned int colourBufferID = 0;
    unsigned int indexBufferID = 0;
};

BufferObject generateVertexArray(
        ShapeDescriptor::cpu::float3* vertices,
        ShapeDescriptor::cpu::float3* normals,
        ShapeDescriptor::cpu::float3* colours,
        unsigned int vertexCount);