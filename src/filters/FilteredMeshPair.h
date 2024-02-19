#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    struct FilteredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh filteredSampleMesh;
        ShapeDescriptor::cpu::Mesh filteredAdditiveNoise;

        std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices;
        std::vector<ShapeDescriptor::OrientedPoint> mappedReferenceVertices;

        void free();
    };
}