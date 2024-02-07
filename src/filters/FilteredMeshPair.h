#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    struct FilteredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh filteredSampleMesh;
        ShapeDescriptor::cpu::Mesh filteredAdditiveNoise;

        ShapeDescriptor::OrientedPoint originalReferenceVertex;
        ShapeDescriptor::OrientedPoint mappedReferenceVertex;

        void free();
    };
}