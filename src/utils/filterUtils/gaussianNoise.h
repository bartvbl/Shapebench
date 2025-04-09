#pragma once

#include "shapeDescriptor/shapeDescriptor.h"

namespace ShapeBench {
    void applyGaussianNoise(ShapeDescriptor::cpu::Mesh& mesh, uint64_t randomSeed, float maxDeviation);
}