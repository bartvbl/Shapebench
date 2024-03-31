#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    ShapeDescriptor::cpu::Mesh simulateKinectCapture(const ShapeDescriptor::cpu::Mesh& inputMesh);
}