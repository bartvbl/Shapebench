#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

bool isPointInCylindricalVolume(ShapeDescriptor::OrientedPoint referencePoint, float cylinderWidth, float cylinderHeight, ShapeDescriptor::cpu::float3 point);
bool isPointInSphericalVolume(ShapeDescriptor::OrientedPoint referencePoint, float supportRadius, ShapeDescriptor::cpu::float3 point);