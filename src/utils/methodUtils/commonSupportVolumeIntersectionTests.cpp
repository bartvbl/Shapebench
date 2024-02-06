#include "commonSupportVolumeIntersectionTests.h"

bool ShapeBench::isPointInCylindricalVolume(ShapeDescriptor::OrientedPoint referencePoint,
                                float cylinderRadius, float cylinderHeight,
                                ShapeDescriptor::cpu::float3 point) {
    float beta = dot(point - referencePoint.vertex, referencePoint.normal) / dot(referencePoint.normal, referencePoint.normal);

    ShapeDescriptor::cpu::float3 projectedPoint = referencePoint.vertex + beta * referencePoint.normal;
    ShapeDescriptor::cpu::float3 delta = projectedPoint - point;
    float alpha = length(delta);
    float cylinderHalfHeight = cylinderHeight / 2.0f;

    return alpha <= cylinderRadius && beta >= -cylinderHalfHeight && beta <= cylinderHalfHeight;
}

bool ShapeBench::isPointInSphericalVolume(ShapeDescriptor::OrientedPoint referencePoint,
                              float supportRadius,
                              ShapeDescriptor::cpu::float3 point) {
    ShapeDescriptor::cpu::float3 delta = point - referencePoint.vertex;
    return length(delta) <= supportRadius;
}