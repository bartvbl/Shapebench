#include "areaEstimator.h"
#include "utils/AreaCalculator.h"

double ShapeBench::computeAreaInCylindricalSupportVolume(const ShapeDescriptor::cpu::Mesh &mesh, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius) {
    double intersectingArea = 0;

    for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeBench::Cylinder cylinder;
        cylinder.centrePoint = toDouble3(referencePoint.vertex);
        cylinder.normalisedDirection = toDouble3(referencePoint.normal);
        cylinder.halfOfHeight = supportRadius / 2.0f;
        cylinder.radius = supportRadius;
        ShapeBench::Triangle3D triangle;
        triangle.vertex0 = toDouble3(mesh.vertices[i]);
        triangle.vertex1 = toDouble3(mesh.vertices[i + 1]);
        triangle.vertex2 = toDouble3(mesh.vertices[i + 2]);
        double triangleIntersectionArea = ShapeBench::computeAreaOfIntersection(cylinder, triangle);
        intersectingArea += triangleIntersectionArea;
    }

    return intersectingArea;
}

double ShapeBench::computeAreaInSphericalSupportVolume(const ShapeDescriptor::cpu::Mesh &mesh, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius) {
    double intersectingArea = 0;

    for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeBench::Sphere sphere;
        sphere.centrePoint = toDouble3(referencePoint.vertex);
        sphere.radius = supportRadius;
        ShapeBench::Triangle3D triangle;
        triangle.vertex0 = toDouble3(mesh.vertices[i]);
        triangle.vertex1 = toDouble3(mesh.vertices[i + 1]);
        triangle.vertex2 = toDouble3(mesh.vertices[i + 2]);
        double triangleIntersectionArea = ShapeBench::computeAreaOfIntersection(sphere, triangle);
        intersectingArea += triangleIntersectionArea;
    }

    return intersectingArea;
}