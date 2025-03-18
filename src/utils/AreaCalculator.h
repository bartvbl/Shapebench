#pragma once

#include "shapeDescriptor/shapeDescriptor.h"

namespace ShapeBench {
    struct Cylinder {
        ShapeDescriptor::cpu::double3 centrePoint = {0, 0, 0};
        ShapeDescriptor::cpu::double3 normalisedDirection = {0, 0, 1};
        double halfOfHeight = 1;
        double radius = 1;
    };

    struct Sphere {
        ShapeDescriptor::cpu::double3 centrePoint = {0, 0, 0};
        double radius = 1;
    };

    struct Triangle3D {
        ShapeDescriptor::cpu::double3 vertex0 = {0, 0, 0};
        ShapeDescriptor::cpu::double3 vertex1 = {0, 0, 0};
        ShapeDescriptor::cpu::double3 vertex2 = {0, 0, 0};
    };

    struct Triangle2D {
        ShapeDescriptor::cpu::double2 vertex0 = {0, 0};
        ShapeDescriptor::cpu::double2 vertex1 = {0, 0};
        ShapeDescriptor::cpu::double2 vertex2 = {0, 0};
    };

    struct Circle2D {
        ShapeDescriptor::cpu::double2 origin = {0, 0};
        double radius = 1;
    };


    // Making copies on purpose!
    double computeAreaOfIntersection(Cylinder cylinder, Triangle3D triangle);
    double computeAreaOfIntersection(Sphere sphere, Triangle3D triangle);
    double computeAreaOfIntersection(Circle2D, Triangle2D triangle);
}