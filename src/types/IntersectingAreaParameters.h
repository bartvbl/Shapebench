#pragma once

namespace ShapeBench {
    struct IntersectionAreaParameters {
        ShapeDescriptor::cpu::Mesh mesh;
        ShapeDescriptor::OrientedPoint descriptorOrigin;
        float supportRadius = 1;
        nlohmann::json const* config;
        uint64_t randomSeed = 0;
    };
}