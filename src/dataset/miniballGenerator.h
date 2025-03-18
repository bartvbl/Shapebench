#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "Seb.h"
#include "nlohmann/json.hpp"

namespace ShapeBench {
    struct Miniball {
        double radius = 0;
        std::array<double, 3> origin = {0, 0, 0};
    };

    Miniball computeMiniball(const ShapeDescriptor::cpu::Mesh& mesh);
    Miniball computeMiniball(const ShapeDescriptor::cpu::PointCloud& pointCloud);

    inline void verifyMiniballValidity(const Miniball ball, const Miniball otherBall) {
        // The functions above are called in two different places, and are resulting in numerical errors
	// between the values stored in the dataset file, and the one being computed to verify it.
	// This is the minimum sufficient error to ensure equivalent values still pass.
	const double MAX_ERROR = 0.00001;
        if(std::abs(otherBall.radius - ball.radius) >= MAX_ERROR) {
            throw std::logic_error("FATAL: The computed bounding sphere radius deviates from the one in the cache: " + std::to_string(otherBall.radius) + " vs " + std::to_string(ball.radius));
        }
        if(std::abs(otherBall.origin.at(0) - ball.origin.at(0)) >= MAX_ERROR
        || std::abs(otherBall.origin.at(1) - ball.origin.at(1)) >= MAX_ERROR
        || std::abs(otherBall.origin.at(2) - ball.origin.at(2)) >= MAX_ERROR) {
            throw std::logic_error("FATAL: The computed bounding sphere coordinate deviated from the one in the cache: (" +
                 std::to_string(otherBall.origin.at(0)) + ", "
               + std::to_string(otherBall.origin.at(1)) + ", "
               + std::to_string(otherBall.origin.at(2))
               + ") vs ("
               + std::to_string(ball.origin.at(0)) + ", "
               + std::to_string(ball.origin.at(1)) + ", "
               + std::to_string(ball.origin.at(2)) + ")");
        }
    }
}

