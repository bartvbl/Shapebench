#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "Seb.h"
#include "json.hpp"

namespace ShapeBench {
    struct Miniball {
        double radius = 0;
        std::array<double, 3> origin = {0, 0, 0};
    };

    inline Miniball computeMiniball(const ShapeDescriptor::cpu::Mesh& mesh) {
        std::vector<Seb::Point<double>> vertices;
        vertices.reserve(mesh.vertexCount);
        std::vector<double> coordinate(3);
        for (uint32_t vertex = 0; vertex < mesh.vertexCount; vertex++) {
            ShapeDescriptor::cpu::float3 point = mesh.vertices[vertex];
            coordinate.at(0) = point.x;
            coordinate.at(1) = point.y;
            coordinate.at(2) = point.z;
            vertices.emplace_back(3, coordinate.begin());
        }
        if(!vertices.empty()) {
            Seb::Smallest_enclosing_ball<double> ball(3, vertices);
            Miniball BALL;
            BALL.radius = ball.radius();
            BALL.origin.at(0) = ball.center_begin()[0];
            BALL.origin.at(1) = ball.center_begin()[1];
            BALL.origin.at(2) = ball.center_begin()[2];
            return BALL;
        }
        Miniball emptyBall;
        return emptyBall;
    }

    inline Miniball computeMiniball(const ShapeDescriptor::cpu::PointCloud& pointCloud) {
        std::vector<Seb::Point<double>> vertices;
        vertices.reserve(pointCloud.pointCount);
        std::vector<double> coordinate(3);
        for (uint32_t vertex = 0; vertex < pointCloud.pointCount; vertex++) {
            ShapeDescriptor::cpu::float3 point = pointCloud.vertices[vertex];
            coordinate.at(0) = point.x;
            coordinate.at(1) = point.y;
            coordinate.at(2) = point.z;
            vertices.emplace_back(3, coordinate.begin());
        }
        if(!vertices.empty()) {
            Seb::Smallest_enclosing_ball<double> ball(3, vertices);
            Miniball BALL; // BALL
            BALL.radius = ball.radius();
            BALL.origin.at(0) = ball.center_begin()[0];
            BALL.origin.at(1) = ball.center_begin()[1];
            BALL.origin.at(2) = ball.center_begin()[2];
            return BALL;
        }
        Miniball emptyBall;
        return emptyBall;
    }

    inline void verifyMiniballValidity(const Miniball ball, const Miniball otherBall) {
        const double MAX_ERROR = 0.0000001;
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

