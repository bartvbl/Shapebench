#include "miniballGenerator.h"

ShapeBench::Miniball ShapeBench::computeMiniball(const ShapeDescriptor::cpu::Mesh& mesh) {
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
            ShapeBench::Miniball BALL;
            BALL.radius = ball.radius();
            BALL.origin.at(0) = ball.center_begin()[0];
            BALL.origin.at(1) = ball.center_begin()[1];
            BALL.origin.at(2) = ball.center_begin()[2];
            return BALL;
        }
        ShapeBench::Miniball emptyBall;
        return emptyBall;
    }

    ShapeBench::Miniball ShapeBench::computeMiniball(const ShapeDescriptor::cpu::PointCloud& pointCloud) {
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
            ShapeBench::Miniball BALL; // BALL
            BALL.radius = ball.radius();
            BALL.origin.at(0) = ball.center_begin()[0];
            BALL.origin.at(1) = ball.center_begin()[1];
            BALL.origin.at(2) = ball.center_begin()[2];
            return BALL;
        }
        ShapeBench::Miniball emptyBall;
        return emptyBall;
    }
