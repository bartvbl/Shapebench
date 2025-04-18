

#include "AreaCalculator.h"
#include "glm/ext/matrix_transform.hpp"
#include "utils/debugUtils/DebugRenderer.h"
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/normal.hpp>
#include <algorithm>

typedef glm::vec<2, double, glm::highp> dvec2;
typedef glm::vec<3, double, glm::highp> dvec3;
typedef glm::vec<4, double, glm::highp> dvec4;

namespace ShapeBench {
    inline double angle(dvec2 a, dvec2 b) {
        dvec2 normalisedA = glm::normalize(a);
        dvec2 normalisedB = glm::normalize(b);
        double determinant = normalisedA.x * normalisedB.y - normalisedA.y * normalisedB.x;
        return std::atan2(glm::dot(normalisedA, normalisedB), determinant);
    }

    dvec3 alignWithXAxis_XY(dvec3 directionVector, dvec3 coordinateToRotate) {
        // This is effectively applying a 2D rotation matrix for counter clockwise rotation
        // where the angle has been negated. Cosine is unaffected, while it negates the sine function
        // Since a normalised vector gives you the sine and cosine of the desired rotation angle directly,
        // this will always align any angle with the x-axis.
        // Does not apply any rotation if rotation vector is zero to avoid NaN
        dvec2 direction2D_not_normalised(directionVector.x, directionVector.y);

        if(direction2D_not_normalised.x == 0 && direction2D_not_normalised.y == 0) {
            return coordinateToRotate;
        }

        dvec2 direction2D = glm::normalize(direction2D_not_normalised);
        double sine = direction2D.y;
        double cosine = direction2D.x;

        const double initialTransformedX = coordinateToRotate.x;
        coordinateToRotate.x = cosine * coordinateToRotate.x + sine * coordinateToRotate.y;
        coordinateToRotate.y = -sine * initialTransformedX + cosine * coordinateToRotate.y;
        return coordinateToRotate;
    }

    dvec3 alignWithZAxis_XZ(dvec3 directionVector, dvec3 coordinateToRotate) {
        // Same as the above, except rotating to the z ("up") axis
        dvec2 direction2D_not_normalised(directionVector.x, directionVector.z);

        if(direction2D_not_normalised.x == 0 && direction2D_not_normalised.y == 0) {
            return coordinateToRotate;
        }

        dvec2 direction2D = glm::normalize(direction2D_not_normalised);
        double sine = direction2D.y;
        double cosine = direction2D.x;

        const double initialTransformedX = coordinateToRotate.x;
        coordinateToRotate.x = sine * coordinateToRotate.x - cosine * coordinateToRotate.z;
        coordinateToRotate.z = cosine * initialTransformedX + sine * coordinateToRotate.z;
        return coordinateToRotate;
    }

    // Assumption: circle is at the origin
    // Assumption: unit circle
    // Assumption: parameters are normalised direction vectors
    inline double computePieWedgeArea(dvec2 startDirection, dvec2 endDirection) {
        double cosine = std::clamp(dot(startDirection, endDirection), -1.0, 1.0);
        double angle = std::acos(cosine);
        assert(angle >= 0);
        return angle / 2.0;
    }

    inline dvec2 calculateAlphaBeta(dvec3 cylinderOrigin, dvec3 cylinderDirection, dvec3 point)
    {
        double beta = dot(point - cylinderOrigin, cylinderDirection) / dot(cylinderDirection, cylinderDirection);
        dvec3 projectedPoint = cylinderOrigin + beta * cylinderDirection;
        dvec3 delta = projectedPoint - point;
        double alpha = length(delta);
        dvec2 alphabeta = {alpha, beta};
        return alphabeta;
    }

    bool edgeLiesOutsideCircle(dvec2 startVertex, dvec2 endVertex) {
        dvec2 delta = endVertex - startVertex;

        if(delta.x == 0 && delta.y == 0) {
            return false;
        }

        dvec2 deltaDirection = glm::normalize(delta);
        dvec2 startDirection = glm::normalize(startVertex);
        dvec2 endDirection = glm::normalize(endVertex);
        const dvec2 origin {0, 0};


        double denominator = dot(delta, delta);
        double scaleFactor = dot(origin - startVertex, delta) / denominator;
        //dvec2 closestPointOnEdgeToOrigin = startVertex + (scaleFactor * delta);
        bool isOutside = scaleFactor < 0 || scaleFactor > 1;
        // Intersection occurs -> closest point to origin lies on edge
        return isOutside;

        //double edgeSquaredDistanceFromOrigin = dot(closestPointOnEdgeToOrigin - origin, closestPointOnEdgeToOrigin - origin);
        //return edgeSquaredDistanceFromOrigin > 1;
    }

    // Assumption: circle is at the origin
    // Assumption: unit circle
    double computeSignedCircleIntersectionArea(dvec2 startVertex, dvec2 endVertex) {
        dvec2 delta = endVertex - startVertex;

        if(delta.x == 0 && delta.y == 0) {
            return 0;
        }

        dvec2 deltaDirection = glm::normalize(delta);
        dvec2 startDirection = glm::normalize(startVertex);
        dvec2 endDirection = glm::normalize(endVertex);
        const dvec2 origin {0, 0};

        double area = 0;

        double denominator = dot(delta, delta);
        double scaleFactor = dot(origin - startVertex, delta) / denominator;
        dvec2 closestPointOnEdgeToOrigin = startVertex + (scaleFactor * delta);
        //bool isOutside = scaleFactor < 0 || scaleFactor > 1;

        double edgeSquaredDistanceFromOrigin = dot(closestPointOnEdgeToOrigin - origin, closestPointOnEdgeToOrigin - origin);
        double squaredDistanceStartVertex = dot(startVertex - origin, startVertex - origin);
        double squaredDistanceEndVertex = dot(endVertex - origin, endVertex - origin);

        dvec3 projected = alignWithXAxis_XY(dvec3(startDirection.x, startDirection.y, 0), dvec3(endDirection.x, endDirection.y, 0));
        dvec2 relativeDirection = dvec2(projected.x, projected.y);
        bool rotatesCounterClockwise = relativeDirection.y >= 0;

        if(edgeSquaredDistanceFromOrigin > 1.0) {
            // Edge lies outside of circle in its entirety
            // Area covered is a pie wedge
            area += computePieWedgeArea(startDirection, endDirection);
        } else if(squaredDistanceStartVertex <= 1 && squaredDistanceEndVertex <= 1) {
            // Edge lies within circle in its entirety
            // Area covered is a triangle between vertices and origin
            area += ShapeDescriptor::computeTriangleArea(ShapeDescriptor::cpu::double3{0, 0, 0}, {startVertex.x, startVertex.y, 0}, {endVertex.x, endVertex.y, 0});
        } else {
            double edgeDistanceFromOrigin = std::sqrt(edgeSquaredDistanceFromOrigin);
            assert(edgeDistanceFromOrigin <= 1);
            double circularSegmentHeight = 1.0 - edgeDistanceFromOrigin;
            double halfChordLength = std::sqrt(2.0 * circularSegmentHeight - circularSegmentHeight * circularSegmentHeight);

            // This computes the distance along the edge to reach the start and end point of the vertex from the
            // closest point along the edge (works because deltaDirection is a unit vector)
            double startPositionAlongEdge = dot(deltaDirection, startVertex - closestPointOnEdgeToOrigin);
            double endPositionAlongEdge = dot(deltaDirection, endVertex - closestPointOnEdgeToOrigin);

            if((startPositionAlongEdge >= halfChordLength && endPositionAlongEdge >= halfChordLength)
            || (startPositionAlongEdge <= -halfChordLength && endPositionAlongEdge <= -halfChordLength)) {
                // Start and end coordinate lie outside the same side of the circle
                // The area is therefore the wedge
                area += computePieWedgeArea(startDirection, endDirection);
            } else {
                // Case of edge lying within circle is already covered earlier
                // We therefore now know there is at least one intersection.
                // We'll first sort the vertices from left to right along the edge to simplify logic
                if(startPositionAlongEdge > endPositionAlongEdge) {
                    std::swap(startPositionAlongEdge, endPositionAlongEdge);
                    std::swap(startVertex, endVertex);
                }

                // Compute boundary crossings. These should be unit vectors since they lie on the edge of the unit circle
                bool leftSideCrossed = startPositionAlongEdge <= -halfChordLength && endPositionAlongEdge >= -halfChordLength;
                dvec2 leftBoundaryCrossingDirection = closestPointOnEdgeToOrigin - halfChordLength * deltaDirection;
                double startDirectionLength = length(leftBoundaryCrossingDirection);
                assert(std::abs(startDirectionLength - 1) < 0.00001);
                bool rightSideCrossed = startPositionAlongEdge <= halfChordLength && endPositionAlongEdge >= halfChordLength;
                dvec2 rightBoundaryCrossingDirection = closestPointOnEdgeToOrigin + halfChordLength * deltaDirection;
                double endDirectionLength = length(rightBoundaryCrossingDirection);
                assert(std::abs(endDirectionLength - 1) < 0.00001);

                dvec2 innerTriangleStartVertex = startVertex;
                dvec2 innerTriangleEndVertex = endVertex;

                if(leftSideCrossed) {
                    // We're crossing the left circle boundary, so we need to account for the wedge area there
                    area += computePieWedgeArea(startDirection, leftBoundaryCrossingDirection);
                    innerTriangleStartVertex = leftBoundaryCrossingDirection;
                }
                if(rightSideCrossed) {
                    area += computePieWedgeArea(rightBoundaryCrossingDirection, endDirection);
                    innerTriangleEndVertex = rightBoundaryCrossingDirection;
                }

                // The parts of the edge that are outside the circle are now covered. We'll continue with the portion inside the circle
                area += ShapeDescriptor::computeTriangleArea(ShapeDescriptor::cpu::double3{0, 0, 0},
                                                             ShapeDescriptor::cpu::double3{innerTriangleStartVertex.x, innerTriangleStartVertex.y, 0},
                                                             ShapeDescriptor::cpu::double3{innerTriangleEndVertex.x, innerTriangleEndVertex.y, 0});
            }
        }

        // If direction is counter-clockwise, area is positive, otherwise negative
        if(!rotatesCounterClockwise) {
            area *= -1;
        }

        return area;
    }

    // Assumption: bottom left corner of rectangle is a (0, 0)
    // Computes area underneath an edge overlapping with a rectangle
    // Positive area if edge moves from left to right, negative otherwise
    double computeSignedRectangleIntersectionArea(dvec2 startVertex, dvec2 endVertex, double rectangleWidth, double rectangleHeight) {


        // Ensure there is a chance that an intersection could occur
        if((startVertex.x < 0 && endVertex.x < 0)
        || (startVertex.x > rectangleWidth && endVertex.x > rectangleWidth)
        || (startVertex.y <= 0 && endVertex.y <= 0)) {
            return 0;
        }
        // No area covered underneath edge
        if(startVertex.x == endVertex.x) {
            return 0;
        }
        bool edgeDirectionLeftToRight = endVertex.x > startVertex.x;
        // In a counter-clockwise
        double sign = edgeDirectionLeftToRight ? -1 : 1;
        if(!edgeDirectionLeftToRight) {
            std::swap(startVertex, endVertex);
        }

        dvec2 delta = endVertex - startVertex;

        // Clip segment to inside of rectangle
        if(startVertex.x < 0) {
            double leftSideIntersectionFactor = (0 - startVertex.x) / delta.x;
            assert(leftSideIntersectionFactor >= 0);
            assert(leftSideIntersectionFactor <= 1);
            startVertex = startVertex + leftSideIntersectionFactor * delta;
            delta = endVertex - startVertex;
        }
        if(endVertex.x > rectangleWidth) {
            double rightSideIntersectionFactor = (rectangleWidth - startVertex.x) / delta.x;
            assert(rightSideIntersectionFactor >= 0);
            assert(rightSideIntersectionFactor <= 1);
            endVertex = startVertex + rightSideIntersectionFactor * delta;
            delta = endVertex - startVertex;
        }

        // We now have an edge whose x coordinates are between 0 and width
        if(startVertex.y <= 0 && endVertex.y <= 0) {
            return 0;
        }

        // Covers entire rectangle. Area is covered area of rectangle
        if(startVertex.y >= rectangleHeight && endVertex.y >= rectangleHeight) {
            return sign * (endVertex.x - startVertex.x) * rectangleHeight;
        }
        if(delta.y == 0) {
            double height = std::min(startVertex.y, rectangleHeight);
            return sign * delta.x * height;
        }

        // Next clipping step: y=0 crossing
        if(startVertex.y * endVertex.y < 0) {
            double deltaY = 0 - startVertex.y;
            double scaleFactor = deltaY / delta.y;
            assert(scaleFactor >= 0);
            assert(scaleFactor <= 1);
            dvec2 intersectionPoint = startVertex + scaleFactor * delta;
            if(startVertex.y > 0) {
                endVertex = intersectionPoint;
            } else {
                startVertex = intersectionPoint;
            }
        }

        double totalArea = 0;

        // y=height crossing. Can check this way because if both are above y=height a previous if would've caught them
        // One side will cover the rectangle completely, the other will cut it.
        if(startVertex.y > rectangleHeight || endVertex.y > rectangleHeight) {
            double deltaY = rectangleHeight - startVertex.y;
            double scaleFactor = deltaY / delta.y;
            assert(scaleFactor >= 0);
            assert(scaleFactor <= 1);
            dvec2 intersectionPoint = startVertex + scaleFactor * delta;
            if(startVertex.y > rectangleHeight) {
                totalArea += (intersectionPoint.x - startVertex.x) * rectangleHeight;
                startVertex = intersectionPoint;
            } else {
                totalArea += (endVertex.x - intersectionPoint.x) * rectangleHeight;
                endVertex = intersectionPoint;
            }
        }
        assert(totalArea >= 0);

        // Edge should now be within the rectangle bounds
        const double allowableDelta = 0.0000001;
        assert(startVertex.x >= -allowableDelta);
        assert(endVertex.x >= -allowableDelta);
        assert(startVertex.x <= rectangleWidth + allowableDelta);
        assert(endVertex.x <= rectangleWidth + allowableDelta);
        assert(startVertex.y >= -allowableDelta);
        assert(endVertex.y >= -allowableDelta);
        assert(startVertex.y <= rectangleHeight + allowableDelta);
        assert(endVertex.y <= rectangleHeight + allowableDelta);

        // Covers the triangle part of the area underneath the edge
        totalArea += 0.5f * (endVertex.x - startVertex.x) * std::abs(endVertex.y - startVertex.y);

        // Covers the rectangle part of the area underneath the edge
        totalArea += (endVertex.x - startVertex.x) * std::min(startVertex.y, endVertex.y);

        assert(totalArea >= -0.0001);
        assert(totalArea <= rectangleWidth * rectangleHeight + 0.0001);

        return sign * totalArea;
    }

    inline ShapeBench::Triangle2D toTriangle2D(ShapeBench::Triangle3D triangle) {
        return {
                {triangle.vertex0.x, triangle.vertex0.y},
                {triangle.vertex1.x, triangle.vertex1.y},
                {triangle.vertex2.x, triangle.vertex2.y}
        };
    }

    ShapeDescriptor::cpu::double3 convert(dvec3 vector) {
        return {vector.x, vector.y, vector.z};
    }

    void clipTriangleAgainstPlane(dvec3 vertex0, dvec3 vertex1, dvec3 vertex2,
                                  double cylinderBoundingPlaneZ,
                                  bool vertexOutside0, bool vertexOutside1, bool vertexOutside2,
                                  std::array<ShapeBench::Triangle3D, 2>& out_clippedTriangles,
                                  int& out_clippedTriangleCount) {

        // At most one of these can be true because we filtered out the case where they are all true
        bool twoVerticesAbove01 = vertexOutside0 && vertexOutside1;
        bool twoVerticesAbove12 = vertexOutside1 && vertexOutside2;
        bool twoVerticesAbove20 = vertexOutside2 && vertexOutside0;

        bool isTriangleWithTwoVerticesAbovePlane = false;

        // Triangle rotation to get vertices in a consistent order
        // Convention: edges 01 and 12 intersect with upper boundary plane
        // - Result: two vertices above plane -> vertex 0 and 2 must be above
        // - Result: one vertex above plane -> vertex 1 must be above
        if(twoVerticesAbove01) {
            std::swap(vertex1, vertex2);
            isTriangleWithTwoVerticesAbovePlane = true;
        } else if(twoVerticesAbove12) {
            std::swap(vertex0, vertex1);
            isTriangleWithTwoVerticesAbovePlane = true;
        } else if(twoVerticesAbove20) {
            // This is what we want, so no change needed
            // Still need this case to ensure we have checked for two vertices above cases
            // before moving on to single vertex above cases
            isTriangleWithTwoVerticesAbovePlane = true;
        } else if(vertexOutside0) {
            std::swap(vertex0, vertex1);
        } else if(vertexOutside2) {
            std::swap(vertex2, vertex1);
        } else if(vertexOutside1) {
            // Also what we want, in this case there's only one vertex sticking out
        } else {
            throw std::runtime_error("This should be logically impossible!");
        }

        // Reminder to myself: variables referring to vertex0 - vertex2 are invalidated at this point

        // We can now assume that edges 01 and 12 contain crossings with boundary plane
        // Next we calculate the intersection points with this plane
        // There can be no boundary crossing without a z-coordinate difference.
        // We therefore don't need to worry about height differences of 0
        dvec3 edgeDelta10 = vertex0 - vertex1;
        dvec3 edgeDelta12 = vertex2 - vertex1;
        assert(edgeDelta10.z != 0);
        assert(edgeDelta12.z != 0);
        double scaleFactor10 = (cylinderBoundingPlaneZ - vertex1.z) / edgeDelta10.z;
        double scaleFactor12 = (cylinderBoundingPlaneZ - vertex1.z) / edgeDelta12.z;
        assert(0 <= scaleFactor10 && scaleFactor10 <= 1);
        assert(0 <= scaleFactor12 && scaleFactor12 <= 1);
        dvec3 intersectionPointEdge01 = vertex1 + scaleFactor10 * edgeDelta10;
        dvec3 intersectionPointEdge12 = vertex1 + scaleFactor12 * edgeDelta12;

        // Two vertices above requires two clipped triangles
        // One vertex above plane requires only one
        if(isTriangleWithTwoVerticesAbovePlane) {
            Triangle3D clippedTriangle0;
            clippedTriangle0.vertex0 = convert(vertex0);
            clippedTriangle0.vertex1 = convert(intersectionPointEdge01);
            clippedTriangle0.vertex2 = convert(vertex2);

            Triangle3D clippedTriangle1;
            clippedTriangle1.vertex0 = convert(vertex2);
            clippedTriangle1.vertex1 = convert(intersectionPointEdge01);
            clippedTriangle1.vertex2 = convert(intersectionPointEdge12);

            out_clippedTriangleCount = 2;
            out_clippedTriangles.at(0) = clippedTriangle0;
            out_clippedTriangles.at(1) = clippedTriangle1;

            //  return  + computeAreaOfIntersection(circle, clippedTriangle1);
        } else {
            Triangle3D clippedTriangle0;
            clippedTriangle0.vertex0 = convert(vertex1);
            clippedTriangle0.vertex1 = convert(intersectionPointEdge01);
            clippedTriangle0.vertex2 = convert(intersectionPointEdge12);

            out_clippedTriangleCount = 1;
            out_clippedTriangles.at(0) = clippedTriangle0;
            //     return computeAreaOfIntersection(circle, clippedTriangle0);
        }
    }


}

void changeWindingOrderToCounterClockwise(ShapeBench::Triangle2D& triangle, bool& isDegenerate) {
    // Area calculation method assumes counter-clockwise winding.
    // Earlier transformations don't guarantee this.
    glm::mat3x3 windingOrderMatrix {triangle.vertex0.x, triangle.vertex1.x, triangle.vertex2.x,
                                    triangle.vertex0.y, triangle.vertex1.y, triangle.vertex2.y,
                                    1, 1, 1};
    double windingOrderDeterminant = glm::determinant(windingOrderMatrix);
    // Indicates degenerate triangle
    isDegenerate = windingOrderDeterminant == 0;
    if(windingOrderDeterminant < 0) {
        // Winding order is clockwise: swap vertices to flip order
        std::swap(triangle.vertex1, triangle.vertex2);
    }
}


double ShapeBench::computeAreaOfIntersection(ShapeBench::Circle2D circle, ShapeBench::Triangle2D triangle) {
    bool isDegenerate = false;
    changeWindingOrderToCounterClockwise(triangle, isDegenerate);
    if(isDegenerate) {
        // a tiny triangle can become degenerate over the course of the calculation sequence
        return 0;
    }

    // Move circle to origin. It is scaled because the area algorithm assumes a unit circle
    double triangleScaleFactor = 1.0 / circle.radius;
    dvec2 origin = {circle.origin.x, circle.origin.y};
    dvec2 d0 = triangleScaleFactor * (dvec2(triangle.vertex0.x, triangle.vertex0.y) - origin);
    dvec2 d1 = triangleScaleFactor * (dvec2(triangle.vertex1.x, triangle.vertex1.y) - origin);
    dvec2 d2 = triangleScaleFactor * (dvec2(triangle.vertex2.x, triangle.vertex2.y) - origin);

    double area = computeSignedCircleIntersectionArea(d0, d1);
    area += computeSignedCircleIntersectionArea(d1, d2);
    area += computeSignedCircleIntersectionArea(d2, d0);

    // We scaled in two dimensions, therefore area must be scaled by the square of the computed area.
    area *= circle.radius * circle.radius;
    return area;
}

inline dvec3 alignWithXYPlane(dvec3 direction, dvec3 point) {
    dvec3 rotatedPoint = ShapeBench::alignWithXAxis_XY(direction, point);
    dvec3 rotatedDirection = ShapeBench::alignWithXAxis_XY(direction, direction);
    return ShapeBench::alignWithZAxis_XZ(rotatedDirection, rotatedPoint);
}


double ShapeBench::computeAreaOfIntersection(ShapeBench::Sphere sphere, ShapeBench::Triangle3D triangle) {
    const double parallelLineErrorMargin = 0.0001;
    const double inputTriangleArea = ShapeDescriptor::computeTriangleArea(triangle.vertex0, triangle.vertex1, triangle.vertex2);
    // Handle degenerate triangles
    if(inputTriangleArea == 0) {
        return 0;
    }

    dvec3 sphereOrigin = {sphere.centrePoint.x, sphere.centrePoint.y, sphere.centrePoint.z};
    dvec3 triangleVertex0 = {triangle.vertex0.x, triangle.vertex0.y, triangle.vertex0.z};
    dvec3 triangleVertex1 = {triangle.vertex1.x, triangle.vertex1.y, triangle.vertex1.z};
    dvec3 triangleVertex2 = {triangle.vertex2.x, triangle.vertex2.y, triangle.vertex2.z};

    // Step 1: Transform triangle to be on the xy plane
    const dvec3 translationToOrigin = triangleVertex0;
    sphereOrigin -= translationToOrigin;
    triangleVertex0 -= translationToOrigin;
    triangleVertex1 -= translationToOrigin;
    triangleVertex2 -= translationToOrigin;

    // Check for valid (nonzero area) triangle. If triangle has 0 area, intersection area is also 0
    // We're dealing with relative coordinates here to triangleVertex0, so we can check these directly
    // Note: this works because we have moved one of the vertices to the origin
    assert(triangle.vertex0 != triangle.vertex1);
    assert(triangle.vertex1 != triangle.vertex2);
    assert(triangle.vertex2 != triangle.vertex0);

    // Any valid triangle should have a valid normal vector
    dvec3 triangleNormal = glm::triangleNormal(triangleVertex0, triangleVertex1, triangleVertex2);
    assert(triangleNormal.x != 0 || triangleNormal.y != 0 || triangleNormal.z != 0);

    triangleVertex1 = alignWithXYPlane(triangleNormal, triangleVertex1);
    assert(std::abs(triangleVertex1.z) < parallelLineErrorMargin);
    triangleVertex2 = alignWithXYPlane(triangleNormal, triangleVertex2);
    assert(std::abs(triangleVertex2.z) < parallelLineErrorMargin);
    sphereOrigin = alignWithXYPlane(triangleNormal, sphereOrigin);

    // Sphere is too far away from the plane to intersect
    if(std::abs(sphereOrigin.z) > sphere.radius) {
        return 0;
    }

    dvec3 sphereCentre = {sphereOrigin.x, sphereOrigin.y, 0};
    triangleVertex0 -= sphereCentre;
    triangleVertex1 -= sphereCentre;
    triangleVertex2 -= sphereCentre;

    double chordHeight = sphere.radius - sphereOrigin.z;
    double circleRadius = std::sqrt(2.0 * sphere.radius * chordHeight - chordHeight * chordHeight);

    ShapeBench::Circle2D circle;
    circle.origin = {0, 0};
    circle.radius = circleRadius;

    ShapeBench::Triangle2D intersectionBaseTriangle;
    intersectionBaseTriangle.vertex0 = {triangleVertex0.x, triangleVertex0.y};
    intersectionBaseTriangle.vertex1 = {triangleVertex1.x, triangleVertex1.y};
    intersectionBaseTriangle.vertex2 = {triangleVertex2.x, triangleVertex2.y};

    double circleTriangleIntersectionArea = computeAreaOfIntersection(circle, intersectionBaseTriangle);

    return circleTriangleIntersectionArea;
}


double ShapeBench::computeAreaOfIntersection(ShapeBench::Cylinder cylinder, ShapeBench::Triangle3D triangle) {

#define enableRenderers false
#if enableRenderers
    ShapeBench::DebugRenderer renderer;
#endif

    const double inputTriangleArea = ShapeDescriptor::computeTriangleArea(triangle.vertex0, triangle.vertex1, triangle.vertex2);
    // Handle degenerate triangles
    if(inputTriangleArea == 0) {
        return 0;
    }

    // Larger error margins also avoid numerical stability issues
    const double parallelLineErrorMargin = 0.0001;
    const dvec3 zAxis {0, 0, 1};
    const dvec3 xAxis {1, 0, 0};


    cylinder.normalisedDirection = normalize(cylinder.normalisedDirection);
    dvec3 cylinderOrigin = {cylinder.centrePoint.x, cylinder.centrePoint.y, cylinder.centrePoint.z};
    dvec3 cylinderDirection = {cylinder.normalisedDirection.x, cylinder.normalisedDirection.y, cylinder.normalisedDirection.z};
    dvec3 triangleVertex0 = {triangle.vertex0.x, triangle.vertex0.y, triangle.vertex0.z};
    dvec3 triangleVertex1 = {triangle.vertex1.x, triangle.vertex1.y, triangle.vertex1.z};
    dvec3 triangleVertex2 = {triangle.vertex2.x, triangle.vertex2.y, triangle.vertex2.z};

    // Handle easy cases where the triangle is either entirely inside or outside the cylinder
    // Saves both on accuracy and computation time
    dvec2 alphaBetaVertex0 = ShapeBench::calculateAlphaBeta(cylinderOrigin, cylinderDirection, triangleVertex0);
    dvec2 alphaBetaVertex1 = ShapeBench::calculateAlphaBeta(cylinderOrigin, cylinderDirection, triangleVertex1);
    dvec2 alphaBetaVertex2 = ShapeBench::calculateAlphaBeta(cylinderOrigin, cylinderDirection, triangleVertex2);

    bool belowBottomPlaneVertex0 = alphaBetaVertex0.y < -cylinder.halfOfHeight;
    bool belowBottomPlaneVertex1 = alphaBetaVertex1.y < -cylinder.halfOfHeight;
    bool belowBottomPlaneVertex2 = alphaBetaVertex2.y < -cylinder.halfOfHeight;
    bool allBelowBottomPlane = belowBottomPlaneVertex0 && belowBottomPlaneVertex1 && belowBottomPlaneVertex2;
    bool oneBelowBottomPLane = belowBottomPlaneVertex0 || belowBottomPlaneVertex1 || belowBottomPlaneVertex2;

    if(allBelowBottomPlane) {
        return 0;
    }
    bool aboveTopPlaneVertex0 = alphaBetaVertex0.y > cylinder.halfOfHeight;
    bool aboveTopPlaneVertex1 = alphaBetaVertex1.y > cylinder.halfOfHeight;
    bool aboveTopPlaneVertex2 = alphaBetaVertex2.y > cylinder.halfOfHeight;
    bool allAboveTopPlane = aboveTopPlaneVertex0 && aboveTopPlaneVertex1 && aboveTopPlaneVertex2;
    bool oneAboveTopPlane = aboveTopPlaneVertex0 || aboveTopPlaneVertex1 || aboveTopPlaneVertex2;
    if(allAboveTopPlane) {
        return 0;
    }
    bool allWithinCylinderRadius = alphaBetaVertex0.x <= cylinder.radius
                                && alphaBetaVertex1.x <= cylinder.radius
                                && alphaBetaVertex2.x <= cylinder.radius;
    bool allInsideCylinder = allWithinCylinderRadius && !oneAboveTopPlane && !oneBelowBottomPLane;
    if(allInsideCylinder) {
        return inputTriangleArea;
    }

    if(allWithinCylinderRadius) {
        // We know there is an intersection with either the top or bottom plane,
        // and all triangles are within the "infinite" cylinder shape
        // It is therefore possible to calculate the intersecting area by clipping the triangle
        double intersectingArea = inputTriangleArea;

        std::array<ShapeBench::Triangle3D, 2> clippedTriangles;
        int clippedTriangleCount = 0;

        // Move cylinder to origin and align direction vector with z-axis to be able to reuse
        // clipping algorithm
        triangleVertex0 -= cylinderOrigin;
        triangleVertex1 -= cylinderOrigin;
        triangleVertex2 -= cylinderOrigin;

        triangleVertex0 = alignWithXYPlane(cylinderDirection, triangleVertex0);
        triangleVertex1 = alignWithXYPlane(cylinderDirection, triangleVertex1);
        triangleVertex2 = alignWithXYPlane(cylinderDirection, triangleVertex2);

        if(oneAboveTopPlane) {
            clipTriangleAgainstPlane(triangleVertex0, triangleVertex1, triangleVertex2, cylinder.halfOfHeight,
                                     aboveTopPlaneVertex0, aboveTopPlaneVertex1, aboveTopPlaneVertex2,
                                     clippedTriangles, clippedTriangleCount);
            intersectingArea -= ShapeDescriptor::computeTriangleArea(clippedTriangles.at(0).vertex0, clippedTriangles.at(0).vertex1, clippedTriangles.at(0).vertex2);
            if(clippedTriangleCount == 2) {
                intersectingArea -= ShapeDescriptor::computeTriangleArea(clippedTriangles.at(1).vertex0, clippedTriangles.at(1).vertex1, clippedTriangles.at(1).vertex2);
            }
        }

        if(oneBelowBottomPLane) {
            clipTriangleAgainstPlane(triangleVertex0, triangleVertex1, triangleVertex2, -cylinder.halfOfHeight,
                                     belowBottomPlaneVertex0, belowBottomPlaneVertex1, belowBottomPlaneVertex2,
                                     clippedTriangles, clippedTriangleCount);
            intersectingArea -= ShapeDescriptor::computeTriangleArea(clippedTriangles.at(0).vertex0, clippedTriangles.at(0).vertex1, clippedTriangles.at(0).vertex2);
            if(clippedTriangleCount == 2) {
                intersectingArea -= ShapeDescriptor::computeTriangleArea(clippedTriangles.at(1).vertex0, clippedTriangles.at(1).vertex1, clippedTriangles.at(1).vertex2);
            }
        }

        return intersectingArea;
    }


    // Scaling down to approximately unit lengths (or at least close to) helps with numerical accuracy
    double originalCylinderRadius = cylinder.radius;
    double sceneScaleFactor = 1.0 / originalCylinderRadius;
    cylinder.radius = 1;
    cylinder.halfOfHeight = cylinder.halfOfHeight * sceneScaleFactor;
    cylinderOrigin = sceneScaleFactor * cylinderOrigin;
    triangleVertex0 = sceneScaleFactor * triangleVertex0;
    triangleVertex1 = sceneScaleFactor * triangleVertex1;
    triangleVertex2 = sceneScaleFactor * triangleVertex2;


#if enableRenderers
    renderer.drawCylinder(cylinderOrigin, cylinderDirection, cylinder.halfOfHeight, cylinder.radius);
    renderer.drawTriangle(triangleVertex0, triangleVertex1, triangleVertex2);
    renderer.drawLine(cylinderOrigin, cylinderOrigin + cylinderDirection * double(cylinder.halfOfHeight), {1, 1, 0});
    renderer.show("Initial setup");
#endif

    if(cylinderDirection.x == 0 && cylinderDirection.y == 0 && cylinderDirection.z == 0) {
        // This should never happen, but can happen i*n case of a bad normal vector.
        return 0;
    }
    cylinderDirection = glm::normalize(cylinderDirection);

    // Step 1: Transform triangle to be on the xy plane
    const dvec3 translationToOrigin = triangleVertex0;
    cylinderOrigin -= translationToOrigin;
    triangleVertex0 -= translationToOrigin;
    triangleVertex1 -= translationToOrigin;
    triangleVertex2 -= translationToOrigin;

#if enableRenderers
    renderer.drawCylinder(cylinderOrigin, cylinderDirection, cylinder.halfOfHeight, cylinder.radius);
    renderer.drawTriangle(triangleVertex0, triangleVertex1, triangleVertex2);
    renderer.drawLine(cylinderOrigin, cylinderOrigin + cylinderDirection * double(cylinder.halfOfHeight), {1, 1, 0});
    renderer.show("translatedToOrigin");
#endif

    // Check for valid (nonzero area) triangle. If triangle has 0 area, intersection area is also 0
    // We're dealing with relative coordinates here to triangleVertex0, so we can check these directly
    // Note: this works because we have moved one of the vertices to the origin
    assert(triangleVertex0 != triangleVertex1);
    assert(triangleVertex1 != triangleVertex2);
    assert(triangleVertex2 != triangleVertex0);

    // Any valid triangle should have a valid normal vector
    dvec3 triangleNormal = glm::triangleNormal(triangleVertex0, triangleVertex1, triangleVertex2);
    assert(triangleNormal.x != 0 || triangleNormal.y != 0 || triangleNormal.z != 0);


#if enableRenderers
    renderer.drawCylinder(cylinderOrigin, cylinderDirection, cylinder.halfOfHeight, cylinder.radius);
    renderer.drawTriangle(triangleVertex0, triangleVertex1, triangleVertex2);
    renderer.drawLine(cylinderOrigin, cylinderOrigin + cylinderDirection * double(cylinder.halfOfHeight), {1, 1, 0});
    //renderer.drawLine({0, 0, 0}, orthogonalDirection * double(cylinder.halfOfHeight), {0, 1, 1});
    renderer.drawLine({0, 0, 0}, triangleNormal * double(cylinder.halfOfHeight), {1, 1, 1});
    renderer.show("Pre transform");
#endif

    triangleVertex1 = alignWithXYPlane(triangleNormal, triangleVertex1);
    assert(std::abs(triangleVertex1.z) < parallelLineErrorMargin);
    triangleVertex2 = alignWithXYPlane(triangleNormal, triangleVertex2);
    assert(std::abs(triangleVertex2.z) < parallelLineErrorMargin);
    cylinderOrigin = alignWithXYPlane(triangleNormal, cylinderOrigin);
    cylinderDirection = alignWithXYPlane(triangleNormal, cylinderDirection);

#if enableRenderers
    triangleNormal = glm::triangleNormal(triangleVertex0, triangleVertex1, triangleVertex2);
    renderer.drawCylinder(cylinderOrigin, cylinderDirection, cylinder.halfOfHeight, cylinder.radius);
    renderer.drawTriangle(triangleVertex0, triangleVertex1, triangleVertex2);
    renderer.drawLine(cylinderOrigin, cylinderOrigin + cylinderDirection * double(cylinder.halfOfHeight), {1, 1, 0});
    //renderer.drawLine({0, 0, 0}, orthogonalDirection * double(cylinder.halfOfHeight), {0, 1, 1});
    renderer.drawLine({0, 0, 0}, triangleNormal * double(cylinder.halfOfHeight), {1, 1, 1});
    renderer.show("Post transformation");
#endif

    // We now have a 2D problem
    // Step 2: locate intersection point of line described by cylinder normal and cylinder origin
    // Edge case: if line parallel to xy plane, problem becomes a rectangle triangle intersection problem, with the width of the rectangle described by the z coordinate of the cylinder normal
    // Constant is a tradeoff between accuracy of calculated area and numerical stability further down the line
    if(std::abs(cylinderDirection.z) < 0.1) {
        // No intersection
        if(cylinderOrigin.z > cylinder.radius || cylinderOrigin.z < -cylinder.radius) {
            return 0;
        }

        // Make rectangle horizontal. Vertex 0 is still at the origin
        // Also note that z coordinates are 0 of the triangle vertices
        cylinderOrigin = alignWithXAxis_XY(cylinderDirection, cylinderOrigin);
        triangleVertex0 = triangleVertex0;
        triangleVertex1 = alignWithXAxis_XY(cylinderDirection, triangleVertex1);
        triangleVertex2 = alignWithXAxis_XY(cylinderDirection, triangleVertex2);
        cylinderDirection = {1, 0, 0};
#if enableRenderers
        renderer.drawCylinder(cylinderOrigin, cylinderDirection, cylinder.halfOfHeight, cylinder.radius);
        renderer.drawTriangle(triangleVertex0, triangleVertex1, triangleVertex2);
        renderer.drawLine(cylinder.centrePoint, cylinder.centrePoint + cylinder.normalisedDirection * cylinder.halfOfHeight);
        renderer.show("Rectangle cross-section");
#endif

        double rectangleLeft = cylinderOrigin.x - cylinder.halfOfHeight;
        // Chord length of a circular segment due to this rectangle being an intersection with a parallel cylinder
        double rectangleXYPlaneIntersectionDistance = cylinder.radius - std::abs(cylinderOrigin.z);
        double rectangleHalfHeight = std::sqrt(2 * cylinder.radius * rectangleXYPlaneIntersectionDistance - rectangleXYPlaneIntersectionDistance * rectangleXYPlaneIntersectionDistance);
        double rectangleBottom = cylinderOrigin.y - rectangleHalfHeight;

        // Move rectangle to the origin, with width of 2 * cylinder.halfOfHeight and height 2 * rectangleHalfHeight
        cylinderOrigin.x -= rectangleLeft;
        triangleVertex0.x -= rectangleLeft;
        triangleVertex1.x -= rectangleLeft;
        triangleVertex2.x -= rectangleLeft;

        cylinderOrigin.y -= rectangleBottom;
        triangleVertex0.y -= rectangleBottom;
        triangleVertex1.y -= rectangleBottom;
        triangleVertex2.y -= rectangleBottom;

        ShapeBench::Triangle2D intersectionPlaneTriangle;
        intersectionPlaneTriangle.vertex0 = {triangleVertex0.x, triangleVertex0.y};
        intersectionPlaneTriangle.vertex1 = {triangleVertex1.x, triangleVertex1.y};
        intersectionPlaneTriangle.vertex2 = {triangleVertex2.x, triangleVertex2.y};

        bool isDegenerate = false;
        changeWindingOrderToCounterClockwise(intersectionPlaneTriangle, isDegenerate);
        if(isDegenerate) {
            return 0;
        }

        // Area is positive if direction of edge is to the right
        double area = ShapeBench::computeSignedRectangleIntersectionArea({intersectionPlaneTriangle.vertex0.x, intersectionPlaneTriangle.vertex0.y},
                                                                         {intersectionPlaneTriangle.vertex1.x, intersectionPlaneTriangle.vertex1.y},
                                                                        2 * cylinder.halfOfHeight, 2 * rectangleHalfHeight);
        area += ShapeBench::computeSignedRectangleIntersectionArea({intersectionPlaneTriangle.vertex1.x, intersectionPlaneTriangle.vertex1.y},
                                                                   {intersectionPlaneTriangle.vertex2.x, intersectionPlaneTriangle.vertex2.y},
                                                                   2 * cylinder.halfOfHeight, 2 * rectangleHalfHeight);
        area += ShapeBench::computeSignedRectangleIntersectionArea({intersectionPlaneTriangle.vertex2.x, intersectionPlaneTriangle.vertex2.y},
                                                                    {intersectionPlaneTriangle.vertex0.x, intersectionPlaneTriangle.vertex0.y},
                                                                   2 * cylinder.halfOfHeight, 2 * rectangleHalfHeight);
        assert(area >= -0.0000001);
#if enableRenderers
        // Easier than sorting edges by height
        renderer.waitForClose();
#endif

        // Since we're a bit lenient on what qualifies as as an orthogonal triangle, we do a scale here to approximate
        // the slightly rotated triangle
        double cosine = std::abs(cylinderDirection.z);
        double sine = std::sqrt(1 - cosine * cosine);
        return area * originalCylinderRadius * originalCylinderRadius * (1.0 / sine);
    }

    double cylinderZeroIntersectionFactor = -cylinderOrigin.z / cylinderDirection.z;
    dvec3 cylinderAxisIntersectionPoint = cylinderOrigin + double(cylinderZeroIntersectionFactor) * cylinderDirection;
    assert(std::abs(cylinderAxisIntersectionPoint.z) < parallelLineErrorMargin);

    // Step 3: Move triangle and cylinder such that the xy plane intersection point is at the origin
    triangleVertex0 -= cylinderAxisIntersectionPoint;
    triangleVertex1 -= cylinderAxisIntersectionPoint;
    triangleVertex2 -= cylinderAxisIntersectionPoint;
    cylinderOrigin -= cylinderAxisIntersectionPoint;

    double areaScaleFactor = 1 * originalCylinderRadius * originalCylinderRadius;

    dvec3 rotatedTriangleVertex0 = triangleVertex0;
    dvec3 rotatedTriangleVertex1 = triangleVertex1;
    dvec3 rotatedTriangleVertex2 = triangleVertex2;

    // Step 4: Intersection with cylinder and cylinder is an oval. The radius in one direction is the radius, the other direction is radius * scale factor dependent on direction.
    // Align the direction of the intersection oval with a major axis
    // Only necessary when the cylinder direction vector is not aligned with the z-axis, as in that case the intersection
    // of the cylinder with the xy plane is a circle anyway
    if(cylinderDirection.x != 0 || cylinderDirection.y != 0) {
        dvec3 partiallyRotatedTriangleVertex0 = alignWithXAxis_XY(cylinderDirection, triangleVertex0);
        dvec3 partiallyRotatedTriangleVertex1 = alignWithXAxis_XY(cylinderDirection, triangleVertex1);
        dvec3 partiallyRotatedTriangleVertex2 = alignWithXAxis_XY(cylinderDirection, triangleVertex2);
        cylinderOrigin = alignWithXAxis_XY(cylinderDirection, cylinderOrigin);
        dvec3 rotatedCylinderDirection = alignWithXAxis_XY(cylinderDirection, cylinderDirection);

#if enableRenderers
        renderer.drawCylinder(cylinderOrigin, rotatedCylinderDirection, cylinder.halfOfHeight, cylinder.radius);
        renderer.drawTriangle(partiallyRotatedTriangleVertex0, partiallyRotatedTriangleVertex1, partiallyRotatedTriangleVertex2);
        renderer.show("Aligning cylinder direction with z-axis step 1");
#endif

        // This implicitly calculates the dot product between the direction vector and the z-axis to get a cosine
        // When the direction is orthogonal to the z-axis, the intersection of the cylinder approaches a rectangle
        // That case should have been filtered out at this point
        assert(rotatedCylinderDirection.z != 0);
        areaScaleFactor *= 1.0 / std::abs(rotatedCylinderDirection.z);

        // We now rotate around the y-axis to align the direction of the cylinder with the z-axis
        rotatedTriangleVertex0 = alignWithZAxis_XZ(rotatedCylinderDirection, partiallyRotatedTriangleVertex0);
        rotatedTriangleVertex1 = alignWithZAxis_XZ(rotatedCylinderDirection, partiallyRotatedTriangleVertex1);
        rotatedTriangleVertex2 = alignWithZAxis_XZ(rotatedCylinderDirection, partiallyRotatedTriangleVertex2);
        cylinderOrigin = alignWithZAxis_XZ(rotatedCylinderDirection, cylinderOrigin);
        rotatedCylinderDirection = alignWithZAxis_XZ(rotatedCylinderDirection, rotatedCylinderDirection);
#if enableRenderers
        renderer.drawCylinder(cylinderOrigin, rotatedCylinderDirection, cylinder.halfOfHeight, cylinder.radius);
        renderer.drawTriangle(rotatedTriangleVertex0, rotatedTriangleVertex1, rotatedTriangleVertex2);
        renderer.show("Aligning cylinder direction with z-axis step 2");
#endif
    }

    // Step 5: Compute the area of intersection between a circle and triangle.
    // And subtract the area of the triangle that exceeds both ends of the cylinder
    // Should be a matter of locating the intersection of the cylinder end planes with the xy plane,
    // and computing the area that sticks out on each side and subtracting it from the total area

    // Currently this is the xy plane intersection point -> cylinder origin vector
    double cylinderOriginZOffset = cylinderOrigin.z;
    double cylinderZLowerBound = cylinderOriginZOffset - cylinder.halfOfHeight;
    double cylinderZUpperBound = cylinderOriginZOffset + cylinder.halfOfHeight;

    bool allAboveUpperBound = rotatedTriangleVertex0.z > cylinderZUpperBound
                           && rotatedTriangleVertex1.z > cylinderZUpperBound
                           && rotatedTriangleVertex2.z > cylinderZUpperBound;
    bool atLeastOneAboveUpperBound = rotatedTriangleVertex0.z > cylinderZUpperBound
                                  || rotatedTriangleVertex1.z > cylinderZUpperBound
                                  || rotatedTriangleVertex2.z > cylinderZUpperBound;
    bool allBelowUpperBound = rotatedTriangleVertex0.z <= cylinderZUpperBound
                              && rotatedTriangleVertex1.z <= cylinderZUpperBound
                              && rotatedTriangleVertex2.z <= cylinderZUpperBound;
    bool allBelowLowerBound = rotatedTriangleVertex0.z < cylinderZLowerBound
                           && rotatedTriangleVertex1.z < cylinderZLowerBound
                           && rotatedTriangleVertex2.z < cylinderZLowerBound;
    bool atLeastOneBelowLowerBound = rotatedTriangleVertex0.z < cylinderZLowerBound
                                  || rotatedTriangleVertex1.z < cylinderZLowerBound
                                  || rotatedTriangleVertex2.z < cylinderZLowerBound;
    bool allAboveLowerBound = rotatedTriangleVertex0.z >= cylinderZLowerBound
                              && rotatedTriangleVertex1.z >= cylinderZLowerBound
                              && rotatedTriangleVertex2.z >= cylinderZLowerBound;

    // Filter out triangles that don't intersect at all
    if(allAboveUpperBound || allBelowLowerBound) {
        return 0;
    }


    // Finally, calculate the total area
    ShapeBench::Circle2D circle;
    circle.origin = {0, 0};
    circle.radius = cylinder.radius;

    ShapeBench::Triangle2D intersectionBaseTriangle;
    intersectionBaseTriangle.vertex0 = {rotatedTriangleVertex0.x, rotatedTriangleVertex0.y};
    intersectionBaseTriangle.vertex1 = {rotatedTriangleVertex1.x, rotatedTriangleVertex1.y};
    intersectionBaseTriangle.vertex2 = {rotatedTriangleVertex2.x, rotatedTriangleVertex2.y};

#if enableRenderers
    renderer.drawCylinder({0, 0, 0}, {0, 0, 1}, 0, cylinder.radius);
    renderer.drawTriangle(intersectionBaseTriangle.vertex0, intersectionBaseTriangle.vertex1, intersectionBaseTriangle.vertex2);
    renderer.show("2D area calculation");
#endif

    // Area of triangles that are entirely in the cylinder can be calculated directly for better numerical stability
    assert(cylinder.radius == 1);
    bool vertexInBounds0 = dot(intersectionBaseTriangle.vertex0, intersectionBaseTriangle.vertex0) <= 1;
    bool vertexInBounds1 = dot(intersectionBaseTriangle.vertex1, intersectionBaseTriangle.vertex1) <= 1;
    bool vertexInBounds2 = dot(intersectionBaseTriangle.vertex2, intersectionBaseTriangle.vertex2) <= 1;
    if(allAboveLowerBound && allBelowUpperBound) {
        if(vertexInBounds0 && vertexInBounds1 && vertexInBounds2) {
#if enableRenderers
            renderer.waitForClose();
#endif
            // Use original coordinates for better accuracy
            return ShapeDescriptor::computeTriangleArea(triangle.vertex0, triangle.vertex1, triangle.vertex2);
        }
    }
    if(!vertexInBounds0 && !vertexInBounds1 && !vertexInBounds2) {
        // Check if triangle lies entirely outside circle -> area is 0
        // Not doing this yields effectively noise as total area
        double triangleScaleFactor = 1.0 / circle.radius;
        dvec2 origin = {circle.origin.x, circle.origin.y};
        dvec2 d0 = triangleScaleFactor * (dvec2(intersectionBaseTriangle.vertex0.x, intersectionBaseTriangle.vertex0.y) - origin);
        dvec2 d1 = triangleScaleFactor * (dvec2(intersectionBaseTriangle.vertex1.x, intersectionBaseTriangle.vertex1.y) - origin);
        dvec2 d2 = triangleScaleFactor * (dvec2(intersectionBaseTriangle.vertex2.x, intersectionBaseTriangle.vertex2.y) - origin);

        bool edgeOutsideCircle0 = edgeLiesOutsideCircle(d0, d1);
        bool edgeOutsideCircle1 = edgeLiesOutsideCircle(d1, d2);
        bool edgeOutsideCircle2 = edgeLiesOutsideCircle(d2, d0);
        // No intersection with cylinder
        if(edgeOutsideCircle0 && edgeOutsideCircle1 && edgeOutsideCircle2) {
#if enableRenderers
            renderer.waitForClose();
#endif
            return 0;
        }
    }


    double circleTriangleIntersectionArea = computeAreaOfIntersection(circle, intersectionBaseTriangle);

    // Crossings with the upper and lower end planes of the cylinder.
    // We already checked for the triangle being above or below it, so this indicates there exists a
    // crossing with the top and bottom planes.
    std::array<ShapeBench::Triangle3D, 2> clippedTriangles;
    int clippedTriangleCount = 0;
    if(atLeastOneAboveUpperBound) {
        bool vertexAbove0 = rotatedTriangleVertex0.z > cylinderZUpperBound;
        bool vertexAbove1 = rotatedTriangleVertex1.z > cylinderZUpperBound;
        bool vertexAbove2 = rotatedTriangleVertex2.z > cylinderZUpperBound;
        // These triangles lie outside the cylinder, so we need to subtract their area
        clipTriangleAgainstPlane(rotatedTriangleVertex0, rotatedTriangleVertex1,
                                 rotatedTriangleVertex2,
                                 cylinderZUpperBound,
                                 vertexAbove0, vertexAbove1, vertexAbove2,
                                 clippedTriangles, clippedTriangleCount);
        double outsideArea = computeAreaOfIntersection(circle, toTriangle2D(clippedTriangles.at(0)));
        if(clippedTriangleCount == 2) {
            outsideArea += computeAreaOfIntersection(circle, toTriangle2D(clippedTriangles.at(1)));
        }
        circleTriangleIntersectionArea -= outsideArea;
        assert(circleTriangleIntersectionArea >= -0.0001);
    }

    if(atLeastOneBelowLowerBound) {
        bool vertexBelow0 = rotatedTriangleVertex0.z < cylinderZLowerBound;
        bool vertexBelow1 = rotatedTriangleVertex1.z < cylinderZLowerBound;
        bool vertexBelow2 = rotatedTriangleVertex2.z < cylinderZLowerBound;
        // These triangles lie outside the cylinder, so we need to subtract their area
        clipTriangleAgainstPlane(rotatedTriangleVertex0, rotatedTriangleVertex1,
                                 rotatedTriangleVertex2,
                                 cylinderZLowerBound,
                                 vertexBelow0, vertexBelow1, vertexBelow2,
                                 clippedTriangles, clippedTriangleCount);
        double outsideArea = computeAreaOfIntersection(circle, toTriangle2D(clippedTriangles.at(0)));
        if(clippedTriangleCount == 2) {
            outsideArea += computeAreaOfIntersection(circle, toTriangle2D(clippedTriangles.at(1)));
        }
        circleTriangleIntersectionArea -= outsideArea;
        assert(circleTriangleIntersectionArea >= -0.0001);
    }

    // Step 7: multiply the area by the amount it was scaled before
    circleTriangleIntersectionArea *= areaScaleFactor;
#if enableRenderers
    renderer.waitForClose();
#endif
    return circleTriangleIntersectionArea;
}


