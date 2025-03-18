#include <benchmarkCore/common-procedures/areaEstimator.h>
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include "catch2/catch_all.hpp"
#include <random>
#include "glm/ext/matrix_transform.hpp"
#include <omp.h>

inline ShapeDescriptor::cpu::double3 pointFromCylindricalCoordinates(ShapeBench::Cylinder cylinder, float angleRadians, float heightAlongSymmetryAxis, float distanceFromAxis) {
    const ShapeDescriptor::cpu::double3 zAxis(0, 0, 1);
    ShapeDescriptor::cpu::double3 orthogonalDirection = normalize(cross(cylinder.normalisedDirection, zAxis));
    glm::vec3 rotationDirection = glm::vec3(cylinder.normalisedDirection.x, cylinder.normalisedDirection.y, cylinder.normalisedDirection.z);
    glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0), angleRadians, rotationDirection);
    glm::vec3 rotatedCoordinate = glm::vec3(rotationMatrix * glm::vec4(orthogonalDirection.x, orthogonalDirection.y, orthogonalDirection.z, 1));
    ShapeDescriptor::cpu::double3 convertedRotatedDirection = {rotatedCoordinate.x, rotatedCoordinate.y, rotatedCoordinate.z};
    convertedRotatedDirection = normalize(convertedRotatedDirection);
    ShapeDescriptor::cpu::double3 selectedPoint = cylinder.centrePoint
            + heightAlongSymmetryAxis * cylinder.normalisedDirection
            + distanceFromAxis * convertedRotatedDirection;
    return selectedPoint;
}

inline double estimateTriangleArea(ShapeBench::Cylinder cylinder,
                                   std::uniform_real_distribution<float> coefficientDistribution,
                                   ShapeDescriptor::cpu::double3 vertex0,
                                   ShapeDescriptor::cpu::double3 vertex1,
                                   ShapeDescriptor::cpu::double3 vertex2,
                                   uint64_t sampleCount,
                                   uint64_t randomSeed) {

    uint64_t samplesWithinCylinder = 0;
    #pragma omp parallel
    {
        uint64_t threadSamplesInCylinder = 0;
        for(uint64_t i = omp_get_thread_num(); i < sampleCount; i += omp_get_num_threads()) {
            std::minstd_rand randomEngine(i * i * i * randomSeed);
            float v1 = coefficientDistribution(randomEngine);
            float v2 = coefficientDistribution(randomEngine);

            ShapeDescriptor::cpu::double3 samplePoint =
                    (1 - sqrt(v1)) * vertex0 +
                    (sqrt(v1) * (1 - v2)) * vertex1 +
                    (sqrt(v1) * v2) * vertex2;

            if(ShapeBench::isPointInCylindricalVolume({toFloat3(cylinder.centrePoint), toFloat3(cylinder.normalisedDirection)}, cylinder.radius, 2.0 * cylinder.halfOfHeight, toFloat3(samplePoint))) {
                threadSamplesInCylinder++;
            }
        }
        #pragma omp atomic
        samplesWithinCylinder += threadSamplesInCylinder;
    };


    double triangleArea = ShapeDescriptor::computeTriangleArea(vertex0, vertex1, vertex2);
    double fractionSamplesInCylinder = double(samplesWithinCylinder) / double(sampleCount);
    double estimatedArea = triangleArea * fractionSamplesInCylinder;
    return estimatedArea;
}

inline bool generateRandomCylinder(ShapeBench::Cylinder& cylinder, std::minstd_rand& randomEngine,
                                   std::uniform_real_distribution<float> cylinderRadiusDistribution,
                                   std::uniform_real_distribution<float> cylinderPositionDistribution,
                                   std::uniform_real_distribution<float> randomDirectionDistribution) {
    cylinder.centrePoint = {cylinderPositionDistribution(randomEngine),
                            cylinderPositionDistribution(randomEngine),
                            cylinderPositionDistribution(randomEngine)};
    // Not a uniform distribution, but good enough for fuzzing
    cylinder.normalisedDirection = {randomDirectionDistribution(randomEngine),
                                    randomDirectionDistribution(randomEngine),
                                    randomDirectionDistribution(randomEngine)};
    if(cylinder.normalisedDirection.x == 0 && cylinder.normalisedDirection.y == 0 && cylinder.normalisedDirection.z == 0) {
        std::cout << "    Normalised direction is 0!" << std::endl;
        return false;
    }
    if(cylinder.normalisedDirection.x == 0 && cylinder.normalisedDirection.y == 0) {
        std::cout << "    Normalised direction is exactly the z axis!" << std::endl;
        return false;
    }
    cylinder.normalisedDirection = normalize(cylinder.normalisedDirection);
    cylinder.radius = cylinderRadiusDistribution(randomEngine);
    cylinder.halfOfHeight = cylinder.radius / 2.0f;
    return true;
}

TEST_CASE("Triangle-cylinder intersection area") {
    const uint64_t randomSeed = 166696615;
    const uint32_t numberOfTests = 1000000;

    std::minstd_rand randomEngine {randomSeed};
    std::uniform_real_distribution<float> cylinderRadiusDistribution(0, 1000);
    std::uniform_real_distribution<float> cylinderPositionDistribution(-1000, 1000);
    std::uniform_real_distribution<float> randomDirectionDistribution(-1, 1);
    std::uniform_real_distribution<float> randomDistribution(0, 1);
    std::uniform_real_distribution<float> randomAngleDistribution(0, 2.0 * M_PI);

    SECTION("Triangle vertices within cylinder") {
        for(uint32_t i = 0; i < numberOfTests; i++) {
            ShapeBench::Cylinder cylinder;
            if(!generateRandomCylinder(cylinder, randomEngine, cylinderRadiusDistribution, cylinderPositionDistribution, randomDirectionDistribution)) {
                continue;
            }


            ShapeBench::Triangle3D triangle;
            triangle.vertex0 = pointFromCylindricalCoordinates(cylinder,
                                                               randomAngleDistribution(randomEngine),
                                                               randomDirectionDistribution(randomEngine) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * cylinder.radius);
            triangle.vertex1 = pointFromCylindricalCoordinates(cylinder,
                                                               randomAngleDistribution(randomEngine),
                                                               randomDirectionDistribution(randomEngine) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * cylinder.radius);
            triangle.vertex2 = pointFromCylindricalCoordinates(cylinder,
                                                               randomAngleDistribution(randomEngine),
                                                               randomDirectionDistribution(randomEngine) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * cylinder.radius);


            double calculatedArea = ShapeBench::computeAreaOfIntersection(cylinder, triangle);
            double triangleArea = ShapeDescriptor::computeTriangleArea(triangle.vertex0, triangle.vertex1, triangle.vertex2);

            REQUIRE_THAT(calculatedArea, Catch::Matchers::WithinRel(triangleArea, 0.001));
        }
    }

    SECTION("Triangle vertices within cylinder: rectangle intersection patterns") {
        for(uint32_t i = 0; i < numberOfTests; i++) {
            ShapeBench::Cylinder cylinder;
            if(!generateRandomCylinder(cylinder, randomEngine, cylinderRadiusDistribution, cylinderPositionDistribution, randomDirectionDistribution)) {
                continue;
            }

            float commonAngle = randomAngleDistribution(randomEngine);
            ShapeBench::Triangle3D triangle;
            triangle.vertex0 = pointFromCylindricalCoordinates(cylinder,
                                                               commonAngle,
                                                               randomDirectionDistribution(randomEngine) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * cylinder.radius);
            triangle.vertex1 = pointFromCylindricalCoordinates(cylinder,
                                                               commonAngle,
                                                               randomDirectionDistribution(randomEngine) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * cylinder.radius);
            triangle.vertex2 = pointFromCylindricalCoordinates(cylinder,
                                                               commonAngle,
                                                               randomDirectionDistribution(randomEngine) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * cylinder.radius);


            double calculatedArea = ShapeBench::computeAreaOfIntersection(cylinder, triangle);
            double triangleArea = ShapeDescriptor::computeTriangleArea(triangle.vertex0, triangle.vertex1, triangle.vertex2);

            REQUIRE_THAT(calculatedArea, Catch::Matchers::WithinRel(triangleArea, 0.001));
        }
    }
    SECTION("Triangle vertices within cylinder: rectangle intersection patterns, triangle sticks out") {
        for(uint32_t i = 0; i < numberOfTests; i++) {
            std::cout << "Testing " << i << std::endl;
            ShapeBench::Cylinder cylinder;
            if(!generateRandomCylinder(cylinder, randomEngine, cylinderRadiusDistribution, cylinderPositionDistribution, randomDirectionDistribution)) {
                continue;
            }

            float commonAngle = randomAngleDistribution(randomEngine);
            float distanceInsideCylinder = randomDistribution(randomEngine) * cylinder.radius;
            float distanceOutsideCylinder = (randomDistribution(randomEngine) * 10.0f) + cylinder.radius;
            float triangleBaseWidth = randomDistribution(randomEngine) * cylinder.halfOfHeight;
            ShapeBench::Triangle3D triangle;
            triangle.vertex0 = pointFromCylindricalCoordinates(cylinder,
                                                               commonAngle,
                                                               -triangleBaseWidth,
                                                               distanceInsideCylinder);
            triangle.vertex1 = pointFromCylindricalCoordinates(cylinder,
                                                               commonAngle,
                                                               triangleBaseWidth,
                                                               distanceInsideCylinder);
            triangle.vertex2 = pointFromCylindricalCoordinates(cylinder,
                                                               commonAngle,
                                                               0,
                                                               distanceOutsideCylinder);





            double calculatedArea = ShapeBench::computeAreaOfIntersection(cylinder, triangle);

            double triangleHeight = distanceOutsideCylinder - distanceInsideCylinder;
            double outsideTriangleHeight = distanceOutsideCylinder - cylinder.radius;
            double outsideTriangleScaleFactor = outsideTriangleHeight / triangleHeight;
            double outsideTriangleArea = outsideTriangleHeight * outsideTriangleScaleFactor * triangleBaseWidth;
            double totalTriangleArea = triangleHeight * triangleBaseWidth;
            double overlappingArea = totalTriangleArea - outsideTriangleArea;

            // Rounding error territory
            if(calculatedArea < 0.0001) {
                std::cout << "    Skipped because area too small: " << calculatedArea << std::endl;
                continue;
            }

            REQUIRE_THAT(calculatedArea, Catch::Matchers::WithinRel(overlappingArea, 0.001));
        }
    }
    
    SECTION("Triangle vertices within cylinder: purely random triangles") {
        for(uint32_t i = 0; i < numberOfTests; i++) {

            ShapeBench::Cylinder cylinder;
            if(!generateRandomCylinder(cylinder, randomEngine, cylinderRadiusDistribution, cylinderPositionDistribution, randomDirectionDistribution)) {
                continue;
            }

            ShapeBench::Triangle3D triangle;
            triangle.vertex0 = pointFromCylindricalCoordinates(cylinder,
                                                               randomAngleDistribution(randomEngine),
                                                               ((randomDistribution(randomEngine) * 3.0f) - 1.5) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * 2.0f * cylinder.radius);
            triangle.vertex1 = pointFromCylindricalCoordinates(cylinder,
                                                               randomAngleDistribution(randomEngine),
                                                               ((randomDistribution(randomEngine) * 3.0f) - 1.5) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * 2.0f * cylinder.radius);
            triangle.vertex2 = pointFromCylindricalCoordinates(cylinder,
                                                               randomAngleDistribution(randomEngine),
                                                               ((randomDistribution(randomEngine) * 3.0f) - 1.5) * cylinder.halfOfHeight,
                                                               randomDistribution(randomEngine) * 2.0f * cylinder.radius);


            double calculatedArea = ShapeBench::computeAreaOfIntersection(cylinder, triangle);
            double estimatedArea = estimateTriangleArea(cylinder, randomDistribution, triangle.vertex0, triangle.vertex1, triangle.vertex2, 1000000000, randomEngine());
            std::cout << "Testing " << i << " - " << calculatedArea << " vs " << estimatedArea << " (" << (std::abs(1 - calculatedArea / estimatedArea)) << ")" << std::endl;

            REQUIRE_THAT(calculatedArea, Catch::Matchers::WithinRel(estimatedArea, 0.002) || Catch::Matchers::WithinAbs(estimatedArea, 0.00001));
        }
    }
}