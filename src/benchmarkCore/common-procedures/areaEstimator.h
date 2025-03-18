#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <barrier>
#include <random>
#include "filters/FilteredMeshPair.h"
#include "pointCloudSampler.h"
#include "utils/AreaCalculator.h"
#include "nlohmann/json.hpp"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include "types/AreaEstimate.h"
#include <omp.h>

namespace ShapeBench {
    double computeAreaInCylindricalSupportVolume(const ShapeDescriptor::cpu::Mesh &mesh, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius);
    double computeAreaInSphericalSupportVolume(const ShapeDescriptor::cpu::Mesh &mesh, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius);

    inline double computeSingleTriangleArea(ShapeDescriptor::cpu::float3 vertex0, ShapeDescriptor::cpu::float3 vertex1, ShapeDescriptor::cpu::float3 vertex2) {
        ShapeDescriptor::cpu::float3 AB = vertex1 - vertex0;
        ShapeDescriptor::cpu::float3 AC = vertex2 - vertex0;

        double area = length(cross(AB, AC)) * 0.5;
        assert(area >= 0);
        return area;
    }

    template<typename DescriptorMethod>
    uint32_t computeSampleCountInSupportVolume(ShapeDescriptor::cpu::Mesh &mesh, double totalArea, uint64_t sampleCount, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius, uint64_t randomSeed) {
        std::chrono::time_point<std::chrono::steady_clock> start_old = std::chrono::steady_clock::now();

        uint64_t samplesInVolume = 0;
        size_t triangleCount = mesh.vertexCount / 3;

        std::mt19937_64 randomEngine(randomSeed);
        if(totalArea == 0) {
            // Mesh is a simulated point cloud. Sample random vertices instead
            std::uniform_int_distribution<uint32_t> sampleDistribution(0, mesh.vertexCount);
            for(uint32_t i = 0; i < sampleCount; i++) {
                uint32_t sourceIndex = sampleDistribution(randomEngine);

                bool isInVolume = DescriptorMethod::isPointInSupportVolume(supportRadius, referencePoint, mesh.vertices[sourceIndex]);
                if(isInVolume) {
                    samplesInVolume++;
                }
            }
        } else {
            // Normal mesh, sample weighted by area
            std::uniform_real_distribution<float> sampleDistribution(0, float(totalArea));
            std::uniform_real_distribution<float> coefficientDistribution(0, 1);

            std::vector<float> samplePoints(sampleCount);
            for(uint32_t i = 0; i < sampleCount; i++) {
                samplePoints.at(i) = sampleDistribution(randomEngine);
            }
            std::sort(samplePoints.begin(), samplePoints.end());

            uint32_t currentTriangleIndex = 0;
            double cumulativeArea = ShapeBench::computeSingleTriangleArea(mesh.vertices[0], mesh.vertices[1], mesh.vertices[2]);
            // MUST be run in serial!
            for(uint32_t i = 0; i < sampleCount; i++) {
                float sampleAreaPoint = samplePoints.at(i);
                float nextSampleBorder = cumulativeArea;
                while(nextSampleBorder < sampleAreaPoint && currentTriangleIndex < (triangleCount - 1)) {
                    currentTriangleIndex++;
                    cumulativeArea += ShapeBench::computeSingleTriangleArea(mesh.vertices[3 * currentTriangleIndex + 0], mesh.vertices[3 * currentTriangleIndex + 1], mesh.vertices[3 * currentTriangleIndex + 2]);
                    nextSampleBorder = cumulativeArea;
                }

                float v1 = coefficientDistribution(randomEngine);
                float v2 = coefficientDistribution(randomEngine);

                ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[3 * currentTriangleIndex + 0];
                ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[3 * currentTriangleIndex + 1];
                ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[3 * currentTriangleIndex + 2];

                ShapeDescriptor::cpu::float3 samplePoint =
                        (1 - sqrt(v1)) * vertex0 +
                        (sqrt(v1) * (1 - v2)) * vertex1 +
                        (sqrt(v1) * v2) * vertex2;


                bool isInVolume = DescriptorMethod::isPointInSupportVolume(supportRadius, referencePoint, samplePoint);
                if(isInVolume) {
                    samplesInVolume++;
                }
            }
        }
        std::chrono::time_point<std::chrono::steady_clock> end_old = std::chrono::steady_clock::now();

        return samplesInVolume;
    }


    template<typename DescriptorMethod>
    AreaEstimate estimateAreaInSupportVolume(ShapeBench::FilteredMeshPair& meshes,
                                             ShapeDescriptor::OrientedPoint pointInOriginalMesh,
                                             ShapeDescriptor::OrientedPoint pointInFilteredMesh,
                                             float supportRadius,
                                             const nlohmann::json& config,
                                             uint64_t randomSeed) {
        AreaEstimate estimate {0, 0};

        ShapeBench::IntersectingAreaEstimationStrategy desiredStrategy = DescriptorMethod::getIntersectingAreaEstimationStrategy();
        if(desiredStrategy == ShapeBench::IntersectingAreaEstimationStrategy::FAST_CYLINDRICAL) {
            double referenceArea = computeAreaInCylindricalSupportVolume(meshes.originalMesh, pointInOriginalMesh, supportRadius);
            double subtractiveMeshArea = computeAreaInCylindricalSupportVolume(meshes.filteredSampleMesh, pointInFilteredMesh, supportRadius);
            double additiveMeshArea = computeAreaInCylindricalSupportVolume(meshes.filteredAdditiveNoise, pointInFilteredMesh, supportRadius);

            estimate.addedArea = float(additiveMeshArea / referenceArea);
            estimate.subtractiveArea = float(subtractiveMeshArea / referenceArea);

            return estimate;
        } else if(desiredStrategy == ShapeBench::IntersectingAreaEstimationStrategy::FAST_SPHERICAL) {
            double referenceArea = computeAreaInSphericalSupportVolume(meshes.originalMesh, pointInOriginalMesh, supportRadius);
            double subtractiveMeshArea = computeAreaInSphericalSupportVolume(meshes.filteredSampleMesh, pointInFilteredMesh, supportRadius);
            double additiveMeshArea = computeAreaInSphericalSupportVolume(meshes.filteredAdditiveNoise, pointInFilteredMesh, supportRadius);

            estimate.addedArea = float(additiveMeshArea / referenceArea);
            estimate.subtractiveArea = float(subtractiveMeshArea / referenceArea);

            return estimate;
        } else if(desiredStrategy == ShapeBench::IntersectingAreaEstimationStrategy::CUSTOM) {
            ShapeBench::IntersectionAreaParameters parameters;
            parameters.mesh = meshes.originalMesh;
            parameters.descriptorOrigin = pointInOriginalMesh;
            parameters.supportRadius = 1;
            parameters.config = &config;
            parameters.randomSeed = randomSeed;
            double referenceArea = DescriptorMethod::computeIntersectingAreaCustom(parameters);

            parameters.mesh = meshes.filteredSampleMesh;
            parameters.descriptorOrigin = pointInFilteredMesh;
            double subtractiveMeshArea = DescriptorMethod::computeIntersectingAreaCustom(parameters);

            parameters.mesh = meshes.filteredAdditiveNoise;
            parameters.descriptorOrigin = pointInFilteredMesh;
            double additiveMeshArea = DescriptorMethod::computeIntersectingAreaCustom(parameters);

            estimate.addedArea = float(additiveMeshArea / referenceArea);
            estimate.subtractiveArea = float(subtractiveMeshArea / referenceArea);

            return estimate;
        } else if(desiredStrategy == ShapeBench::IntersectingAreaEstimationStrategy::SLOW_MONTE_CARLO_ESTIMATION) {
            double originalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshes.originalMesh);
            double filteredOriginalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshes.filteredSampleMesh);
            double filteredAdditiveMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshes.filteredAdditiveNoise);

            ShapeBench::AreaEstimateSampleCounts sampleCounts = ShapeBench::computeAreaEstimateSampleCounts(config, originalMeshArea, filteredOriginalMeshArea, filteredAdditiveMeshArea);

            uint64_t subtractiveMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.filteredSampleMesh, filteredOriginalMeshArea, sampleCounts.filteredOriginalMesh, pointInFilteredMesh, supportRadius, randomSeed);
            uint64_t referenceMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.originalMesh, originalMeshArea, sampleCounts.originalMesh, pointInOriginalMesh, supportRadius, randomSeed);
            uint64_t additiveMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.filteredAdditiveNoise, filteredAdditiveMeshArea, sampleCounts.filteredAdditiveMesh, pointInFilteredMesh, supportRadius, randomSeed);

            estimate.addedArea = float(double(additiveMeshSamples) / double(referenceMeshSamples));
            estimate.subtractiveArea = float(double(subtractiveMeshSamples) / double(referenceMeshSamples));

            return estimate;
        } else {
            throw std::runtime_error("Method has unknown area estimation strategy. Is it missing from this handler?");
        }
    }
}

