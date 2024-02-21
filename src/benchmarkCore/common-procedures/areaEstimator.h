#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <barrier>
#include <random>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"
#include "pointCloudSampler.h"

namespace ShapeBench {
    struct AreaEstimate {
        float addedAdrea = 0;
        float subtractiveArea = 0;
    };

    inline double computeSingleTriangleArea(ShapeDescriptor::cpu::float3 vertex0, ShapeDescriptor::cpu::float3 vertex1, ShapeDescriptor::cpu::float3 vertex2) {
        ShapeDescriptor::cpu::float3 AB = vertex1 - vertex0;
        ShapeDescriptor::cpu::float3 AC = vertex2 - vertex0;

        double area = length(cross(AB, AC)) * 0.5;
        assert(area >= 0);
        return area;
    }

    template<typename DescriptorMethod>
    uint32_t computeSampleCountInSupportVolume(ShapeDescriptor::cpu::Mesh &mesh, uint64_t sampleCount, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius, uint64_t randomSeed) {
        size_t triangleCount = mesh.vertexCount / 3;

        double totalArea = 0;
        for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
            double area = ShapeBench::computeSingleTriangleArea(mesh.vertices[i], mesh.vertices[i + 1], mesh.vertices[i + 2]);
            totalArea += area;
        }

        uint64_t samplesInVolume = 0;

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

        return samplesInVolume;
    }

    template<typename DescriptorMethod>
    AreaEstimate estimateAreaInSupportVolume(ShapeBench::FilteredMeshPair& meshes,
                                             ShapeDescriptor::OrientedPoint pointInOriginalMesh,
                                             ShapeDescriptor::OrientedPoint pointInFilteredMesh,
                                             float supportRadius,
                                             const nlohmann::json& config,
                                             uint64_t randomSeed) {
        // We need to compute the sample count in such a way that all surfaces have a similar point density
        // We do this by computing a sample count for the base mesh, and scaling it by the area of the subtractive and additive meshes

        double originalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshes.originalMesh);
        double filteredOriginalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshes.filteredSampleMesh);
        double filteredAdditiveMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(meshes.filteredAdditiveNoise);

        uint64_t sampleCountOriginalMesh = ShapeBench::computeSampleCount(originalMeshArea, config) / 10;
        uint64_t sampleCountFilteredOriginalMesh = uint64_t((filteredOriginalMeshArea / originalMeshArea) * double(sampleCountOriginalMesh));
        uint64_t sampleCountFilteredAdditiveMesh = uint64_t((filteredAdditiveMeshArea / originalMeshArea) * double(sampleCountOriginalMesh));

        uint64_t referenceMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.originalMesh, sampleCountOriginalMesh, pointInOriginalMesh, supportRadius, randomSeed);
        uint64_t subtractiveMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.filteredSampleMesh, sampleCountFilteredOriginalMesh, pointInFilteredMesh, supportRadius, randomSeed);
        uint64_t additiveMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.filteredAdditiveNoise, sampleCountFilteredAdditiveMesh, pointInFilteredMesh, supportRadius, randomSeed);


        AreaEstimate estimate;
        estimate.addedAdrea = double(additiveMeshSamples) / double(referenceMeshSamples);
        estimate.subtractiveArea = double(subtractiveMeshSamples) / double(referenceMeshSamples);
        //std::cout << "Area estimate: added " << estimate.addedAdrea << ", subtracted " << estimate.subtractiveArea << std::endl;

        return estimate;
    }
}

