#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <barrier>
#include "filters/FilteredMeshPair.h"
#include "json.hpp"
#include "pointCloudSampler.h"

namespace ShapeBench {
    struct AreaEstimate {
        float addedAdrea = 0;
        float subtractiveArea = 0;
    };

    template<typename DescriptorMethod>
    uint32_t computeSampleCountInSupportVolume(ShapeDescriptor::cpu::Mesh &mesh, uint32_t sampleCount, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius, uint64_t randomSeed) {
        ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::sampleMesh(mesh, sampleCount, randomSeed);
        uint32_t samplesInVolume = 0;
        for(uint32_t i = 0; i < cloud.pointCount; i++) {
            bool isInVolume = DescriptorMethod::isPointInSupportVolume(supportRadius, referencePoint, cloud.vertices[i]);
            if(isInVolume) {
                samplesInVolume++;
            }
        }
        ShapeDescriptor::free(cloud);
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

        uint32_t sampleCountOriginalMesh = ShapeBench::computeSampleCount(originalMeshArea, config);
        uint32_t sampleCountFilteredOriginalMesh = uint32_t((filteredOriginalMeshArea / originalMeshArea) * double(sampleCountOriginalMesh));
        uint32_t sampleCountFilteredAdditiveMesh = uint32_t((filteredAdditiveMeshArea / originalMeshArea) * double(sampleCountOriginalMesh));

        uint32_t referenceMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.originalMesh, sampleCountOriginalMesh, pointInOriginalMesh, supportRadius, randomSeed);
        uint32_t subtractiveMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.filteredSampleMesh, sampleCountFilteredOriginalMesh, pointInFilteredMesh, supportRadius, randomSeed);
        uint32_t additiveMeshSamples = computeSampleCountInSupportVolume<DescriptorMethod>(meshes.filteredAdditiveNoise, sampleCountFilteredAdditiveMesh, pointInFilteredMesh, supportRadius, randomSeed);


        AreaEstimate estimate;
        estimate.addedAdrea = double(additiveMeshSamples) / double(referenceMeshSamples);
        estimate.subtractiveArea = double(subtractiveMeshSamples) / double(referenceMeshSamples);
        std::cout << "Area estimate: added " << estimate.addedAdrea << ", subtracted " << estimate.subtractiveArea << std::endl;

        return estimate;
    }
}

