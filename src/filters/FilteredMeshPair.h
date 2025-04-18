#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "glm/glm.hpp"

namespace ShapeBench {
    struct ChosenVertexPRC {
        uint32_t meshID = 0;
        ShapeDescriptor::OrientedPoint vertex = {{0, 0, 0}, {0, 0, 0}};
    };

    struct AdditiveNoiseObjectInfo {
        uint32_t vertexCount = 0;
        uint32_t meshID = 0;
        glm::mat4 transformation = glm::mat4(1.0);
        bool included = false;
    };

    struct FilteredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh filteredSampleMesh;
        ShapeDescriptor::cpu::Mesh filteredAdditiveNoise;


        std::vector<uint32_t> mappedReferenceVertexIndices;
        std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices;
        std::vector<ShapeDescriptor::OrientedPoint> mappedReferenceVertices;
        std::vector<bool> mappedVertexIncluded;
        std::vector<bool> remainingTrianglesFromOriginalMesh;

        // PRC related data
        std::vector<AdditiveNoiseObjectInfo> additiveNoiseInfo;
        glm::mat4 sampleMeshTransformation = glm::mat4(1.0);
        glm::mat3 sampleMeshNormalTransformation = glm::mat3(1.0);

        // Scale factor for number of points to sample when converting this mesh into a point cloud
        float pointCloudConversionScaleFactor = 1.0;

        void free();
        ShapeDescriptor::cpu::Mesh combinedFilteredMesh();
        FilteredMeshPair clone();
    };

    template<typename DescriptorMethod>
    void writeFilteredMesh(FilteredMeshPair& filteredMesh, std::filesystem::path outputFile, ShapeDescriptor::OrientedPoint referencePoint = {{0, 0, 0}, {0, 0, 0}}, float supportRadius = 0, bool onlyIncludeSupportVolume = false) {
        uint32_t numberOfIndicators = 0;//filteredMesh.mappedReferenceVertices.size();
        ShapeDescriptor::cpu::Mesh tempExtendedMesh = filteredMesh.combinedFilteredMesh();

        FilteredMeshPair tempScene;
        tempScene.filteredSampleMesh = tempExtendedMesh;
        tempScene.filteredAdditiveNoise = ShapeDescriptor::cpu::Mesh(numberOfIndicators * 3);

        ShapeDescriptor::cpu::Mesh extendedMesh = tempScene.combinedFilteredMesh();
        ShapeDescriptor::free(tempExtendedMesh);

        const float arrowSize = 0.1;
        for(uint32_t i = 0; i < numberOfIndicators; i++) {
            if(!filteredMesh.mappedVertexIncluded.at(i)) {
                continue;
            }
            ShapeDescriptor::OrientedPoint baseVertex = filteredMesh.mappedReferenceVertices.at(i);
            ShapeDescriptor::cpu::float3 vertex0 = baseVertex.vertex;
            ShapeDescriptor::cpu::float3 vertex1 = baseVertex.vertex + baseVertex.normal + ShapeDescriptor::cpu::float3(arrowSize / 2.0f, 0, 0);
            ShapeDescriptor::cpu::float3 vertex2 = baseVertex.vertex + baseVertex.normal + ShapeDescriptor::cpu::float3(-arrowSize / 2.0f, 0, 0);
            ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::cpu::float3(0, 0, -1);

            extendedMesh.vertices[filteredMesh.filteredSampleMesh.vertexCount + 3 * i + 0] = vertex0;
            extendedMesh.vertices[filteredMesh.filteredSampleMesh.vertexCount + 3 * i + 1] = vertex1;
            extendedMesh.vertices[filteredMesh.filteredSampleMesh.vertexCount + 3 * i + 2] = vertex2;
            extendedMesh.normals[filteredMesh.filteredSampleMesh.vertexCount + 3 * i + 0] = normal;
            extendedMesh.normals[filteredMesh.filteredSampleMesh.vertexCount + 3 * i + 1] = normal;
            extendedMesh.normals[filteredMesh.filteredSampleMesh.vertexCount + 3 * i + 2] = normal;
        }

        uint32_t removedTriangleCount = 0;
        if(onlyIncludeSupportVolume) {
            for(uint32_t triangleIndex = 0; triangleIndex < extendedMesh.vertexCount / 3; triangleIndex++) {
                extendedMesh.vertices[3 * (triangleIndex - removedTriangleCount) + 0] = extendedMesh.vertices[3 * triangleIndex + 0];
                extendedMesh.vertices[3 * (triangleIndex - removedTriangleCount) + 1] = extendedMesh.vertices[3 * triangleIndex + 1];
                extendedMesh.vertices[3 * (triangleIndex - removedTriangleCount) + 2] = extendedMesh.vertices[3 * triangleIndex + 2];

                extendedMesh.normals[3 * (triangleIndex - removedTriangleCount) + 0] = extendedMesh.normals[3 * triangleIndex + 0];
                extendedMesh.normals[3 * (triangleIndex - removedTriangleCount) + 1] = extendedMesh.normals[3 * triangleIndex + 1];
                extendedMesh.normals[3 * (triangleIndex - removedTriangleCount) + 2] = extendedMesh.normals[3 * triangleIndex + 2];

                if(!DescriptorMethod::isPointInSupportVolume(supportRadius, referencePoint, extendedMesh.vertices[3 * triangleIndex + 0])
                && !DescriptorMethod::isPointInSupportVolume(supportRadius, referencePoint, extendedMesh.vertices[3 * triangleIndex + 1])
                && !DescriptorMethod::isPointInSupportVolume(supportRadius, referencePoint, extendedMesh.vertices[3 * triangleIndex + 2])) {
                    removedTriangleCount++;
                }
            }
        }

        extendedMesh.vertexCount = extendedMesh.vertexCount - 3*removedTriangleCount;

        ShapeDescriptor::writeOBJ(extendedMesh, outputFile);

        ShapeDescriptor::free(extendedMesh);
    }
}