#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    struct FilteredMeshPair {
        ShapeDescriptor::cpu::Mesh originalMesh;
        ShapeDescriptor::cpu::Mesh filteredSampleMesh;
        ShapeDescriptor::cpu::Mesh filteredAdditiveNoise;

        std::vector<uint32_t> referenceVertexIndices;
        std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices;
        std::vector<ShapeDescriptor::OrientedPoint> mappedReferenceVertices;
        std::vector<bool> mappedVertexIncluded;

        void free();
        ShapeDescriptor::cpu::Mesh combinedFilteredMesh();

    };

    template<typename DescriptorMethod>
    void writeFilteredMesh(FilteredMeshPair& filteredMesh, std::filesystem::path outputFile, ShapeDescriptor::OrientedPoint referencePoint = {{0, 0, 0}, {0, 0, 0}}, float supportRadius = 0, bool onlyIncludeSupportVolume = false) {
        uint32_t numberOfIndicators = 1; //filteredMesh.mappedReferenceVertices.size()
        ShapeDescriptor::cpu::Mesh combinedMesh = filteredMesh.filteredSampleMesh.clone();
        ShapeDescriptor::cpu::Mesh extendedMesh(combinedMesh.vertexCount + 3 * numberOfIndicators);
        std::copy(combinedMesh.vertices, combinedMesh.vertices + combinedMesh.vertexCount, extendedMesh.vertices);
        std::copy(combinedMesh.normals, combinedMesh.normals + combinedMesh.vertexCount, extendedMesh.normals);
        ShapeDescriptor::free(combinedMesh);

        const float arrowSize = 0.1;
        for(uint32_t i = 0; i < numberOfIndicators; i++) {
            ShapeDescriptor::OrientedPoint baseVertex = filteredMesh.mappedReferenceVertices.at(i);
            ShapeDescriptor::cpu::float3 vertex0 = baseVertex.vertex;
            ShapeDescriptor::cpu::float3 vertex1 = baseVertex.vertex + baseVertex.normal + ShapeDescriptor::cpu::float3(arrowSize / 2.0f, 0, 0);
            ShapeDescriptor::cpu::float3 vertex2 = baseVertex.vertex + baseVertex.normal + ShapeDescriptor::cpu::float3(-arrowSize / 2.0f, 0, 0);
            ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::cpu::float3(0, 0, -1);

            extendedMesh.vertices[combinedMesh.vertexCount + 3 * i + 0] = vertex0;
            extendedMesh.vertices[combinedMesh.vertexCount + 3 * i + 1] = vertex1;
            extendedMesh.vertices[combinedMesh.vertexCount + 3 * i + 2] = vertex2;
            extendedMesh.normals[combinedMesh.vertexCount + 3 * i + 0] = normal;
            extendedMesh.normals[combinedMesh.vertexCount + 3 * i + 1] = normal;
            extendedMesh.normals[combinedMesh.vertexCount + 3 * i + 2] = normal;
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