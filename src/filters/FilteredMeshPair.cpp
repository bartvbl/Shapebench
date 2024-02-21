#include "FilteredMeshPair.h"

void ShapeBench::FilteredMeshPair::free() {
    ShapeDescriptor::free(originalMesh);
    ShapeDescriptor::free(filteredSampleMesh);
    ShapeDescriptor::free(filteredAdditiveNoise);
}

ShapeDescriptor::cpu::Mesh ShapeBench::FilteredMeshPair::combinedFilteredMesh() {
    uint32_t combinedVertexCount = filteredSampleMesh.vertexCount + filteredAdditiveNoise.vertexCount;

    ShapeDescriptor::cpu::Mesh outputMesh(combinedVertexCount);
    std::copy(filteredSampleMesh.vertices, filteredSampleMesh.vertices + filteredSampleMesh.vertexCount, outputMesh.vertices);
    std::copy(filteredAdditiveNoise.vertices, filteredAdditiveNoise.vertices + filteredAdditiveNoise.vertexCount, outputMesh.vertices + filteredSampleMesh.vertexCount);
    std::copy(filteredSampleMesh.normals, filteredSampleMesh.normals + filteredSampleMesh.vertexCount, outputMesh.normals);
    std::copy(filteredAdditiveNoise.normals, filteredAdditiveNoise.normals + filteredAdditiveNoise.vertexCount, outputMesh.normals + filteredSampleMesh.vertexCount);

    return outputMesh;
}

void ShapeBench::FilteredMeshPair::writeFilteredMesh(std::filesystem::path outputFile) {
    ShapeDescriptor::cpu::Mesh combinedMesh = combinedFilteredMesh();
    ShapeDescriptor::cpu::Mesh extendedMesh(combinedMesh.vertexCount + 3 * mappedReferenceVertices.size());
    std::copy(combinedMesh.vertices, combinedMesh.vertices + combinedMesh.vertexCount, extendedMesh.vertices);
    std::copy(combinedMesh.normals, combinedMesh.normals + combinedMesh.vertexCount, extendedMesh.normals);
    ShapeDescriptor::free(combinedMesh);

    const float arrowSize = 0.1;
    for(uint32_t i = 0; i < mappedReferenceVertices.size(); i++) {
        ShapeDescriptor::OrientedPoint baseVertex = mappedReferenceVertices.at(i);
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

    ShapeDescriptor::writeOBJ(extendedMesh, outputFile);

    ShapeDescriptor::free(extendedMesh);
}
