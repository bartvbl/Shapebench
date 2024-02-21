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
