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

ShapeBench::FilteredMeshPair ShapeBench::FilteredMeshPair::clone() {
    ShapeBench::FilteredMeshPair duplicated;
    duplicated.originalMesh = originalMesh.clone();
    duplicated.filteredSampleMesh = filteredSampleMesh.clone();
    duplicated.filteredAdditiveNoise = filteredAdditiveNoise.clone();

    duplicated.mappedReferenceVertexIndices = mappedReferenceVertexIndices;
    duplicated.originalReferenceVertices = originalReferenceVertices;
    duplicated.mappedReferenceVertices = mappedReferenceVertices;
    duplicated.mappedVertexIncluded = mappedVertexIncluded;
    duplicated.remainingTrianglesFromOriginalMesh = remainingTrianglesFromOriginalMesh;

    duplicated.additiveNoiseInfo = additiveNoiseInfo;
    duplicated.sampleMeshTransformation = sampleMeshTransformation;
    duplicated.sampleMeshNormalTransformation = sampleMeshNormalTransformation;

    duplicated.pointCloudConversionScaleFactor = pointCloudConversionScaleFactor = 1.0;

    return duplicated;
}
