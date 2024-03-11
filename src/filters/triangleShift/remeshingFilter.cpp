#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "remeshingFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing.h"
#include "utils/filterUtils/pmpConverter.h"


ShapeBench::RemeshingFilterOutput ShapeBench::remesh(ShapeBench::FilteredMeshPair& scene) {
    // Convert to PMP Mesh
    pmp::SurfaceMesh sampleMesh = ShapeBench::convertSDMeshToPMP(scene.filteredSampleMesh);
    pmp::SurfaceMesh additiveNoiseMesh = ShapeBench::convertSDMeshToPMP(scene.filteredAdditiveNoise);

    // Using the same approach as PMP library's remeshing tool
    // We calculate the average of both meshes combined
    pmp::Scalar averageEdgeLength = 0;
    uint32_t edgeIndex = 0;
    for (const auto& edgeInMesh : sampleMesh.edges()) {
        averageEdgeLength += distance(sampleMesh.position(sampleMesh.vertex(edgeInMesh, 0)),
                                      sampleMesh.position(sampleMesh.vertex(edgeInMesh, 1))) / pmp::Scalar(edgeIndex + 1);
        edgeIndex++;
    }
    for (const auto& edgeInMesh : additiveNoiseMesh.edges()) {
        averageEdgeLength += distance(additiveNoiseMesh.position(additiveNoiseMesh.vertex(edgeInMesh, 0)),
                                      additiveNoiseMesh.position(additiveNoiseMesh.vertex(edgeInMesh, 1))) / pmp::Scalar(edgeIndex + 1);
        edgeIndex++;
    }

    // Mario Botsch and Leif Kobbelt. A remeshing approach to multiresolution modeling. In Proceedings of Eurographics Symposium on Geometry Processing, pages 189–96, 2004.
    pmp::uniform_remeshing(sampleMesh, averageEdgeLength);
    pmp::uniform_remeshing(additiveNoiseMesh, averageEdgeLength);

    // Convert back to original format
    ShapeDescriptor::free(scene.filteredSampleMesh);
    ShapeDescriptor::free(scene.filteredAdditiveNoise);

    scene.filteredSampleMesh = ShapeBench::convertPMPMeshToSD(sampleMesh);
    scene.filteredAdditiveNoise = ShapeBench::convertPMPMeshToSD(additiveNoiseMesh);

    // Update reference points


    std::vector<float> bestDistances(scene.mappedReferenceVertices.size(), std::numeric_limits<float>::max());
    std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices = scene.mappedReferenceVertices;

    for(uint32_t meshVertexIndex = 0; meshVertexIndex < scene.filteredSampleMesh.vertexCount; meshVertexIndex++) {
        ShapeDescriptor::cpu::float3 meshVertex = scene.filteredSampleMesh.vertices[meshVertexIndex];
        for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
            ShapeDescriptor::OrientedPoint referencePoint = scene.mappedReferenceVertices.at(i);
            ShapeDescriptor::cpu::float3 referenceVertex = referencePoint.vertex;
            float distanceToReferenceVertex = length(referenceVertex - meshVertex);
            if(distanceToReferenceVertex < bestDistances.at(i)) {
                bestDistances.at(i) = distanceToReferenceVertex;
                scene.mappedReferenceVertices.at(i).vertex = meshVertex;
                scene.mappedReferenceVertexIndices.at(i) = meshVertexIndex;
            }
        }
    }

    ShapeBench::RemeshingFilterOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["triangle-shift-displacement-distance"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
        output.metadata.push_back(entry);
    }

    return output;
}
