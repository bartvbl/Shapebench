#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "remeshingFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing.h"
#include "utils/filterUtils/pmpConverter.h"


ShapeBench::RemeshingFilterOutput ShapeBench::remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config) {
    // Convert to PMP Mesh
    pmp::SurfaceMesh sampleMesh;
    ShapeBench::convertSDMeshToPMP(scene.filteredSampleMesh, sampleMesh);
    pmp::SurfaceMesh additiveNoiseMesh;
    ShapeBench::convertSDMeshToPMP(scene.filteredAdditiveNoise, additiveNoiseMesh);

    // Using the same approach as PMP library's remeshing tool
    // We calculate the average of both meshes combined
    double averageEdgeLength = 0;
    uint32_t edgeIndex = 0;
    internal::calculateAverageEdgeLength(scene.filteredSampleMesh, averageEdgeLength, edgeIndex);
    internal::calculateAverageEdgeLength(scene.filteredAdditiveNoise, averageEdgeLength, edgeIndex);
    //std::cout << "Average length: " << averageEdgeLength << std::endl;
    //averageEdgeLength = 0.05;
    //averageEdgeLength *= 1.5;

    float minEdgeLengthRatio = config.at("filterSettings").at("alternateTriangulation").at("minEdgeLengthFactor");
    float maxEdgeLengthRatio = config.at("filterSettings").at("alternateTriangulation").at("maxEdgeLengthFactor");
    float maxErrorFactor = config.at("filterSettings").at("alternateTriangulation").at("maxErrorFactor");
    uint32_t iterationCount = config.at("filterSettings").at("alternateTriangulation").at("remeshIterationCount");

    // Mario Botsch and Leif Kobbelt. A remeshing approach to multiresolution modeling. In Proceedings of Eurographics Symposium on Geometry Processing, pages 189–96, 2004.
    pmp::uniform_remeshing(sampleMesh, averageEdgeLength, iterationCount, true);
    pmp::uniform_remeshing(additiveNoiseMesh, averageEdgeLength, iterationCount, true);

   /*

    pmp::adaptive_remeshing(sampleMesh, minEdgeLengthRatio * float(averageEdgeLength),
                            maxEdgeLengthRatio * float(averageEdgeLength),
                            maxErrorFactor * float(averageEdgeLength),
                            iterationCount);
*/
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
                scene.mappedReferenceVertices.at(i).normal = scene.filteredSampleMesh.normals[meshVertexIndex];
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

    //ShapeDescriptor::writeOBJ(scene.filteredSampleMesh, "remeshed.obj");

    malloc_trim(0);

    return output;
}

void ShapeBench::internal::calculateAverageEdgeLength(const ShapeDescriptor::cpu::Mesh& mesh, double &averageEdgeLength,uint32_t &edgeIndex) {
    for (uint32_t triangleBaseIndex = 0; triangleBaseIndex < mesh.vertexCount; triangleBaseIndex += 3) {
        ShapeDescriptor::cpu::float3& vertex0 = mesh.vertices[triangleBaseIndex];
        ShapeDescriptor::cpu::float3& vertex1 = mesh.vertices[triangleBaseIndex + 1];
        ShapeDescriptor::cpu::float3& vertex2 = mesh.vertices[triangleBaseIndex + 2];
        double length10 = length(vertex1 - vertex0);
        double length21 = length(vertex2 - vertex1);
        double length02 = length(vertex0 - vertex2);
        //std::cout << vertex0 << ", " << vertex1 << ", " << vertex2 << " - " << length10 << ", " << length21 << ", " << length02 << " - " << averageEdgeLength << std::endl;
        averageEdgeLength += (length10 - averageEdgeLength) / double(edgeIndex + 1);
        edgeIndex++;
        //std::cout << averageEdgeLength << ", " << edgeIndex << std::endl;
        averageEdgeLength += (length21 - averageEdgeLength) / double(edgeIndex + 1);
        edgeIndex++;
        //std::cout << averageEdgeLength << ", " << edgeIndex << std::endl;
        averageEdgeLength += (length02 - averageEdgeLength) / double(edgeIndex + 1);
        edgeIndex++;
        //std::cout << averageEdgeLength << ", " << edgeIndex << std::endl;
    }
}
