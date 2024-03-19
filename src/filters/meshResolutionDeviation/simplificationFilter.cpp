#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "simplificationFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/decimation.h"
#include "benchmarkCore/randomEngine.h"
#include "utils/filterUtils/pmpConverter.h"


ShapeBench::MeshSimplificationFilterOutput ShapeBench::simplifyMesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config, uint64_t randomSeed) {
    // Convert to PMP Mesh
    pmp::SurfaceMesh sampleMesh;
    uint32_t sampleRemovedCount = 0;
    ShapeBench::convertSDMeshToPMP(scene.filteredSampleMesh, sampleMesh, &sampleRemovedCount);
    uint32_t additiveRemovedCount = 0;
    pmp::SurfaceMesh additiveNoiseMesh;
    ShapeBench::convertSDMeshToPMP(scene.filteredAdditiveNoise, additiveNoiseMesh, &additiveRemovedCount);

    float minVertexCountFactor = config.at("filterSettings").at("meshResolutionDeviation").at("minVertexCountFactor");

    ShapeBench::randomEngine engine(randomSeed);
    std::uniform_real_distribution<float> vertexCountScaleFactorDistribution(minVertexCountFactor, 1);

    float chosenVertexScaleFactor = vertexCountScaleFactorDistribution(engine);

    // Leif Kobbelt, Swen Campagna, and Hans-Peter Seidel. A general framework for mesh decimation. In Proceedings of Graphics Interface, pages 43–50, 1998.
    // Michael Garland and Paul Seagrave Heckbert. Surface simplification using quadric error metrics. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH '97, pages 209–216, 1997.

    uint32_t referenceMeshVertexCount = uint32_t(chosenVertexScaleFactor * float(sampleMesh.n_vertices()));
    uint32_t additiveNoiseMeshVertexCount = uint32_t(chosenVertexScaleFactor * float(additiveNoiseMesh.n_vertices())) / 3;

    pmp::decimate(sampleMesh, referenceMeshVertexCount, 10, 0, 0, 180, 0, 0, 0);
    pmp::decimate(additiveNoiseMesh, additiveNoiseMeshVertexCount , 10, 0, 0, 180, 0, 0, 0);

    // Convert back to original format
    ShapeDescriptor::free(scene.filteredSampleMesh);
    ShapeDescriptor::free(scene.filteredAdditiveNoise);

    scene.filteredSampleMesh = ShapeBench::convertPMPMeshToSD(sampleMesh);
    scene.filteredAdditiveNoise = ShapeBench::convertPMPMeshToSD(additiveNoiseMesh);
    //std::cout << "Simplification: " << scene.originalMesh.vertexCount << " -> " << scene.filteredSampleMesh.vertexCount << " vs target " << 3 * referenceMeshVertexCount << " / " << (chosenVertexScaleFactor * scene.originalMesh.vertexCount) << std::endl;
    //ShapeDescriptor::writeOBJ(scene.filteredSampleMesh, "result.obj");

    // Update reference points
    std::vector<float> bestDistances(scene.mappedReferenceVertices.size(), std::numeric_limits<float>::max());
    std::vector<ShapeDescriptor::OrientedPoint> originalReferenceVertices = scene.mappedReferenceVertices;

    for (uint32_t meshVertexIndex = 0; meshVertexIndex < scene.filteredSampleMesh.vertexCount; meshVertexIndex++) {
        ShapeDescriptor::cpu::float3 meshVertex = scene.filteredSampleMesh.vertices[meshVertexIndex];
        for (uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
            ShapeDescriptor::OrientedPoint referencePoint = originalReferenceVertices.at(i);
            ShapeDescriptor::cpu::float3 referenceVertex = referencePoint.vertex;
            float distanceToReferenceVertex = length(referenceVertex - meshVertex);
            if (distanceToReferenceVertex < bestDistances.at(i)) {
                bestDistances.at(i) = distanceToReferenceVertex;
                scene.mappedReferenceVertices.at(i).vertex = meshVertex;
                scene.mappedReferenceVertices.at(i).normal = scene.filteredSampleMesh.normals[meshVertexIndex];
                scene.mappedReferenceVertexIndices.at(i) = meshVertexIndex;
            }
        }
    }


    ShapeBench::MeshSimplificationFilterOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["mesh-resolution-vertex-displacement"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
        entry["mesh-resolution-scale-factor"] = chosenVertexScaleFactor;
        entry["mesh-resolution-removed-count-sample"] = sampleRemovedCount;
        entry["mesh-resolution-removed-count-additive"] = additiveRemovedCount;
        output.metadata.push_back(entry);
    }

    return output;
}
