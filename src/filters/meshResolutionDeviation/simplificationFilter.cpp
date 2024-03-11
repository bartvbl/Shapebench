#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "simplificationFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing.h"
#include "pmp/algorithms/decimation.h"
#include "benchmarkCore/randomEngine.h"

pmp::SurfaceMesh convertSDMeshToPMP(const ShapeDescriptor::cpu::Mesh& mesh) {
    pmp::SurfaceMesh pmpMesh;
    for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 sourceVertex0 = mesh.vertices[i + 0];
        ShapeDescriptor::cpu::float3 sourceVertex1 = mesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 sourceVertex2 = mesh.vertices[i + 2];

        pmp::Vertex vertex0 = pmpMesh.add_vertex(pmp::Point(sourceVertex0.x, sourceVertex0.y, sourceVertex0.z));
        pmp::Vertex vertex1 = pmpMesh.add_vertex(pmp::Point(sourceVertex1.x, sourceVertex1.y, sourceVertex1.z));
        pmp::Vertex vertex2 = pmpMesh.add_vertex(pmp::Point(sourceVertex2.x, sourceVertex2.y, sourceVertex2.z));

        pmpMesh.add_triangle(vertex0, vertex1, vertex2);
    }
    return pmpMesh;
}

ShapeDescriptor::cpu::Mesh convertPMPMeshToSD(const pmp::SurfaceMesh& mesh) {
    uint32_t vertexCount = 3 * mesh.n_faces();
    ShapeDescriptor::cpu::Mesh outMesh(vertexCount);

    pmp::VertexProperty<pmp::Point> points = mesh.get_vertex_property<pmp::Point>("v:point");

    uint32_t nextVertexIndex = 0;
    for (auto f : mesh.faces()) {
        for (auto v: mesh.vertices(f)) {
            auto idx = v.idx();
            const pmp::Point p = points.vector().at(idx);
            assert(nextVertexIndex < vertexCount);
            outMesh.vertices[nextVertexIndex] = {p.data()[0], p.data()[1], p.data()[2]};
            nextVertexIndex++;
        }
    }
    return outMesh;
}

ShapeBench::MeshSimplificationFilterOutput ShapeBench::simplifyMesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config, uint64_t randomSeed) {
    // Convert to PMP Mesh
    pmp::SurfaceMesh sampleMesh = convertSDMeshToPMP(scene.filteredSampleMesh);
    pmp::SurfaceMesh additiveNoiseMesh = convertSDMeshToPMP(scene.filteredAdditiveNoise);

    float minVertexCountFactor = config.at("filterSettings").at("meshResolutionDeviation").at("minVertexCountFactor");

    ShapeBench::randomEngine engine(randomSeed);
    std::uniform_real_distribution<float> vertexCountScaleFactorDistribution(minVertexCountFactor, 1);

    float chosenVertexScaleFactor = vertexCountScaleFactorDistribution(engine);

    // Leif Kobbelt, Swen Campagna, and Hans-Peter Seidel. A general framework for mesh decimation. In Proceedings of Graphics Interface, pages 43–50, 1998.
    // Michael Garland and Paul Seagrave Heckbert. Surface simplification using quadric error metrics. In Proceedings of the 24th Annual Conference on Computer Graphics and Interactive Techniques, SIGGRAPH '97, pages 209–216, 1997.

    uint32_t referenceMeshVertexCount = uint32_t(chosenVertexScaleFactor * float(scene.filteredSampleMesh.vertexCount));
    uint32_t additiveNoiseMeshVertexCount = uint32_t(chosenVertexScaleFactor * float(scene.filteredAdditiveNoise.vertexCount));

    pmp::decimate(sampleMesh, referenceMeshVertexCount);
    pmp::decimate(additiveNoiseMesh, additiveNoiseMeshVertexCount);

    // Convert back to original format
    ShapeDescriptor::free(scene.filteredSampleMesh);
    ShapeDescriptor::free(scene.filteredAdditiveNoise);

    scene.filteredSampleMesh = convertPMPMeshToSD(sampleMesh);
    scene.filteredAdditiveNoise = convertPMPMeshToSD(additiveNoiseMesh);

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

    ShapeBench::MeshSimplificationFilterOutput output;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        nlohmann::json entry;
        entry["mesh-resolution-vertex-displacement"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
        entry["mesh-resolution-scale-factor"] = chosenVertexScaleFactor;
        output.metadata.push_back(entry);
    }

    return output;
}
