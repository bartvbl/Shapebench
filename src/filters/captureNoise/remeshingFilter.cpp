#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "remeshingFilter.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing.h"

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
    ShapeDescriptor::cpu::Mesh outMesh(mesh.n_faces());

    pmp::VertexProperty<pmp::Point> points = mesh.get_vertex_property<pmp::Point>("v:point");

    uint32_t nextVertexIndex = 0;
    for (auto f : mesh.faces()) {
        for (auto v: mesh.vertices(f)) {
            auto idx = v.idx();
            const pmp::Point p = points.vector().at(idx);
            assert(nextVertexIndex < mesh.vertexCount);
            outMesh.vertices[nextVertexIndex] = {p.data()[0], p.data()[1], p.data()[2]};
            nextVertexIndex++;
        }
    }
    return outMesh;
}

void ShapeBench::remesh(ShapeBench::FilteredMeshPair& scene) {
    // Convert to PMP Mesh
    pmp::SurfaceMesh sampleMesh = convertSDMeshToPMP(scene.filteredSampleMesh);
    pmp::SurfaceMesh additiveNoiseMesh = convertSDMeshToPMP(scene.filteredAdditiveNoise);

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

    scene.filteredSampleMesh = convertPMPMeshToSD(sampleMesh);
    scene.filteredAdditiveNoise = convertPMPMeshToSD(additiveNoiseMesh);
}