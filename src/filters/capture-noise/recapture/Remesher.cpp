#include <iostream>
#include <shapeDescriptor/shapeDescriptor.h>
#include "Remesher.h"
#include "pmp/surface_mesh.h"
#include "pmp/algorithms/remeshing.h"

void remesh(Shapebench::FiltereredMeshPair& scene) {
    // Convert to PMP Mesh
    pmp::SurfaceMesh pmpMesh;
    for(uint32_t i = 0; i < scene.alteredMesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 sourceVertex0 = scene.alteredMesh.vertices[i + 0];
        ShapeDescriptor::cpu::float3 sourceVertex1 = scene.alteredMesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 sourceVertex2 = scene.alteredMesh.vertices[i + 2];

        pmp::Vertex vertex0 = pmpMesh.add_vertex(pmp::Point(sourceVertex0.x, sourceVertex0.y, sourceVertex0.z));
        pmp::Vertex vertex1 = pmpMesh.add_vertex(pmp::Point(sourceVertex1.x, sourceVertex1.y, sourceVertex1.z));
        pmp::Vertex vertex2 = pmpMesh.add_vertex(pmp::Point(sourceVertex2.x, sourceVertex2.y, sourceVertex2.z));

        pmpMesh.add_triangle(vertex0, vertex1, vertex2);
    }

    // Using the same approach as PMP library's remeshing tool
    pmp::Scalar averageEdgeLength = 0;
    uint32_t edgeIndex = 0;
    for (const auto& edgeInMesh : pmpMesh.edges()) {
        averageEdgeLength += distance(pmpMesh.position(pmpMesh.vertex(edgeInMesh, 0)),
                                      pmpMesh.position(pmpMesh.vertex(edgeInMesh, 1))) / pmp::Scalar(edgeIndex + 1);
    }

    // Mario Botsch and Leif Kobbelt. A remeshing approach to multiresolution modeling. In Proceedings of Eurographics Symposium on Geometry Processing, pages 189–96, 2004.
    pmp::uniform_remeshing(pmpMesh, averageEdgeLength);

    // Convert back to original format
    pmp::VertexProperty<pmp::Point> points = pmpMesh.get_vertex_property<pmp::Point>("v:point");
    uint32_t nextVertexIndex = 0;
    for (auto f : pmpMesh.faces()) {
        for (auto v: pmpMesh.vertices(f)) {
            auto idx = v.idx();
            const pmp::Point p = points.vector().at(idx);
            scene.alteredMesh.vertices[nextVertexIndex] = {p.data()[0], p.data()[1], p.data()[2]};
            nextVertexIndex++;
            assert(nextVertexIndex <= scene.alteredMesh.vertexCount);
        }
    }
}
