#include <shapeDescriptor/shapeDescriptor.h>
#include "pmpConverter.h"
#include "pmp/surface_mesh.h"

pmp::SurfaceMesh ShapeBench::convertSDMeshToPMP(const ShapeDescriptor::cpu::Mesh& mesh) {
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

ShapeDescriptor::cpu::Mesh ShapeBench::convertPMPMeshToSD(const pmp::SurfaceMesh& mesh) {
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

    for(uint32_t i = 0; i < outMesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = outMesh.vertices[i];
        ShapeDescriptor::cpu::float3 vertex1 = outMesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = outMesh.vertices[i + 2];
        ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);
        outMesh.normals[i] = normal;
        outMesh.normals[i + 1] = normal;
        outMesh.normals[i + 2] = normal;
    }

    return outMesh;
}
