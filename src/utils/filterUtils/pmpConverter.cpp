#include <shapeDescriptor/shapeDescriptor.h>
#include "pmpConverter.h"
#include "pmp/surface_mesh.h"
#include "pmp/io/read_obj.h"

void ShapeBench::convertSDMeshToPMP(const ShapeDescriptor::cpu::Mesh& mesh, pmp::SurfaceMesh& pmpMesh, uint32_t* removedCount) {
    std::vector<pmp::Vertex> vertices(3);

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<unsigned int> vertexIndexBuffer(mesh.vertexCount);

    condensedVertices.reserve(mesh.vertexCount);
    std::unordered_map<ShapeDescriptor::cpu::float3, unsigned int> seenVerticesIndex;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        const ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
        if(!seenVerticesIndex.contains(vertex)) {
            // Vertex has not been seen before
            seenVerticesIndex.insert({vertex, condensedVertices.size()});
            condensedVertices.push_back(vertex);
        }
        vertexIndexBuffer.at(i) = seenVerticesIndex.at(vertex);
    }

    for(uint32_t i = 0; i < condensedVertices.size(); i++) {
        ShapeDescriptor::cpu::float3 vertex = condensedVertices.at(i);
        pmpMesh.add_vertex(pmp::Point(vertex.x, vertex.y, vertex.z));
    }

    uint32_t numComplaints = 0;
    for(uint32_t i = 0; i < vertexIndexBuffer.size(); i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = condensedVertices.at(vertexIndexBuffer.at(i));
        ShapeDescriptor::cpu::float3 vertex1 = condensedVertices.at(vertexIndexBuffer.at(i + 1));
        ShapeDescriptor::cpu::float3 vertex2 = condensedVertices.at(vertexIndexBuffer.at(i + 2));
        if(ShapeDescriptor::computeTriangleArea(vertex0, vertex1, vertex2) == 0) {
            continue;
        }

        vertices.clear();
        vertices.emplace_back(vertexIndexBuffer.at(i));
        vertices.emplace_back(vertexIndexBuffer.at(i + 1));
        vertices.emplace_back(vertexIndexBuffer.at(i + 2));
        try {
            pmpMesh.add_face(vertices);
        } catch(const pmp::TopologyException& e) {
            // ignore
            numComplaints++;
        }
    }

    if(removedCount != nullptr) {
        *removedCount = numComplaints;
    }
    //std::cout << "Complaint count: " << numComplaints << std::endl;
    //std::cout << "Reduced index buffer from " << mesh.vertexCount << " to " << condensedVertices.size() << std::endl;
}

ShapeDescriptor::cpu::Mesh ShapeBench::convertPMPMeshToSD(const pmp::SurfaceMesh& mesh) {
    uint32_t vertexCount = 3 * mesh.n_faces();
    ShapeDescriptor::cpu::Mesh outMesh(vertexCount);

    pmp::VertexProperty<pmp::Point> points = mesh.get_vertex_property<pmp::Point>("v:point");

    uint32_t nextVertexIndex = 0;
    uint32_t removedFaceCount = 0;
    for (auto f : mesh.faces()) {
        if(f.is_valid()) {
            // Check whether face is valid, otherwise remove the triangle
            bool canBeAdded = true;
            for (auto v: mesh.vertices(f)) {
                pmp::IndexType idx = v.idx();
                if(idx >= points.vector().size()) {
                    canBeAdded = false;
                    break;
                }
            }
            if(!canBeAdded) {
                removedFaceCount++;
                continue;
            }
            // Add triangle to the mesh
            for (auto v: mesh.vertices(f)) {
                pmp::IndexType idx = v.idx();
                const pmp::Point p = points.vector().at(idx);
                assert(nextVertexIndex < vertexCount);
                outMesh.vertices[nextVertexIndex] = {p.data()[0], p.data()[1], p.data()[2]};
                nextVertexIndex++;
            }
        }
    }

    outMesh.vertexCount -= 3 * removedFaceCount;
    if(removedFaceCount > 0) {
        std::cout << "Removed " << removedFaceCount << " faces." << std::endl;
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
