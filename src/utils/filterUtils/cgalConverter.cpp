#include "cgalConverter.h"
class SDMeshWriter
{
    bool hasNormals;
    std::vector<ShapeDescriptor::cpu::float3> tempVertexBuffer;
    std::vector<ShapeDescriptor::cpu::float3> tempNormalBuffer;
    uint32_t nextVertexIndex = 0;
    uint32_t verticesPerFace = 0;

public:
    ShapeDescriptor::cpu::Mesh mesh;

    SDMeshWriter() {}

    void write_header(std::ostream& os,
                      std::size_t vertices,
                      std::size_t /*halfedges*/,
                      std::size_t facets,
                      bool colors = false,
                      bool normals = false,
                      bool textures = false) {
        mesh = ShapeDescriptor::cpu::Mesh(3 * facets);
        hasNormals = normals;
        tempVertexBuffer.reserve(vertices);
        nextVertexIndex = 0;
    }

    void write_footer() {}

    void write_vertex(const double x, const double y, const double z) {
        tempVertexBuffer.emplace_back(x, y, z);
    }

    void write_vertex_normal(const double x, const double y, const double z) {
        tempNormalBuffer.emplace_back(x, y, z);
    }

    void write_vertex_color(const double r, const double g, const double b) {}
    void write_vertex_texture(const double tx, const double ty) {}
    void write_facet_header() {}

    void write_face_color(const double r, const double g, const double b) {}

    void write_facet_begin(std::size_t) {
        verticesPerFace = 0;
    }
    void write_facet_vertex_index(std::size_t idx) {
        mesh.vertices[nextVertexIndex] = tempVertexBuffer.at(idx);
        nextVertexIndex++;
        verticesPerFace++;
    }
    void write_facet_end() {
        assert(verticesPerFace == 3);
    }
};

class SnitchingPrinter : public CGAL::IO::internal::Generic_facegraph_printer<std::ostream, CGALMesh, SDMeshWriter> {
public:
    SnitchingPrinter(std::ostream& os, SDMeshWriter writer) : CGAL::IO::internal::Generic_facegraph_printer<std::ostream, CGALMesh, SDMeshWriter>(os, writer) { }
    ShapeDescriptor::cpu::Mesh getMesh() {
        return m_writer.mesh;
    }
};

CGALMesh ShapeBench::convertSDMeshToCGAL(const ShapeDescriptor::cpu::Mesh &mesh, uint32_t *removedCount) {
    CGALMesh outputMesh;

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<unsigned int> vertexIndexBuffer(mesh.vertexCount);
    std::vector<CGALMesh::Vertex_index> cgalIndexBuffer;
    cgalIndexBuffer.reserve(mesh.vertexCount);

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

    for(unsigned int i = 0; i < condensedVertices.size(); i++) {
        ShapeDescriptor::cpu::float3 vertex = condensedVertices.at(i);
        cgalIndexBuffer.push_back(outputMesh.add_vertex(CGALMesh::Point(vertex.x, vertex.y, vertex.z)));
    }

    uint32_t numComplaints = 0;
    for(uint32_t i = 0; i < mesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = mesh.vertices[i];
        ShapeDescriptor::cpu::float3 vertex1 = mesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = mesh.vertices[i + 2];
        if(ShapeDescriptor::computeTriangleArea(vertex0, vertex1, vertex2) == 0) {
            continue;
        }

        CGALMesh::Vertex_index index0 = cgalIndexBuffer.at(vertexIndexBuffer.at(i));
        CGALMesh::Vertex_index index1 = cgalIndexBuffer.at(vertexIndexBuffer.at(i + 1));
        CGALMesh::Vertex_index index2 = cgalIndexBuffer.at(vertexIndexBuffer.at(i + 2));

        outputMesh.add_face(index0, index1, index2);
    }

    return outputMesh;
}

ShapeDescriptor::cpu::Mesh ShapeBench::convertCGALMeshToSD(const CGALMesh &cgalMesh) {
    std::stringstream stream;
    SDMeshWriter writer;
    SnitchingPrinter printer(stream, writer);
    printer(cgalMesh, CGAL::parameters::all_default());

    ShapeDescriptor::cpu::Mesh outMesh = printer.getMesh();

    for(uint32_t i = 0; i < outMesh.vertexCount; i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = outMesh.vertices[i];
        ShapeDescriptor::cpu::float3 vertex1 = outMesh.vertices[i + 1];
        ShapeDescriptor::cpu::float3 vertex2 = outMesh.vertices[i + 2];

        ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);
        if(std::isnan(normal.x) || std::isnan(normal.y) || std::isnan(normal.z)) {
            normal = {1, 0, 0};
        }
        outMesh.normals[i] = normal;
        outMesh.normals[i + 1] = normal;
        outMesh.normals[i + 2] = normal;
    }

    return outMesh;
}
