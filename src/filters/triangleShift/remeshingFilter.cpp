#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "remeshingFilter.h"

ShapeBench::RemeshingFilterOutput ShapeBench::remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config) {
    ShapeDescriptor::cpu::Mesh remeshedMesh = internal::remeshMesh(scene.filteredSampleMesh, config);
    ShapeDescriptor::free(scene.filteredSampleMesh);
    scene.filteredSampleMesh = remeshedMesh;

    if(scene.filteredAdditiveNoise.vertexCount > 0) {
        ShapeDescriptor::cpu::Mesh remeshedAdditiveMesh = internal::remeshMesh(scene.filteredAdditiveNoise, config);
        ShapeDescriptor::free(scene.filteredAdditiveNoise);
        scene.filteredAdditiveNoise = remeshedAdditiveMesh;
    }

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

    malloc_trim(0);

    return output;
}

ShapeDescriptor::cpu::Mesh ShapeBench::internal::remeshMesh(const ShapeDescriptor::cpu::Mesh& mesh, const nlohmann::json &config) {



    /*std::vector<Vector3> convertedVertices_sampleMesh;
    std::vector<std::vector<size_t>> convertedIndices_sampleMesh;
    internal::createRemeshingMesh(mesh, convertedVertices_sampleMesh, convertedIndices_sampleMesh);
    IsotropicRemesher isotropicRemesher(&convertedVertices_sampleMesh, &convertedIndices_sampleMesh);

    isotropicRemesher.setTargetTriangleCount(mesh.vertexCount);
    uint32_t iterationCount = config.at("filterSettings").at("alternateTriangulation").at("remeshIterationCount");
    isotropicRemesher.remesh(iterationCount);*/
    ShapeDescriptor::cpu::Mesh remeshedMesh;// = internal::convertRemeshingToSDMesh(isotropicRemesher.remeshedHalfedgeMesh());
    return remeshedMesh;
}
/*
ShapeDescriptor::cpu::Mesh ShapeBench::internal::convertRemeshingToSDMesh(IsotropicHalfedgeMesh *halfedgeMesh) {
    std::vector<ShapeDescriptor::cpu::float3> vertices;
    std::vector<uint32_t> indices;

    size_t outputIndex = 0;
    for (IsotropicHalfedgeMesh::Vertex *vertex = halfedgeMesh->moveToNextVertex(nullptr); nullptr != vertex; vertex = halfedgeMesh->moveToNextVertex(vertex)) {
        vertices.emplace_back(vertex->position[0], vertex->position[1], vertex->position[2]);
        vertex->outputIndex = outputIndex++;
    }
    for (IsotropicHalfedgeMesh::Face *face = halfedgeMesh->moveToNextFace(nullptr); nullptr != face; face = halfedgeMesh->moveToNextFace(face)) {
        indices.push_back(face->halfedge->previousHalfedge->startVertex->outputIndex);
        indices.push_back(face->halfedge->startVertex->outputIndex);
        indices.push_back(face->halfedge->nextHalfedge->startVertex->outputIndex);
    }

    ShapeDescriptor::cpu::Mesh mesh(indices.size());
    for(uint32_t i = 0; i < indices.size(); i += 3) {
        ShapeDescriptor::cpu::float3 vertex0 = vertices.at(indices.at(i));
        ShapeDescriptor::cpu::float3 vertex1 = vertices.at(indices.at(i + 1));
        ShapeDescriptor::cpu::float3 vertex2 = vertices.at(indices.at(i + 2));

        ShapeDescriptor::cpu::float3 normal = ShapeDescriptor::computeTriangleNormal(vertex0, vertex1, vertex2);
        mesh.vertices[i] = vertex0;
        mesh.vertices[i + 1] = vertex1;
        mesh.vertices[i + 2] = vertex2;

        if(std::isnan(normal.x) || std::isnan(normal.y) || std::isnan(normal.z)) {
            normal = {1, 0, 0};
        }
        mesh.normals[i] = normal;
        mesh.normals[i + 1] = normal;
        mesh.normals[i + 2] = normal;
    }

    return mesh;
}

void ShapeBench::internal::createRemeshingMesh(const ShapeDescriptor::cpu::Mesh &mesh, std::vector<Vector3> &output_vertices,
                                          std::vector<std::vector<size_t>> &output_indices) {
    output_indices.resize(mesh.vertexCount / 3);
    output_vertices.reserve(mesh.vertexCount);

    std::unordered_map<ShapeDescriptor::cpu::float3, unsigned int> seenVerticesIndex;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        const ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
        if(!seenVerticesIndex.contains(vertex)) {
            // Vertex has not been seen before
            seenVerticesIndex.insert({vertex, output_vertices.size()});
            output_vertices.emplace_back(vertex.x, vertex.y, vertex.z);
        }
        output_indices.at(i/3).resize(3);
        output_indices.at(i/3).at(i % 3) = seenVerticesIndex.at(vertex);
    }

    //std::cout << mesh.vertexCount << " -> " << output_indices.size() << ", " << output_vertices.size() << std::endl;
}
*/