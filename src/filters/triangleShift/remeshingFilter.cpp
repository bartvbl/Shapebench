//#define CGAL_PMP_REMESHING_VERBOSE
//#define CGAL_PMP_REMESHING_VERBOSE_PROGRESS
#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "remeshingFilter.h"
#include "utils/filterUtils/cgalConverter.h"
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <vector>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <boost/iterator/function_output_iterator.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/surface_Delaunay_remeshing.h>
#include <CGAL/Mesh_constant_domain_field_3.h>
#include <CGAL/Polygon_mesh_processing/repair.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel   CGALK;
typedef CGAL::Surface_mesh<CGALK::Point_3>                    CGALMesh;
typedef boost::graph_traits<CGALMesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<CGALMesh>::edge_descriptor            edge_descriptor;
namespace PMP = CGAL::Polygon_mesh_processing;
struct halfedge2edge
{
    halfedge2edge(const CGALMesh& m, std::vector<edge_descriptor>& edges)
            : m_mesh(m), m_edges(edges)
    {}
    void operator()(const halfedge_descriptor& h) const
    {
        m_edges.push_back(edge(h, m_mesh));
    }
    const CGALMesh& m_mesh;
    std::vector<edge_descriptor>& m_edges;
};

void remeshMesh(ShapeDescriptor::cpu::Mesh& meshToRemesh, double targetEdgeLength, uint32_t iterationCount) {
    CGALMesh mesh = ShapeBench::convertSDMeshToCGAL(meshToRemesh);

    std::vector<edge_descriptor> border;
    PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
    PMP::split_long_edges(border, targetEdgeLength, mesh);
    CGAL::Polygon_mesh_processing::remove_isolated_vertices(mesh);
    CGAL::Polygon_mesh_processing::duplicate_non_manifold_vertices(mesh);

    if(!CGAL::is_valid_polygon_mesh(mesh)) {
        throw std::runtime_error("Mesh invalid");
    }

    PMP::isotropic_remeshing(faces(mesh), targetEdgeLength, mesh,
                             CGAL::parameters::number_of_iterations(iterationCount)
                                     .protect_constraints(false)); //i.e. protect border, here

    mesh.collect_garbage();

    /*
     * using EIFMap = boost::property_map<CGALMesh, CGAL::edge_is_feature_t>::type;
    EIFMap eif = get(CGAL::edge_is_feature, mesh);
    PMP::detect_sharp_edges(mesh, 45, eif);
     * CGALMesh outmesh = PMP::surface_Delaunay_remeshing(mesh,
                                                   CGAL::parameters::mesh_edge_size(targetEdgeLength/15.0)
                                                           .mesh_facet_distance(targetEdgeLength/6.0)
                                                           .number_of_iterations(iterationCount));
*/
    //mesh.collect_garbage();
    ShapeDescriptor::free(meshToRemesh);
    meshToRemesh = ShapeBench::convertCGALMeshToSD(mesh);
}



ShapeBench::RemeshingFilterOutput ShapeBench::remesh(ShapeBench::FilteredMeshPair& scene, const nlohmann::json& config) {
    /*if(scene.filteredSampleMesh.vertexCount > 1000000) {
        throw std::runtime_error("Mesh too large!");
    }*/

    ShapeBench::RemeshingFilterOutput output;
    {
        float minEdgeLength = config.at("filterSettings").at("alternateTriangulation").at("minEdgeLength");
        float maxEdgeLength = config.at("filterSettings").at("alternateTriangulation").at("maxEdgeLength");
        uint32_t iterationCount = config.at("filterSettings").at("alternateTriangulation").at("remeshIterationCount");

        uint32_t initialVertexCount = scene.filteredSampleMesh.vertexCount;

        // Determine target edge length
        double averageEdgeLength = 0;
        uint32_t edgeIndex = 0;
        internal::calculateAverageEdgeLength(scene.filteredSampleMesh, averageEdgeLength, edgeIndex);
        internal::calculateAverageEdgeLength(scene.filteredAdditiveNoise, averageEdgeLength, edgeIndex);
        averageEdgeLength = std::clamp<double>(averageEdgeLength, minEdgeLength, maxEdgeLength);


        // Do remeshing
        remeshMesh(scene.filteredSampleMesh, averageEdgeLength, iterationCount);

        if (scene.filteredAdditiveNoise.vertexCount != 0) {
            remeshMesh(scene.filteredAdditiveNoise, averageEdgeLength, iterationCount);
        }

        // Update corresponding vertices

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


        for (uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
            nlohmann::json entry;
            entry["triangle-shift-displacement-distance"] = length(scene.mappedReferenceVertexIndices.at(i) - originalReferenceVertices.at(i).vertex);
            entry["triangle-shift-average-edge-length"] = averageEdgeLength;
            entry["triangle-shift-sample-mesh-initial-vertexcount"] = initialVertexCount;
            entry["triangle-shift-sample-mesh-filtered-vertexcount"] = scene.filteredSampleMesh.vertexCount;
            output.metadata.push_back(entry);
        }

        //ShapeDescriptor::writeOBJ(scene.filteredSampleMesh, "remeshed.obj");
    }
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


