//#define CGAL_PMP_REMESHING_VERBOSE
//#define CGAL_PMP_REMESHING_VERBOSE_PROGRESS
#define CGAL_PMP_USE_CERES_SOLVER
#include <shapeDescriptor/shapeDescriptor.h>
#include <malloc.h>
#include "AlternateTriangulationFilter.h"
#include "utils/filterUtils/cgalConverter.h"
#include <vector>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/detect_features.h>
#include <CGAL/Polygon_mesh_processing/repair.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel   CGALK;
typedef CGAL::Surface_mesh<CGALK::Point_3>                    CGALMesh;
typedef boost::graph_traits<CGALMesh>::edge_descriptor            edge_descriptor;
namespace PMP = CGAL::Polygon_mesh_processing;


void remeshMesh(ShapeDescriptor::cpu::Mesh& meshToRemesh, uint32_t iterationCount, uint32_t& sharpEdgeCount) {
    CGALMesh mesh = ShapeBench::convertSDMeshToCGAL(meshToRemesh);

    // Constrain edges with a dihedral angle over 60°
    typedef boost::property_map<CGALMesh, CGAL::edge_is_feature_t>::type EIFMap;
    EIFMap eif = get(CGAL::edge_is_feature, mesh);
    PMP::detect_sharp_edges(mesh, 80, eif);
    for(edge_descriptor e : edges(mesh))
        if(get(eif, e))
            sharpEdgeCount++;

    // Smooth with both angle and area criteria + Delaunay flips
    PMP::angle_and_area_smoothing(mesh, CGAL::parameters::number_of_iterations(iterationCount)
            .use_safety_constraints(false) // authorize all moves
            .edge_is_constrained_map(eif));

    mesh.collect_garbage();

    ShapeDescriptor::free(meshToRemesh);
    meshToRemesh = ShapeBench::convertCGALMeshToSD(mesh);
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


void ShapeBench::AlternateTriangulationFilter::init(const nlohmann::json &config, bool invalidateCaches) {

}

void ShapeBench::AlternateTriangulationFilter::destroy() {

}

void ShapeBench::AlternateTriangulationFilter::saveCaches(const nlohmann::json& config) {

}

ShapeBench::FilterOutput
ShapeBench::AlternateTriangulationFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                                const Dataset &dataset,
                                                ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    if(scene.filteredSampleMesh.vertexCount > config.at("filterSettings").at("alternateTriangulation").at("triangleLimit")) {
        //throw std::runtime_error("Mesh too large!");
    }

    ShapeBench::FilterOutput output;
    {
        uint32_t iterationCount = config.at("filterSettings").at("alternateTriangulation").at("remeshIterationCount");

        uint32_t initialVertexCount = scene.filteredSampleMesh.vertexCount;

        // Determine average edge length (used in the output results as a means to denote object scale)
        double averageEdgeLength = 0;
        uint32_t edgeIndex = 0;
        internal::calculateAverageEdgeLength(scene.filteredSampleMesh, averageEdgeLength, edgeIndex);
        internal::calculateAverageEdgeLength(scene.filteredAdditiveNoise, averageEdgeLength, edgeIndex);

        // Do remeshing
        uint32_t sampleSharpEdgeCount = 0;
        uint32_t additiveNoiseSharpEdgeCount = 0;
        remeshMesh(scene.filteredSampleMesh, iterationCount, sampleSharpEdgeCount);

        if (scene.filteredAdditiveNoise.vertexCount != 0) {
            remeshMesh(scene.filteredAdditiveNoise, iterationCount, additiveNoiseSharpEdgeCount);
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
            entry["triangle-shift-displacement-distance"] = length(scene.mappedReferenceVertices.at(i).vertex - originalReferenceVertices.at(i).vertex);
            entry["triangle-shift-average-edge-length"] = averageEdgeLength;
            entry["triangle-shift-sample-sharp-edge-count"] = sampleSharpEdgeCount;
            entry["triangle-shift-additive-noise-sharp-edge-count"] = additiveNoiseSharpEdgeCount;
            entry["triangle-shift-sample-mesh-initial-vertexcount"] = initialVertexCount;
            entry["triangle-shift-sample-mesh-filtered-vertexcount"] = scene.filteredSampleMesh.vertexCount;
            output.metadata.push_back(entry);
        }

        //ShapeDescriptor::writeOBJ(scene.filteredSampleMesh, "remeshed.obj");
    }
    malloc_trim(0);

    // The mesh itself does not move, so we don't modify these values
    // They're included here for the sake of completion
    scene.sampleMeshTransformation *= glm::mat4(1.0);
    scene.sampleMeshNormalTransformation *= glm::mat3(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::mat4(1.0);
    }

    return output;
}

ShapeBench::FilterOutput
ShapeBench::AlternateTriangulationFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                                      ShapeBench::FilteredMeshPair &scene,
                                                      const ShapeBench::Dataset &dataset,
                                                      ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::AlternateTriangulationFilter::getFilterName() const {
    return "alternate-triangulation";
}
