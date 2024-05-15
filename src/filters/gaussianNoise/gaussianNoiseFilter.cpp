#include "gaussianNoiseFilter.h"
#include "benchmarkCore/randomEngine.h"
#include "glm/glm.hpp"
#include "glm/ext/matrix_transform.hpp"

void applyGaussianNoise(ShapeDescriptor::cpu::Mesh& mesh, uint64_t randomSeed, float maxDeviation) {
    if(mesh.vertexCount == 0 || mesh.vertices == nullptr) {
        return;
    }
    std::vector<unsigned int> vertexIndexBuffer(mesh.vertexCount);

    ShapeBench::randomEngine engine(randomSeed);
    std::normal_distribution<float> distribution(0, maxDeviation);

    std::vector<ShapeDescriptor::cpu::float3> condensedVertices;
    std::vector<ShapeDescriptor::cpu::float3> normalSums;
    std::vector<ShapeDescriptor::cpu::float3> lastNormals;

    condensedVertices.reserve(mesh.vertexCount);
    normalSums.reserve(mesh.vertexCount);
    lastNormals.reserve(mesh.vertexCount);

    std::unordered_map<ShapeDescriptor::cpu::float3, unsigned int> seenVerticesIndex;

    for(unsigned int i = 0; i < mesh.vertexCount; i++) {
        const ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
        if(std::isnan(vertex.x) || std::isnan(vertex.y) || std::isnan(vertex.z)
        || std::isinf(vertex.x) || std::isinf(vertex.y) || std::isinf(vertex.z)) {
            continue;
        }
        if(!seenVerticesIndex.contains(vertex)) {
            // Vertex has not been seen before
            seenVerticesIndex.insert({vertex, condensedVertices.size()});
            condensedVertices.push_back(vertex);
            normalSums.emplace_back(0, 0, 0);
            lastNormals.emplace_back(0, 0, 0);
        }
        uint32_t vertexIndex = seenVerticesIndex.at(vertex);
        normalSums.at(vertexIndex) += mesh.normals[i];
        lastNormals.at(vertexIndex) = mesh.normals[i];
        vertexIndexBuffer.at(i) = vertexIndex;
    }

    for(uint32_t i = 0; i < condensedVertices.size(); i++) {
        ShapeDescriptor::cpu::float3 direction = normalSums.at(i);
        if(direction.x == 0 && direction.y == 0 && direction.z == 0) {
            // Normals sum up to 0, so just pick one of them
            direction = lastNormals.at(i);
        }
        direction = normalize(direction);
        float displacement = distribution(engine);
        condensedVertices.at(i) += displacement * direction;
    }

    // Updated mesh vertices with displaced vertices
    for(uint32_t i = 0; i < mesh.vertexCount; i++) {
        mesh.vertices[i] = condensedVertices.at(vertexIndexBuffer.at(i));
    }
}

ShapeBench::FilterOutput ShapeBench::GaussianNoiseFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const ShapeBench::Dataset &dataset, uint64_t randomSeed) {
    ShapeBench::FilterOutput meta;

    float minStandardDeviation = config.at("filterSettings").at("gaussianNoise").at("minStandardDeviation");
    float maxStandardDeviation = config.at("filterSettings").at("gaussianNoise").at("maxStandardDeviation");

    ShapeBench::randomEngine engine(randomSeed);

    std::uniform_real_distribution<float> intensityDistribution(minStandardDeviation, maxStandardDeviation);
    float deviation = intensityDistribution(engine);

    applyGaussianNoise(scene.filteredSampleMesh, engine(), deviation);
    applyGaussianNoise(scene.filteredAdditiveNoise, engine(), deviation);

    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        if(!scene.mappedVertexIncluded.at(i)) {
            nlohmann::json metadataEntry;
            metadataEntry["gaussian-noise-max-deviation"] = deviation;
            meta.metadata.push_back(metadataEntry);
            continue;
        }
        ShapeDescriptor::cpu::float3 originalVertex = scene.mappedReferenceVertices.at(i).vertex;

        // Update vertex location to the displaced location
        scene.mappedReferenceVertices.at(i).vertex = scene.filteredSampleMesh.vertices[scene.mappedReferenceVertexIndices.at(i)];

        nlohmann::json metadataEntry;
        metadataEntry["gaussian-noise-max-deviation"] = deviation;
        metadataEntry["gaussian-noise-vertex-deviation"] = length(originalVertex - scene.mappedReferenceVertices.at(i).vertex);
        meta.metadata.push_back(metadataEntry);
    }

    // The mesh itself does not move, so we don't modify these values
    // They're included here for the sake of completion
    scene.sampleMeshTransformation *= glm::mat4(1.0);
    for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
        scene.additiveNoiseInfo.at(i).transformation *= glm::mat4(1.0);
    }


    //std::cout << randomSeed << " -> " << deviation << std::endl;

    return meta;
}

void ShapeBench::GaussianNoiseFilter::init(const nlohmann::json &config) {

}

void ShapeBench::GaussianNoiseFilter::destroy() {

}

void ShapeBench::GaussianNoiseFilter::saveCaches(const nlohmann::json& config) {

}
