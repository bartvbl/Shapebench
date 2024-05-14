#include "Dataset.h"
#include <fstream>
#include <random>
#include <iostream>
#include <unordered_set>
#include "benchmarkCore/randomEngine.h"
#include "nlohmann/json.hpp"
#include "benchmarkCore/constants.h"
#include "benchmarkCore/CompressedDatasetCreator.h"
#include "benchmarkCore/MissingBenchmarkConfigurationException.h"

void ShapeBench::Dataset::loadCache(const nlohmann::json& cacheFileContents) {
    assert(cacheFileContents.contains("files"));
    uint32_t fileCount = cacheFileContents.at("files").size();
    entries.reserve(fileCount);

    uint32_t excludedCount = 0;
    for(uint32_t i = 0; i < fileCount; i++) {
        nlohmann::json jsonEntry = cacheFileContents.at("files").at(i);
        bool isPointCloud = jsonEntry.at("isPointCloud");
        bool isNotEmpty = jsonEntry.at("vertexCount") > 0;
        bool parseFailed = jsonEntry.contains("parseFailed") && jsonEntry.at("parseFailed");
        // Exclude all point clouds, empty meshes, and meshes that failed to parse
        if(!isPointCloud && isNotEmpty && !parseFailed) {
            DatasetEntry entry;
            entry.vertexCount = jsonEntry.at("vertexCount");
            entry.id = jsonEntry.at("id");
            entry.computedObjectRadius = jsonEntry.at("boundingSphereRadius");
            entry.computedObjectCentre = {jsonEntry.at("boundingSphereCentre")[0], jsonEntry.at("boundingSphereCentre")[1], jsonEntry.at("boundingSphereCentre")[2]};
            entry.meshFile = std::string(jsonEntry.at("filePath"));
            entries.push_back(entry);
        } else {
            excludedCount++;
        }
    }
    std::cout << "    Dataset contains " << entries.size() << " meshes." << std::endl;
    std::cout << "    Excluded " << excludedCount << " meshes and/or point clouds." << std::endl;
    std::sort(entries.begin(), entries.end());
}

std::vector<ShapeBench::VertexInDataset> ShapeBench::Dataset::sampleVertices(uint64_t randomSeed, uint32_t count, uint32_t verticesPerObject) const {
    assert(count % verticesPerObject == 0);
    std::vector<uint32_t> sampleHistogram(entries.size());
    std::vector<VertexInDataset> sampledEntries(count);

    ShapeBench::randomEngine engine(randomSeed);
    std::uniform_int_distribution<uint32_t> distribution(0, entries.size() - 1);

    uint32_t objectsToSample = count / verticesPerObject;

    for(uint32_t i = 0; i < objectsToSample; i++) {
        uint32_t chosenMeshIndex = distribution(engine);
        sampleHistogram.at(chosenMeshIndex)++;
    }
    uint32_t nextIndex = 0;
    std::unordered_set<uint32_t> seenVertexIndices;
    for(uint32_t i = 0; i < entries.size(); i++) {
        seenVertexIndices.clear();
        for(uint32_t j = 0; j < sampleHistogram.at(i) * verticesPerObject; j++) {
            sampledEntries.at(nextIndex).meshID = i;
            std::uniform_int_distribution<uint32_t> vertexIndexDistribution(0, entries.at(i).vertexCount - 1);
            uint32_t chosenVertexIndex = vertexIndexDistribution(engine);
            // Avoid duplicate vertices, but only if we need to pick more than one vertex,
            // and if selecting unique vertices is possible
            if(sampleHistogram.at(i) * verticesPerObject > 1 && sampleHistogram.at(i) * verticesPerObject < entries.at(i).vertexCount) {
                while(seenVertexIndices.contains(chosenVertexIndex)) {
                    chosenVertexIndex = vertexIndexDistribution(engine);
                }
                seenVertexIndices.insert(chosenVertexIndex);
            }
            sampledEntries.at(nextIndex).vertexIndex = chosenVertexIndex;
            nextIndex++;
        }
    }
    assert(nextIndex == count);
    return sampledEntries;
}

std::vector<ShapeBench::VertexInDataset> ShapeBench::Dataset::resampleVerticesInSameObject(unsigned long randomSeed, const uint32_t verticesPerObject, const std::vector<ShapeBench::VertexInDataset>& originalVertexList) const {
    std::vector<VertexInDataset> alteredVertexList = originalVertexList;

    ShapeBench::randomEngine engine(randomSeed);
    uint32_t objectCount = originalVertexList.size() / verticesPerObject;
    std::unordered_set<uint32_t> seenVertexIndices;
    for(uint32_t objectIndex = 0; objectIndex < objectCount; objectIndex++) {
        seenVertexIndices.clear();
        uint32_t objectBaseIndex = objectIndex * verticesPerObject;
        uint32_t objectMeshID = originalVertexList.at(objectBaseIndex).meshID;
        uint32_t objectVertexCount = entries.at(objectMeshID).vertexCount;
        std::uniform_int_distribution<uint32_t> objectVertexDistribution(0, objectVertexCount - 1);
        for(uint32_t vertexIndex = 0; vertexIndex < verticesPerObject; vertexIndex++) {
            uint32_t chosenVertexIndex = objectVertexDistribution(engine);
            if(verticesPerObject > 1 && verticesPerObject < objectVertexCount) {
                while(seenVertexIndices.contains(chosenVertexIndex)) {
                    chosenVertexIndex = objectVertexDistribution(engine);
                }
                seenVertexIndices.insert(chosenVertexIndex);
            }
            alteredVertexList.at(objectBaseIndex + vertexIndex).vertexIndex = chosenVertexIndex;
        }
    }

    return alteredVertexList;
}

const ShapeBench::DatasetEntry &ShapeBench::Dataset::at(uint32_t meshID) const {
    return entries.at(meshID);
}

bool ShapeBench::DatasetEntry::operator<(DatasetEntry &other) {
    return id < other.id;
}

ShapeBench::Dataset ShapeBench::Dataset::computeOrLoadCache(
        const nlohmann::json& configuration,
        const std::filesystem::path& cacheDirectory) {
    if(!configuration.at("datasetSettings").contains("compressedRootDir")) {
        throw ShapeBench::MissingBenchmarkConfigurationException("compressedRootDir");
    }
    const std::filesystem::path baseDatasetDirectory = configuration.at("datasetSettings").at("objaverseRootDir");
    const std::filesystem::path derivedDatasetDirectory = configuration.at("datasetSettings").at("compressedRootDir");
    const std::filesystem::path datasetCacheFile = cacheDirectory / ShapeBench::datasetCacheFileName;

    nlohmann::json datasetCacheJson = ShapeBench::computeOrReadDatasetCache(baseDatasetDirectory, derivedDatasetDirectory, datasetCacheFile);
    ShapeBench::Dataset dataset;
    dataset.loadCache(datasetCacheJson);
    return dataset;
}

ShapeDescriptor::cpu::Mesh ShapeBench::Dataset::loadMesh(const ShapeBench::DatasetEntry &entry) {
    return ShapeDescriptor::cpu::Mesh();
}


