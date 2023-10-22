#pragma once

#include <vector>
#include <iostream>
#include <random>
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"
#include "methods/Method.h"
#include "referenceDescriptorSet.h"
#include "referenceSetDistanceKernel.cuh"

namespace Shapebench {
    inline std::vector<ShapeDescriptor::cpu::Mesh> loadMeshRange(const nlohmann::json& config, const Dataset& dataset, const std::vector<VertexInDataset>& vertices, uint32_t startIndex, uint32_t endIndex) {
        // Load up to < endIndex, so this is correct
        uint32_t meshCount = endIndex - startIndex;

        // We load the mesh for each vertex. This assumes that most of these will be unique anyway
        // That is, there is only one vertex sampled per mesh.
        // If this assumption changes later we'll have to create a second vector containing an index buffer
        // which mesh in a condensed vector to use
        std::vector<ShapeDescriptor::cpu::Mesh> meshes(meshCount);
        #pragma omp parallel for default(none) shared(meshCount, dataset, startIndex, meshes, config, vertices)
        for(uint32_t i = 0; i < meshCount; i++) {
            const DatasetEntry& entry = dataset.at(vertices.at(startIndex + i).meshID);
            meshes.at(i) = readDatasetMesh(config, entry.meshFile, entry.computedObjectRadius);
        }

        return meshes;
    }

    inline void freeMeshRange(std::vector<ShapeDescriptor::cpu::Mesh>& meshes) {
        for(ShapeDescriptor::cpu::Mesh& mesh : meshes) {
            ShapeDescriptor::free(mesh);
        }
    }

    template<typename DescriptorType>
    void freeDescriptorVector(std::vector<ShapeDescriptor::gpu::array<DescriptorType>>& descriptorList) {
        for(ShapeDescriptor::gpu::array<DescriptorType>& descriptorArray : descriptorList) {
            ShapeDescriptor::free(descriptorArray);
        }
    }

    void printDistancesTable(const std::vector<DescriptorDistance> &distances,
                             uint32_t numberOfSampleDescriptors,
                             float supportRadiusStart,
                             float supportRadiusStep,
                             uint32_t supportRadiusCount) {
        std::stringstream outputBuffer;
        for(uint32_t radius = 0; radius < supportRadiusCount; radius++) {
            outputBuffer << radius << ", "
                         << (float(radius) * supportRadiusStep + supportRadiusStart) << ", ";
            float meanOfMeans = 0;
            for(uint32_t i = 0; i < numberOfSampleDescriptors; i++) {
                meanOfMeans += (distances.at(i).mean - meanOfMeans) / float(i + 1);
            }
            outputBuffer << meanOfMeans << std::endl;
        }

        std::ofstream outputFile("support_radii.txt");
        outputFile << outputBuffer.str();

        std::cout << outputBuffer.str() << std::endl;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    float estimateSupportRadius(const nlohmann::json& config, const Dataset& dataset, uint64_t randomSeed) {
        static_assert(std::is_base_of<Shapebench::Method<DescriptorType>, DescriptorMethod>::value, "The DescriptorMethod template type parameter must be an object inheriting from Shapebench::Method");

        uint32_t representativeSetSize = config.at("representativeSetObjectCount");
        std::mt19937_64 randomEngine(randomSeed);

        const nlohmann::json& supportRadiusConfig = config.at("parameterSelection").at("supportRadius");
        uint32_t sampleDescriptorSetSize = supportRadiusConfig.at("sampleDescriptorSetSize");
        float supportRadiusStart = supportRadiusConfig.at("radiusDeviationStep");
        float supportRadiusStep = supportRadiusConfig.at("radiusSearchStep");
        uint32_t numberOfSupportRadiiToTry = supportRadiusConfig.at("numberOfSupportRadiiToTry");

        uint32_t referenceBatchSizeLimit =
                config.contains("limits") && config.at("limits").contains("representativeSetBatchSizeLimit")
                ? uint32_t(config.at("limits").at("representativeSetBatchSizeLimit"))
                : representativeSetSize;
        uint32_t sampleBatchSizeLimit =
                config.contains("limits") && config.at("limits").contains("sampleBatchSizeLimit")
                ? uint32_t(config.at("limits").at("sampleBatchSizeLimit"))
                : sampleDescriptorSetSize;

        std::vector<ShapeDescriptor::gpu::array<DescriptorType>> sampleDescriptors;
        std::vector<ShapeDescriptor::gpu::array<DescriptorType>> referenceDescriptors;

        std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomEngine(), representativeSetSize);
        std::vector<VertexInDataset> sampleVerticesSet = dataset.sampleVertices(randomEngine(), sampleDescriptorSetSize);

        std::vector<DescriptorDistance> descriptorDistances(sampleDescriptorSetSize * numberOfSupportRadiiToTry);

        std::vector<float> supportRadiiToTry(numberOfSupportRadiiToTry);
        for(uint32_t radiusStep = 0; radiusStep < numberOfSupportRadiiToTry; radiusStep++) {
            supportRadiiToTry.at(radiusStep) = supportRadiusStart + float(radiusStep) * float(supportRadiusStep);
        }

        std::chrono::time_point start = std::chrono::steady_clock::now();

        for(uint32_t referenceStartIndex = 0; referenceStartIndex < representativeSetSize; referenceStartIndex += referenceBatchSizeLimit) {
            uint32_t referenceEndIndex = std::min<uint32_t>(referenceStartIndex + referenceBatchSizeLimit, representativeSetSize);
            std::cout << "    Processing reference batch " << (referenceStartIndex + 1) << "-" << referenceEndIndex << "/" << representativeSetSize << std::endl;
            std::cout << "    Loading meshes.." << std::endl;
            std::vector<ShapeDescriptor::cpu::Mesh> representativeSetMeshes = loadMeshRange(config, dataset,representativeSet,referenceStartIndex, referenceEndIndex);

            std::cout << "    Computing reference descriptors.." << std::endl;
            referenceDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(
                    representativeSet, representativeSetMeshes, config, supportRadiiToTry, randomEngine(), referenceStartIndex, referenceEndIndex);

            for(uint32_t sampleStartIndex = 0; sampleStartIndex < sampleDescriptorSetSize; sampleStartIndex += sampleBatchSizeLimit) {
                uint32_t sampleEndIndex = std::min<uint32_t>(sampleStartIndex + sampleBatchSizeLimit, sampleDescriptorSetSize);
                std::cout << "        Computing ranks for sample " << (sampleStartIndex + 1) << "-" << sampleEndIndex << "/" << sampleDescriptorSetSize << " in representative vertex " << (referenceStartIndex + 1) << "-" << referenceEndIndex << "/" << representativeSetSize << std::endl;
                std::vector<ShapeDescriptor::cpu::Mesh> sampleSetMeshes = loadMeshRange(config, dataset,sampleVerticesSet,sampleStartIndex, sampleEndIndex);
                sampleDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(
                        sampleVerticesSet, sampleSetMeshes, config, supportRadiiToTry, randomEngine(), sampleStartIndex, sampleEndIndex);

                for(uint32_t i = 0; i < supportRadiiToTry.size(); i++) {
                    ShapeDescriptor::cpu::array<DescriptorDistance> distances = computeReferenceSetDistance<DescriptorMethod, DescriptorType>(sampleDescriptors.at(i), referenceDescriptors.at(i));

                    uint32_t distancesStartIndex = i * sampleDescriptorSetSize + sampleStartIndex;
                    std::copy(distances.content, distances.content + distances.length, descriptorDistances.data() + distancesStartIndex);

                    ShapeDescriptor::free(distances);
                }
                freeDescriptorVector<DescriptorType>(sampleDescriptors);
                freeMeshRange(sampleSetMeshes);
            }

            freeDescriptorVector<DescriptorType>(referenceDescriptors);
            freeMeshRange(representativeSetMeshes);

            printDistancesTable(descriptorDistances,
                                sampleDescriptorSetSize,
                                supportRadiusStart,
                                supportRadiusStep,
                                numberOfSupportRadiiToTry);
        }

        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        return 0;
    }


}