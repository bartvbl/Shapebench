#pragma once

#include <vector>
#include <iostream>
#include <random>
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"
#include "methods/Method.h"
#include "referenceDescriptorSet.h"

namespace Shapebench {
    inline std::vector<ShapeDescriptor::cpu::Mesh> loadMeshRange(const nlohmann::json& config, const Dataset& dataset, const std::vector<VertexInDataset>& vertices, uint32_t startIndex, uint32_t endIndex) {
        // Load up to < endIndex, so this is correct
        uint32_t meshCount = endIndex - startIndex;
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

        std::vector<uint32_t> computedRanks(sampleDescriptorSetSize * numberOfSupportRadiiToTry);

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
            // TODO: make this function use cached meshes
            referenceDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(
                    representativeSet, dataset, config, supportRadiiToTry, randomEngine(), referenceStartIndex, referenceEndIndex);

            for(uint32_t sampleStartIndex = 0; sampleStartIndex < sampleDescriptorSetSize; sampleStartIndex += sampleBatchSizeLimit) {
                uint32_t sampleEndIndex = std::min<uint32_t>(sampleStartIndex + sampleBatchSizeLimit, sampleDescriptorSetSize);
                std::cout << "        Computing ranks for sample " << (sampleStartIndex + 1) << "-" << sampleEndIndex << "/" << sampleDescriptorSetSize << " in representative vertex " << (referenceStartIndex + 1) << "-" << referenceEndIndex << "/" << representativeSetSize << std::endl;
                sampleDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(
                        sampleVerticesSet, dataset, config, supportRadiiToTry, randomEngine(), sampleStartIndex, sampleEndIndex);

                for(uint32_t i = 0; i < supportRadiiToTry.size(); i++) {
                    ShapeDescriptor::cpu::array<uint32_t> ranks = DescriptorMethod::computeDescriptorRanks(sampleDescriptors.at(i), referenceDescriptors.at(i));
                    // TODO: copy ranks to output array
                    // TODO: ranks function should output zeroes since descriptors are the same but it does not because it assumes you want to compare to the descriptor at index i.
                    ShapeDescriptor::free(ranks);
                }

                freeDescriptorVector<DescriptorType>(sampleDescriptors);
            }



            freeDescriptorVector<DescriptorType>(referenceDescriptors);
            freeMeshRange(representativeSetMeshes);
        }

        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        return 0;
    }


}