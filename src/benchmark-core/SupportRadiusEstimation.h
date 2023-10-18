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
    uint64_t sum(ShapeDescriptor::cpu::array<uint32_t> ranks) {
        uint64_t rankSum = 0;
        for(uint32_t i = 0; i < ranks.length; i++) {
            rankSum += ranks[i];
        }
        return rankSum;
    }


    template<typename DescriptorMethod, typename DescriptorType>
    float estimateSupportRadius(const nlohmann::json& config, const Dataset& dataset, uint64_t randomSeed) {
        static_assert(std::is_base_of<Shapebench::Method<DescriptorType>, DescriptorMethod>::value, "The DescriptorMethod template type parameter must be an object inheriting from Shapebench::Method");

        uint32_t representativeSetSize = config.at("representativeSetObjectCount");
        std::mt19937_64 randomEngine(randomSeed);

        std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomEngine(), representativeSetSize);

        uint32_t batchSizeLimit =
                config.contains("limits") && config.at("limits").contains("representativeSetBatchSizeLimit")
                ? uint32_t(config.at("limits").at("representativeSetBatchSizeLimit"))
                : representativeSet.size();

        ShapeDescriptor::gpu::array<DescriptorType> sampleDescriptors;
        ShapeDescriptor::gpu::array<DescriptorType> referenceDescriptors;

        const nlohmann::json& supportRadiusConfig = config.at("parameterSelection").at("supportRadius");
        uint32_t sampleDescriptorSetSize = supportRadiusConfig.at("sampleDescriptorSetSize");
        float supportRadiusStart = supportRadiusConfig.at("radiusDeviationStep");
        float supportRadiusStep = supportRadiusConfig.at("radiusSearchStep");
        uint32_t numberOfSupportRadiiToTry = supportRadiusConfig.at("numberOfSupportRadiiToTry");

        std::vector<VertexInDataset> sampleVerticesSet = dataset.sampleVertices(randomEngine(), sampleDescriptorSetSize);

        std::vector<uint64_t> sumsOfRanks(numberOfSupportRadiiToTry);

        std::chrono::time_point start = std::chrono::steady_clock::now();

        for(uint32_t radiusStep = 0; radiusStep < numberOfSupportRadiiToTry; radiusStep++) {
            float supportRadius = supportRadiusStart + supportRadiusStep * float(supportRadiusStep);

            for(uint32_t index = 0; index < representativeSetSize; index += batchSizeLimit) {
                std::cout << "\r    Testing support radius " << supportRadius << "/" << (float(numberOfSupportRadiiToTry) * float(radiusStep) + supportRadiusStart) << " - vertex " << index << "/" << representativeSetSize << std::flush;
                uint32_t startIndex = index;
                uint32_t endIndex = std::min<uint32_t>(index + batchSizeLimit, representativeSetSize);

                sampleDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(sampleVerticesSet, config, supportRadius);

                referenceDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, config, supportRadius, startIndex, endIndex);

                ShapeDescriptor::cpu::array<uint32_t> ranks = DescriptorMethod::computeDescriptorRanks(sampleDescriptors, referenceDescriptors);
                sumsOfRanks.at(radiusStep) += sum(ranks);

                ShapeDescriptor::free(ranks);
                ShapeDescriptor::free(referenceDescriptors);
                ShapeDescriptor::free(sampleDescriptors);
            }
        }

        std::vector<double> averageRanks(numberOfSupportRadiiToTry);
        for(int i = 0; i < averageRanks.size(); i++) {
            averageRanks.at(i) = double(sumsOfRanks.at(i)) / double(numberOfSupportRadiiToTry);
            float supportRadius = supportRadiusStart + i * float(supportRadiusStep);
            std::cout << i << ", " << supportRadius << ", " << averageRanks.at(i);
        }

        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        return 0;
    }


}