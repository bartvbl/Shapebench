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
    double computeAverageRank(ShapeDescriptor::cpu::array<uint32_t> ranks, uint32_t representativeSetSize) {
        uint64_t rankSum = 0;
        for(uint32_t i = 0; i < ranks.length; i++) {
            rankSum += ranks[i];
        }
        return double(rankSum) / double(representativeSetSize);
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

        std::array<ShapeDescriptor::gpu::array<DescriptorType>, 3> sampleDescriptors;
        ShapeDescriptor::gpu::array<DescriptorType> referenceDescriptors;

        std::vector<double> averageDistancesToReferenceSet;
        std::vector<double> averageDeviatedDistancesLow;
        std::vector<double> averageDeviatedDistancesHigh;

        const nlohmann::json& supportRadiusConfig = config.at("parameterSelection").at("supportRadius");
        uint32_t sampleDescriptorSetSize = supportRadiusConfig.at("sampleDescriptorSetSize");
        float supportRadiusStart = supportRadiusConfig.at("radiusDeviationStep");
        float supportRadiusStep = supportRadiusConfig.at("radiusSearchStep");
        float radiusSensitivityDeviation = supportRadiusConfig.at("radiusSensitivityDeviation");
        uint32_t numberOfSupportRadiiToTry = supportRadiusConfig.at("numberOfSupportRadiiToTry");

        std::vector<VertexInDataset> sampleVerticesSet = dataset.sampleVertices(randomEngine(), sampleDescriptorSetSize);

        averageDistancesToReferenceSet.resize(numberOfSupportRadiiToTry);
        averageDeviatedDistancesLow.resize(numberOfSupportRadiiToTry);
        averageDeviatedDistancesHigh.resize(numberOfSupportRadiiToTry);

        std::chrono::time_point start = std::chrono::steady_clock::now();

        for(uint32_t radiusStep = 0; radiusStep < numberOfSupportRadiiToTry; radiusStep++) {
            float supportRadius = supportRadiusStart + supportRadiusStep * float(supportRadiusStep);
            float deviatedSupportRadiusLow = (1 - radiusSensitivityDeviation) * supportRadius;
            float deviatedSupportRadiusHigh = (1 + radiusSensitivityDeviation) * supportRadius;

            for(uint32_t index = 0; index < representativeSetSize; index += batchSizeLimit) {
                std::cout << "\r    Testing support radius " << supportRadius << "/" << (float(numberOfSupportRadiiToTry) * float(radiusStep) + supportRadiusStart) << " - vertex " << index << "/" << representativeSetSize << std::flush;
                uint32_t startIndex = index;
                uint32_t endIndex = std::min<uint32_t>(index + batchSizeLimit, representativeSetSize);

                std::array<float, 3> sampleSupportRadii = {deviatedSupportRadiusLow, supportRadius, deviatedSupportRadiusHigh};
                sampleDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType, 3>(sampleVerticesSet, config, sampleSupportRadii);

                referenceDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(representativeSet, config, supportRadius, startIndex, endIndex);

                ShapeDescriptor::cpu::array<uint32_t> ranksDeviatedLow = DescriptorMethod::computeDescriptorRanks(sampleDescriptors.at(0), referenceDescriptors);
                averageDeviatedDistancesLow.at(index) += computeAverageRank(ranksDeviatedLow, representativeSetSize);

                ShapeDescriptor::cpu::array<uint32_t> ranks = DescriptorMethod::computeDescriptorRanks(sampleDescriptors.at(1), referenceDescriptors);
                averageDistancesToReferenceSet.at(index) += computeAverageRank(ranksDeviatedLow, representativeSetSize);

                ShapeDescriptor::cpu::array<uint32_t> ranksDeviatedHigh = DescriptorMethod::computeDescriptorRanks(sampleDescriptors.at(2), referenceDescriptors);
                averageDeviatedDistancesHigh.at(index) += computeAverageRank(ranksDeviatedLow, representativeSetSize);

                ShapeDescriptor::free(ranksDeviatedLow);
                ShapeDescriptor::free(ranks);
                ShapeDescriptor::free(ranksDeviatedHigh);
                ShapeDescriptor::free(referenceDescriptors);
                ShapeDescriptor::free(sampleDescriptors.at(0));
                ShapeDescriptor::free(sampleDescriptors.at(1));
                ShapeDescriptor::free(sampleDescriptors.at(2));
            }
        }

        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        return 0;
    }


}