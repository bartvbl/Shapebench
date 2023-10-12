#pragma once

#include <vector>
#include <iostream>
#include "Dataset.h"
#include "json.hpp"
#include "methods/Method.h"
#include "Batch.h"

namespace Shapebench {
    template<typename DescriptorMethod, typename DescriptorType>
    float estimateSupportRadius(const nlohmann::json& config, const Dataset& dataset, uint64_t randomSeed) {
        static_assert(std::is_base_of<Shapebench::Method<DescriptorType>, DescriptorMethod>::value, "The DescriptorMethod template type parameter must be an object inheriting from Shapebench::Method");

        uint32_t representativeSetSize = config.at("representativeSetObjectCount");
        std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomSeed, representativeSetSize);

        uint32_t batchSizeLimit =
                config.contains("limits") && config.at("limits").contains("representativeSetBatchSizeLimit")
                ? uint32_t(config.at("limits").at("representativeSetBatchSizeLimit")) : std::numeric_limits<uint32_t>::max();


        Batch<uint32_t> batch(representativeSetSize, batchSizeLimit);
        for()


        std::chrono::time_point start = std::chrono::steady_clock::now();
        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        return 0;
    }


}