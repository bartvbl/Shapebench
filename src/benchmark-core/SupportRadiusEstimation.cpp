#include "SupportRadiusEstimation.h"
#include "RepresentativeSet.h"
#include <methods/Method.h>

template<typename DescriptorMethod, typename DescriptorType>
ShapeDescriptor::gpu::array<DescriptorType> computeRepresentativeDescriptors(const std::vector<VertexInDataset> &representativeSet) {
    return {};
}

template<typename DescriptorMethod, typename DescriptorType, typename distanceFunction>
float Shapebench::estimateSupportRadius(const nlohmann::json &config, const Dataset &dataset, uint64_t randomSeed) {
    static_assert(std::is_base_of<Shapebench::Method<DescriptorType>, DescriptorMethod>::value, "The DescriptorMethod template type parameter must be an object inheriting from Shapebench::Method");

    std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomSeed, config.at("representativeSetObjectCount"));

    std::chrono::time_point start = std::chrono::steady_clock::now();
    Shapebench::computeRepresentativeSet<DescriptorMethod, DescriptorType>(dataset, 0, randomSeed, 0.5);
    std::chrono::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

    return 0;
}