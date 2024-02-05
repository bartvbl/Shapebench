#pragma once

template<typename DescriptorMethod, typename DescriptorType>
void runClutterExperiment(const nlohmann::json& config, const ComputedConfig& computedConfig, const Dataset& dataset, uint64_t randomSeed) {
    ClutteredScene scene = createClutteredScene(config, dataset, randomSeed);
}


