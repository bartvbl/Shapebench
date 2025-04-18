#include "filterSequenceRunner.h"
#include "fmt/format.h"

void ShapeBench::applyFilterSequence(
        ShapeBench::LocalDatasetCache *fileCache,
        const std::string modelFilterSequenceJSONEntryLabel,
        const std::string sceneFilterSequenceJSONEntryLabel,
        const nlohmann::json &configuration,
        const ShapeBench::Dataset &dataset,
        const std::unordered_map<std::string, std::unique_ptr<ShapeBench::Filter>> &filterInstanceMap,
        const uint32_t verticesPerSampleObject,
        const nlohmann::json &experimentConfig,
        std::vector<ShapeBench::ExperimentResultsEntry>& resultsEntries,
        ShapeBench::randomEngine &experimentInstanceRandomEngine,
        ShapeBench::FilteredMeshPair &filteredModelObject,
        ShapeBench::FilteredMeshPair &filteredSceneObject) {

    nlohmann::json emptyObject;
    const nlohmann::json& modelFilterSequence = experimentConfig.contains(modelFilterSequenceJSONEntryLabel)
            ? experimentConfig.at(modelFilterSequenceJSONEntryLabel) : emptyObject;
    const nlohmann::json& sceneFilterSequence = experimentConfig.contains(sceneFilterSequenceJSONEntryLabel)
            ? experimentConfig.at(sceneFilterSequenceJSONEntryLabel) : emptyObject;

    const uint32_t modelFilterCount = modelFilterSequence.size();
    const uint32_t sceneFilterCount = sceneFilterSequence.size();

    // Pregenerate seeds to ensure the order in which filters are applied does not matter for their outcome
    std::vector<uint64_t> modelRandomSeeds(modelFilterCount);
    for(uint32_t i = 0; i < modelFilterCount; i++) {
        modelRandomSeeds.at(i) = experimentInstanceRandomEngine();
    }

    uint32_t nextModelFilterIndex = 0;
    uint32_t nextSceneFilterIndex = 0;

    while(nextModelFilterIndex < modelFilterCount || nextSceneFilterIndex < sceneFilterCount) {
        bool hasModelFiltersLeft = nextModelFilterIndex < modelFilterCount;
        bool hasSceneFiltersLeft = nextSceneFilterIndex < sceneFilterCount;

        // Used to override specific settings
        nlohmann::json copyOfConfiguration;

        // Do a first pass to determine information about the filters
        bool modelFilterRequiresBothMeshes = false;
        bool modelFilterHasOverrideSettings = false;
        std::string nextModelFilterType;
        std::string nextModelFilterName;
        if(hasModelFiltersLeft) {
            const nlohmann::json& modelFilterConfig = modelFilterSequence.at(nextModelFilterIndex);
            modelFilterHasOverrideSettings = modelFilterConfig.contains("overrideFilterSettings");
            if(modelFilterHasOverrideSettings) {
                // Only create a copy when we absolutely have to
                copyOfConfiguration = configuration;
                copyOfConfiguration.at("filterSettings").merge_patch(modelFilterConfig.at("overrideFilterSettings"));
            }
            nextModelFilterType = modelFilterConfig.at("type");
            const std::unique_ptr<Filter>& nextModelFilter = filterInstanceMap.at(nextModelFilterType);
            modelFilterRequiresBothMeshes = nextModelFilter->mustBeAppliedOnBothMeshes();
            nextModelFilterName = nextModelFilter->getFilterName();
        }

        bool sceneFilterRequiresBothMeshes = false;
        bool sceneFilterHasOverrideSettings = false;
        std::string nextSceneFilterType;
        std::string nextSceneFilterName;
        if(hasSceneFiltersLeft) {
            const nlohmann::json& sceneFilterConfig = sceneFilterSequence.at(nextSceneFilterIndex);
            sceneFilterHasOverrideSettings = sceneFilterConfig.contains("overrideFilterSettings");
            if(sceneFilterHasOverrideSettings) {
                // Only create a copy when we absolutely have to
                if(!modelFilterHasOverrideSettings) {
                    copyOfConfiguration = configuration;
                }
                copyOfConfiguration.at("filterSettings").merge_patch(sceneFilterConfig.at("overrideFilterSettings"));
            }
            nextSceneFilterType = sceneFilterConfig.at("type");
            const std::unique_ptr<Filter>& nextSceneFilter = filterInstanceMap.at(nextSceneFilterType);
            sceneFilterRequiresBothMeshes = nextSceneFilter->mustBeAppliedOnBothMeshes();
            nextSceneFilterName = nextSceneFilter->getFilterName();
        }

        bool modelFilterIsBlocked = modelFilterRequiresBothMeshes && !sceneFilterRequiresBothMeshes;
        bool sceneFilterIsBlocked = sceneFilterRequiresBothMeshes && !modelFilterRequiresBothMeshes;

        const nlohmann::json& configToUse = (modelFilterHasOverrideSettings || sceneFilterHasOverrideSettings)
                                            ? copyOfConfiguration : configuration;

        // Sanity checks
        if(modelFilterIsBlocked && !hasSceneFiltersLeft) {
            throw std::runtime_error(fmt::format("Invalid filter sequence detected: model filter at index {} is waiting for a filter that applies to both meshes (\"{}\"), but that filter appears to be missing from the scene filter sequence", nextModelFilterIndex, nextModelFilterName));
        }
        if(sceneFilterIsBlocked && !hasModelFiltersLeft) {
            throw std::runtime_error(fmt::format("Invalid filter sequence detected: scene filter at index {} is waiting for a filter that applies to both meshes (\"{}\"), but that filter appears to be missing from the model filter sequence", nextSceneFilterIndex, nextSceneFilterName));
        }

        if(hasModelFiltersLeft && hasSceneFiltersLeft && modelFilterRequiresBothMeshes && sceneFilterRequiresBothMeshes) {
            if (nextModelFilterName != nextSceneFilterName) {
                throw std::runtime_error(fmt::format(
                        R"(Filter sequence contains two filters that apply to both objects, but these are not applied in the same order. Error occurred at model filter "{}" and scene filter "{}".)",
                        nextModelFilterName, nextSceneFilterName));
            }

            // Run filter with both meshes
            // Scene random seed is not used right now, but this ensures it can be used in the future should there be a need
            [[maybe_unused]] uint64_t sceneRandomSeed = experimentInstanceRandomEngine();
            ShapeBench::FilterOutput output = filterInstanceMap.at(nextModelFilterType)->applyToBoth(
                    configToUse, filteredModelObject, filteredSceneObject, dataset, fileCache,
                    modelRandomSeeds.at(nextModelFilterIndex));
            if(!output.metadata.empty()) {
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntries.at(i).sceneObject.filterOutput.merge_patch(output.metadata.at(i));
                }
            }
            if(!output.metadata_filterAppliedToBoth.empty()) {
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntries.at(i).modelObject.filterOutput.merge_patch(output.metadata_filterAppliedToBoth.at(i));
                }
            }
            nextModelFilterIndex++;
            nextSceneFilterIndex++;
        } else if(hasModelFiltersLeft) {
            ShapeBench::FilterOutput output = filterInstanceMap.at(nextModelFilterType)->apply(
                    configToUse, filteredModelObject, dataset, fileCache, modelRandomSeeds.at(nextModelFilterIndex));
            if(!output.metadata.empty()) {
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntries.at(i).modelObject.filterOutput.merge_patch(output.metadata.at(i));
                }
            }
            nextModelFilterIndex++;
        } else if(hasSceneFiltersLeft) {
            uint64_t sceneRandomSeed = experimentInstanceRandomEngine();
            ShapeBench::FilterOutput output = filterInstanceMap.at(nextSceneFilterType)->apply(
                    configToUse, filteredSceneObject, dataset, fileCache, sceneRandomSeed);
            if(!output.metadata.empty()) {
                for (uint32_t i = 0; i < verticesPerSampleObject; i++) {
                    resultsEntries.at(i).sceneObject.filterOutput.merge_patch(output.metadata.at(i));
                }
            }
            nextSceneFilterIndex++;
        }
    }
}