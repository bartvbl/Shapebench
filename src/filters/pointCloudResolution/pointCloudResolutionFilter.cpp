#include "pointCloudResolutionFilter.h"
#include "benchmarkCore/randomEngine.h"


ShapeBench::FilterOutput
ShapeBench::PointCloudResolutionFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                              const Dataset &dataset,
                                              ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {


    float minDeviation = config.at("filterSettings").at("pointCloudResolution").at("minScaleFactor");
    float maxDeviation = config.at("filterSettings").at("pointCloudResolution").at("maxScaleFactor");

    ShapeBench::randomEngine engine(randomSeed);
    std::uniform_real_distribution<float> pointCountDeviationDistribution(minDeviation, maxDeviation);

    float chosenPointCountFactor = pointCountDeviationDistribution(engine);

    scene.pointCloudConversionScaleFactor *= chosenPointCountFactor;

    ShapeBench::FilterOutput meta;
    nlohmann::json metadataEntry;
    metadataEntry["point-cloud-resolution-scale-factor"] = chosenPointCountFactor;
    for(uint32_t i = 0; i < scene.mappedReferenceVertices.size(); i++) {
        meta.metadata.push_back(metadataEntry);
    }

    return meta;
}

void ShapeBench::PointCloudResolutionFilter::init(const nlohmann::json &config, bool invalidateCaches) {

}

void ShapeBench::PointCloudResolutionFilter::destroy() {

}

void ShapeBench::PointCloudResolutionFilter::saveCaches(const nlohmann::json& config) {

}


ShapeBench::FilterOutput
ShapeBench::PointCloudResolutionFilter::applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                                    ShapeBench::FilteredMeshPair &scene,
                                                    const ShapeBench::Dataset &dataset,
                                                    ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) {
    throw std::runtime_error("This filter is not meant to be applied on both meshes!");
    return {};
}

const std::string ShapeBench::PointCloudResolutionFilter::getFilterName() const {
    return "alternate-point-cloud-resolution";
}

