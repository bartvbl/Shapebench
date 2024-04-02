#include "glad/gl.h"
#include "OcclusionFilter.h"
#include "utils/gl/GLUtils.h"
#include "utils/gl/VAOGenerator.h"
#include "utils/gl/Shader.h"
#include "utils/gl/ShaderLoader.h"
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include "GLFW/glfw3.h"
#include "benchmarkCore/randomEngine.h"
#include <random>


void ShapeBench::OcclusionFilter::init(const nlohmann::json &config) {
    uint32_t visibilityImageWidth = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(0);
    uint32_t visibilityImageHeight = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(1);
    sceneGenerator.init(visibilityImageWidth, visibilityImageHeight);
}

void ShapeBench::OcclusionFilter::destroy() {
    sceneGenerator.destroy();
}

void ShapeBench::OcclusionFilter::saveCaches(const nlohmann::json &config) {

}

ShapeBench::FilterOutput
ShapeBench::OcclusionFilter::apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene,
                                   const ShapeBench::Dataset &dataset, uint64_t randomSeed) {

    OcclusionRendererSettings renderSettings;
    renderSettings.nearPlaneDistance = config.at("filterSettings").at("subtractiveNoise").at("nearPlaneDistance");
    renderSettings.farPlaneDistance = config.at("filterSettings").at("subtractiveNoise").at("farPlaneDistance");
    renderSettings.fovy = config.at("filterSettings").at("subtractiveNoise").at("fovYAngleRadians");
    renderSettings.objectDistanceFromCamera = config.at("filterSettings").at("subtractiveNoise").at("objectDistanceFromCamera");

    return sceneGenerator.computeOccludedMesh(renderSettings, scene, randomSeed);
}
