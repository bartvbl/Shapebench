#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <mutex>
#include "nlohmann/json.hpp"
#include "benchmarkCore/ComputedConfig.h"
#include "dataset/Dataset.h"
#include "utils/gl/Shader.h"
#include "utils/gl/GeometryBuffer.h"
#include "filters/FilteredMeshPair.h"
#include "filters/Filter.h"
#include "utils/filterUtils/OccludedSceneGenerator.h"
#include "benchmarkCore/randomEngine.h"
#include "meshIntersectionCalculator.h"
#include <random>

namespace ShapeBench {
    template<typename DescriptorMethod>
    class MultiViewOcclusionFilter : public ShapeBench::Filter {
        OccludedSceneGenerator sceneGenerator;
        float supportRadius = 0;

    public:
        // Yes this is sneaking it into the back door, but this is still cleaner than change the entire function signature for all filters
        MultiViewOcclusionFilter(float supportRadius) {
            this->supportRadius = supportRadius;
        }

        void init(const nlohmann::json &config, bool invalidateCaches) {
            uint32_t visibilityImageWidth = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(0);
            uint32_t visibilityImageHeight = config.at("filterSettings").at("subtractiveNoise").at("visibilityImageResolution").at(1);
            sceneGenerator.init(visibilityImageWidth, visibilityImageHeight);
        }

        void destroy() {
            sceneGenerator.destroy();
        }

        void saveCaches(const nlohmann::json &config) {

        }

        FilterOutput applyToBoth(const nlohmann::json &config, ShapeBench::FilteredMeshPair &model,
                                 ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                                 ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) override {
            // Determine transformation settings
            ShapeBench::randomEngine randomEngine(randomSeed);
            OcclusionRendererSettings renderSettings;
            renderSettings.nearPlaneDistance = config.at("filterSettings").at("multiViewOcclusion").at("nearPlaneDistance");
            renderSettings.farPlaneDistance = config.at("filterSettings").at("multiViewOcclusion").at("farPlaneDistance");
            renderSettings.fovy = config.at("filterSettings").at("multiViewOcclusion").at("fovYAngleRadians");
            renderSettings.objectDistanceFromCamera = config.at("filterSettings").at("multiViewOcclusion").at("objectDistanceFromCamera");
            float maxAngleBetweenObjectsDegrees = config.at("filterSettings").at("multiViewOcclusion").at("maxAngleBetweenObjectsDegrees");

            std::uniform_real_distribution<float> distribution(0, 1);
            float angleBetweenObjects = ((distribution(randomEngine) * maxAngleBetweenObjectsDegrees) / 180.0f) * float(M_PI);
            renderSettings.yawDeviation = angleBetweenObjects / 2.0f;
            renderSettings.yaw = float(distribution(randomEngine) * 2.0 * M_PI);
            renderSettings.pitch = float((distribution(randomEngine) - 0.5) * M_PI);
            renderSettings.roll = float(distribution(randomEngine) * 2.0 * M_PI);

            glm::mat4 modelInverseTransform = glm::inverse(model.sampleMeshTransformation);
            glm::mat3 modelInverseNormalTransform = glm::inverse(model.sampleMeshNormalTransformation);
            glm::mat4 sceneInverseTransform = glm::inverse(scene.sampleMeshTransformation);
            glm::mat3 sceneInverseNormalTransform = glm::inverse(scene.sampleMeshNormalTransformation);

            // Undo transformations such that the meshes align. This allows them to be transformed in a corresponding manner
            ShapeBench::transformMesh(model, modelInverseTransform, modelInverseNormalTransform);
            ShapeBench::transformMesh(scene, sceneInverseTransform, sceneInverseNormalTransform);

            // Compute occluded meshes from each perspective
            sceneGenerator.computeOccludedMesh(renderSettings, model);
            renderSettings.yawDeviation *= -1;
            sceneGenerator.computeOccludedMesh(renderSettings, scene);


            // Determine common area on filtered meshes
            ShapeDescriptor::cpu::Mesh commonMesh = ShapeBench::computeCommonTrianglesMesh(model, scene);
            ShapeDescriptor::cpu::Mesh emptyClutterMesh;
            ShapeBench::FilteredMeshPair fakeMeshPair;
            fakeMeshPair.originalMesh = model.originalMesh;
            fakeMeshPair.filteredSampleMesh = commonMesh;
            fakeMeshPair.filteredAdditiveNoise = emptyClutterMesh;

            ShapeBench::FilterOutput output;
            nlohmann::json entry;
            entry["multi-view-occlusion-pitch"] = renderSettings.pitch;
            entry["multi-view-occlusion-yaw"] = renderSettings.yaw;
            entry["multi-view-occlusion-roll"] = renderSettings.roll;
            entry["multi-view-occlusion-angle-between-objects"] = angleBetweenObjects;

            // Computing the area the two viewpoints have in common only needs to be done for one of the two meshes
            for(uint32_t i = 0; i < model.mappedReferenceVertices.size(); i++) {
                if (model.mappedVertexIncluded.at(i)) {
                    ShapeBench::AreaEstimate modelAreaEstimate = ShapeBench::estimateAreaInSupportVolume<DescriptorMethod>(
                            fakeMeshPair, model.originalReferenceVertices.at(i),
                            model.mappedReferenceVertices.at(i),
                            supportRadius, config, randomSeed);
                    entry["multi-view-occlusion-common-area"] = modelAreaEstimate.subtractiveArea;
                }
                output.metadata.push_back(entry);
            }

            // This filter resets the transformations of filtered objects to be able to compute a common area mesh
            // The sample mesh transform should practically be the identity matrix after this
            model.sampleMeshTransformation *= modelInverseTransform;
            for(uint32_t i = 0; i < model.additiveNoiseInfo.size(); i++) {
                model.additiveNoiseInfo.at(i).transformation *= modelInverseTransform;
            }

            scene.sampleMeshTransformation *= sceneInverseTransform;
            for(uint32_t i = 0; i < scene.additiveNoiseInfo.size(); i++) {
                scene.additiveNoiseInfo.at(i).transformation *= sceneInverseTransform;
            }

            return output;
        }

        FilterOutput apply(const nlohmann::json &config, ShapeBench::FilteredMeshPair &scene, const Dataset &dataset,
                           ShapeBench::LocalDatasetCache *fileCache, uint64_t randomSeed) override {
            throw std::runtime_error("This filter can only be applied on both objects at the same time!");
        }

        constexpr bool mustBeAppliedOnBothMeshes() {
            return true;
        }

        const std::string getFilterName() const override {
            return "multi-view-occlusion";
        }

        constexpr bool appliesNonrigidTransformation() override {
            return false;
        }


    };


}
