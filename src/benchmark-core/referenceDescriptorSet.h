#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "DescriptorGenerator.h"
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"
#include "SupportRadiusEstimation.h"
#include "PointCloudSampler.h"

namespace Shapebench {
    inline ShapeDescriptor::cpu::Mesh readDatasetMesh(const nlohmann::json& config, const std::filesystem::path& pathInDataset, float computedBoundingSphereRadius) {
        std::filesystem::path datasetBasePath = config.at("compressedDatasetRootDir");
        std::filesystem::path currentMeshPath = datasetBasePath / pathInDataset;
        currentMeshPath = currentMeshPath.replace_extension(".cm");
        ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(currentMeshPath);

        // Scale mesh down to a unit sphere
        float scaleFactor = 1.0f / float(computedBoundingSphereRadius);
        for(uint32_t i = 0; i < mesh.vertexCount; i++) {
            mesh.vertices[i] = scaleFactor * mesh.vertices[i];
        }
        return mesh;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    std::vector<ShapeDescriptor::cpu::array<DescriptorType>> computeReferenceDescriptors(
            const std::vector<VertexInDataset> &verticesToRender,
            const std::vector<ShapeDescriptor::cpu::Mesh>& meshes,
            const std::vector<ShapeDescriptor::cpu::PointCloud>& pointClouds,
            const nlohmann::json &config,
            uint32_t randomSeed,
            const std::vector<float>& supportRadii,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {
        std::vector<ShapeDescriptor::cpu::array<DescriptorType>> outputDescriptors(supportRadii.size());

        if (verticesToRender.empty()) {
            ShapeDescriptor::cpu::array<DescriptorType> emptyArray = {0, nullptr};
            std::fill(outputDescriptors.begin(), outputDescriptors.end(), emptyArray);
            return outputDescriptors;
        }

        const uint32_t indicesToProcess = std::min<uint32_t>(endIndex, verticesToRender.size()) - startIndex;
        const uint32_t descriptorCountToCompute = supportRadii.size() * indicesToProcess;
        uint64_t computedDescriptorCount = 0;

        for (uint32_t radiusIndex = 0; radiusIndex < supportRadii.size(); radiusIndex++) {
            outputDescriptors.at(radiusIndex) = ShapeDescriptor::cpu::array<DescriptorType>(indicesToProcess);
        }

        endIndex = std::min<uint32_t>(endIndex, verticesToRender.size());
        #pragma omp parallel for schedule(dynamic) default(none) shared(supportRadii, verticesToRender, randomSeed, startIndex, endIndex, meshes, config, std::cout, pointClouds, outputDescriptors, indicesToProcess, descriptorCountToCompute, computedDescriptorCount)
        for (uint32_t i = startIndex; i < endIndex; i++) {
            ShapeDescriptor::cpu::PointCloud cloud;
            uint32_t meshIndex = i - startIndex;
            const ShapeDescriptor::cpu::Mesh &currentMesh = meshes.at(meshIndex);

            if (pointClouds.empty()) {
                cloud = Shapebench::computePointCloud(currentMesh, config, randomSeed);
            } else {
                cloud = pointClouds.at(meshIndex);
            }

            for (uint32_t radiusIndex = 0; radiusIndex < supportRadii.size(); radiusIndex++) {
                uint32_t vertexIndex = verticesToRender.at(i).vertexIndex;
                ShapeDescriptor::OrientedPoint originPoint = {currentMesh.vertices[vertexIndex], currentMesh.normals[vertexIndex]};

                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> convertedOriginArray = {1, &originPoint};

                ShapeDescriptor::cpu::array<DescriptorType> descriptors;
                if (DescriptorMethod::usesPointCloudInput()) {
                    descriptors = DescriptorMethod::computeDescriptors(cloud, convertedOriginArray, config, supportRadii.at(radiusIndex));
                } else {
                    descriptors = DescriptorMethod::computeDescriptors(currentMesh, convertedOriginArray, config, supportRadii.at(radiusIndex));
                }

                outputDescriptors.at(radiusIndex)[meshIndex] = descriptors.content[0];

                ShapeDescriptor::free(descriptors);

                #pragma omp critical
                {
                    computedDescriptorCount++;
                    if (computedDescriptorCount % 100 == 99) {
                        std::cout << "\r        Completed " << (computedDescriptorCount + 1) << "/"
                                  << descriptorCountToCompute << "    " << std::flush;
                    }
                };
            }

            if (DescriptorMethod::usesPointCloudInput() && pointClouds.empty()) {
                ShapeDescriptor::free(cloud);
            }
        }

        /*for (uint32_t radiusIndex = 0; radiusIndex < supportRadii.size(); radiusIndex++) {
            ShapeDescriptor::writeDescriptorImages(outputDescriptors.at(radiusIndex), "out_radiusindex" + std::to_string(radiusIndex) + "_" + ShapeDescriptor::generateUniqueFilenameString() + ".png", 50);
        }*/

        return outputDescriptors;
    }
}