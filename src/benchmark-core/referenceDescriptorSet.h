#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"

namespace Shapebench {
    template<typename DescriptorMethod>
    ShapeDescriptor::cpu::Mesh readDatasetMesh(const nlohmann::json& config, const std::filesystem::path& pathInDataset) {
        std::filesystem::path datasetBasePath = config.at("compressedDatasetRootDir");
        std::filesystem::path currentMeshPath = datasetBasePath / pathInDataset;
        currentMeshPath = currentMeshPath.replace_extension(".cm");
        return ShapeDescriptor::loadMesh(currentMeshPath);
    }

    template<typename DescriptorMethod, typename DescriptorType, int supportRadiusCount>
    std::array<ShapeDescriptor::gpu::array<DescriptorType>, supportRadiusCount> computeReferenceDescriptors(
            const std::vector<VertexInDataset> &verticesToRender,
            const nlohmann::json &config,
            std::array<float, supportRadiusCount> supportRadius,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {
        std::array<ShapeDescriptor::gpu::array<DescriptorType>, supportRadiusCount> outputDescriptors;
        if(verticesToRender.empty()) {
            ShapeDescriptor::gpu::array<DescriptorType> emptyArray = {0, nullptr};
            std::fill(outputDescriptors.begin(), outputDescriptors.end(), emptyArray);
            return outputDescriptors;
        }

        std::filesystem::path currentMeshPath = verticesToRender.at(0).meshFile;
        ShapeDescriptor::cpu::Mesh currentMesh = readDatasetMesh<DescriptorMethod>(config, verticesToRender.at(startIndex).meshFile);
        std::vector<ShapeDescriptor::OrientedPoint> vertexOrigins;

        endIndex = std::min<uint32_t>(endIndex, verticesToRender.size());
        for(uint32_t i = startIndex; i < endIndex; i++) {
            // We have moved on to a new mesh. Load the new one
            if(verticesToRender.at(startIndex).meshFile != currentMeshPath) {
                ShapeDescriptor::cpu::array<

                ShapeDescriptor::free(currentMesh);
                currentMeshPath = verticesToRender.at(startIndex).meshFile;
                currentMesh = readDatasetMesh<DescriptorMethod>(config, currentMeshPath);
            }
            vertexOrigins.push_back(verticesToRender.at(i).vertexIndex);
        }
    }

    template<typename DescriptorMethod, typename DescriptorType>
    ShapeDescriptor::gpu::array<DescriptorType> computeReferenceDescriptors(
            const std::vector<VertexInDataset> &verticesToRender,
            const nlohmann::json &config,
            float supportRadius,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {
        return computeReferenceDescriptors<DescriptorMethod, DescriptorType, 1>(verticesToRender, config, {supportRadius}, startIndex, endIndex).at(0);
    }
}