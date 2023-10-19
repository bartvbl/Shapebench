#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "DescriptorGenerator.h"
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"

namespace Shapebench {
    template<typename DescriptorMethod>
    ShapeDescriptor::cpu::Mesh readDatasetMesh(const nlohmann::json& config, const std::filesystem::path& pathInDataset, float computedBoundingSphereRadius) {
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
    ShapeDescriptor::gpu::array<DescriptorType> computeReferenceDescriptors(
            const std::vector<VertexInDataset> &verticesToRender,
            const Dataset& dataset,
            const nlohmann::json &config,
            float supportRadius,
            uint64_t randomSeed,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {


        if(verticesToRender.empty()) {
            ShapeDescriptor::gpu::array<DescriptorType> emptyArray = {0, nullptr};
            return emptyArray;
        }

        uint32_t indicesToProcess = std::min<uint32_t>(endIndex, verticesToRender.size()) - startIndex;
        ShapeDescriptor::gpu::array<DescriptorType> outputDescriptors(indicesToProcess);

        uint32_t currentMeshIndex = verticesToRender.at(0).meshID;
        std::vector<ShapeDescriptor::OrientedPoint> vertexOrigins;
        std::vector<uint32_t> vertexIndices;

        endIndex = std::min<uint32_t>(endIndex, verticesToRender.size());
        for(uint32_t i = startIndex; i <= endIndex; i++) {
            // We have moved on to a new mesh. Load the new one. Also includes a case for the final iteration
            if(i == endIndex || currentMeshIndex != verticesToRender.at(i).meshID) {
                const DatasetEntry& currentDatasetEntry = dataset.at(currentMeshIndex);
                ShapeDescriptor::cpu::Mesh currentMesh = readDatasetMesh<DescriptorMethod>(config, currentDatasetEntry.meshFile, currentDatasetEntry.computedObjectRadius);
                ShapeDescriptor::gpu::Mesh currentMeshGPU = ShapeDescriptor::copyToGPU(currentMesh);

                for(uint32_t index = 0; index < vertexIndices.size(); index++) {
                    uint32_t vertexIndex = vertexIndices.at(index);
                    vertexOrigins.push_back({currentMesh.vertices[vertexIndex], currentMesh.normals[vertexIndex]});
                }

                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> convertedOriginArray = {vertexOrigins.size(), vertexOrigins.data()};
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins = ShapeDescriptor::copyToGPU(convertedOriginArray);

                ShapeDescriptor::gpu::array<DescriptorType> descriptors;
                if(DescriptorMethod::usesPointCloudInput()) {
                    double totalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(currentMesh);
                    double sampleDensity = config.at("pointSampleDensity");
                    uint32_t sampleCount = uint32_t(totalMeshArea * sampleDensity);
                    std::cout << "Sampling with point density: " << sampleCount << std::endl;
                    ShapeDescriptor::gpu::PointCloud cloud = ShapeDescriptor::sampleMesh(currentMeshGPU, sampleCount, randomSeed);
                    descriptors = DescriptorMethod::computeDescriptors(cloud, gpuOrigins, config, supportRadius);
                    ShapeDescriptor::free(cloud);
                } else {
                    descriptors = DescriptorMethod::computeDescriptors(currentMeshGPU, gpuOrigins, config, supportRadius);
                }

                uint32_t targetStartIndex = i - vertexIndices.size();
                cudaMemcpy(outputDescriptors.content + targetStartIndex, descriptors.content, descriptors.length * sizeof(DescriptorType), cudaMemcpyDeviceToDevice);

                ShapeDescriptor::free(descriptors);
                ShapeDescriptor::free(gpuOrigins);
                ShapeDescriptor::free(currentMesh);
                ShapeDescriptor::free(currentMeshGPU);

                vertexIndices.clear();
                vertexOrigins.clear();
            }

            if(i < endIndex) {
                currentMeshIndex = verticesToRender.at(i).meshID;
                vertexIndices.push_back(verticesToRender.at(i).vertexIndex);
            }
        }

        return outputDescriptors;
    }
}