#pragma once

#include "shapeDescriptor/shapeDescriptor.h"
#include "DescriptorGenerator.h"
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"

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
            const nlohmann::json &config,
            const std::vector<float>& supportRadii,
            uint64_t randomSeed,
            uint32_t startIndex = 0,
            uint32_t endIndex = 0xFFFFFFFF) {


        std::vector<ShapeDescriptor::cpu::array<DescriptorType>> outputDescriptors(supportRadii.size());
        if(verticesToRender.empty()) {
            ShapeDescriptor::cpu::array<DescriptorType> emptyArray = {0, nullptr};
            std::fill(outputDescriptors.begin(), outputDescriptors.end(), emptyArray);
            return outputDescriptors;
        }

        const uint32_t indicesToProcess = std::min<uint32_t>(endIndex, verticesToRender.size()) - startIndex;
        const uint32_t descriptorCountToCompute = supportRadii.size() * indicesToProcess;
        uint64_t computedDescriptorCount = 0;

        #pragma omp parallel for schedule(dynamic) default(none) shared(supportRadii, verticesToRender, startIndex, endIndex, meshes, config, std::cout, randomSeed, outputDescriptors, indicesToProcess, descriptorCountToCompute, computedDescriptorCount)
        for(uint32_t radiusIndex = 0; radiusIndex < supportRadii.size(); radiusIndex++) {

            ShapeDescriptor::cpu::array<DescriptorType> radiusDescriptors(indicesToProcess);

            uint32_t currentMeshIndex = verticesToRender.at(startIndex).meshID;
            std::vector<ShapeDescriptor::OrientedPoint> vertexOrigins;
            std::vector<uint32_t> vertexIndices;

            endIndex = std::min<uint32_t>(endIndex, verticesToRender.size());
            for(uint32_t i = startIndex; i <= endIndex; i++) {
                // We have moved on to a new mesh. Load the new one. Also includes a case for the final iteration
                if(i == endIndex || currentMeshIndex != verticesToRender.at(i).meshID) {
                    // Should not trigger out of bounds exception as the first iteration should not trigger the above if statement
                    const ShapeDescriptor::cpu::Mesh& currentMesh = meshes.at(i - startIndex - 1);
                    //ShapeDescriptor::gpu::Mesh currentMeshGPU = ShapeDescriptor::copyToGPU(currentMesh);

                    for(uint32_t index = 0; index < vertexIndices.size(); index++) {
                        uint32_t vertexIndex = vertexIndices.at(index);
                        vertexOrigins.push_back({currentMesh.vertices[vertexIndex], currentMesh.normals[vertexIndex]});
                    }

                    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> convertedOriginArray = {vertexOrigins.size(), vertexOrigins.data()};
                    //ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> gpuOrigins = ShapeDescriptor::copyToGPU(convertedOriginArray);

                    ShapeDescriptor::cpu::array<DescriptorType> descriptors;
                    if(DescriptorMethod::usesPointCloudInput()) {
                        double totalMeshArea = ShapeDescriptor::calculateMeshSurfaceArea(currentMesh);
                        double sampleDensity = config.at("pointSampleDensity");
                        uint32_t sampleCount = uint32_t(totalMeshArea * sampleDensity);
                        std::cout << "Sampling with point density: " << sampleCount << std::endl;
                        ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::sampleMesh(currentMesh, sampleCount, randomSeed);
                        descriptors = DescriptorMethod::computeDescriptors(cloud, convertedOriginArray, config, supportRadii.at(radiusIndex));
                        ShapeDescriptor::free(cloud);
                    } else {
                        descriptors = DescriptorMethod::computeDescriptors(currentMesh, convertedOriginArray, config, supportRadii.at(radiusIndex));
                    }

                    uint32_t targetStartIndex = i - vertexIndices.size();
                    //cudaMemcpy(radiusDescriptors.content + targetStartIndex, descriptors.content, descriptors.length * sizeof(DescriptorType), cudaMemcpyDeviceToDevice);
                    std::copy(descriptors.content, descriptors.content + descriptors.length, radiusDescriptors.content + startIndex);

                    ShapeDescriptor::free(descriptors);
                    //ShapeDescriptor::free(gpuOrigins);
                    //ShapeDescriptor::free(currentMeshGPU);

                    vertexIndices.clear();
                    vertexOrigins.clear();

                    #pragma omp critical
                    {
                        computedDescriptorCount++;
                        if(computedDescriptorCount % 100 == 0) {
                            std::cout << "\r        Completed " << computedDescriptorCount << "/" << descriptorCountToCompute << "    " << std::flush;
                        }
                    };
                }

                if(i < endIndex) {
                    currentMeshIndex = verticesToRender.at(i).meshID;
                    vertexIndices.push_back(verticesToRender.at(i).vertexIndex);
                }
            }
            outputDescriptors.at(radiusIndex) = radiusDescriptors;
        }

        return outputDescriptors;
    }
}