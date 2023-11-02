#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <malloc.h>
#include "Dataset.h"
#include "json.hpp"
#include "Batch.h"
#include "methods/Method.h"
#include "referenceDescriptorSet.h"
#include "referenceSetDistanceKernel.cuh"
#include "PointCloudSampler.h"

namespace Shapebench {
    inline std::vector<ShapeDescriptor::cpu::Mesh> loadMeshRange(const nlohmann::json& config, const Dataset& dataset, const std::vector<VertexInDataset>& vertices, uint32_t startIndex, uint32_t endIndex) {
        // Load up to < endIndex, so this is correct
        uint32_t meshCount = endIndex - startIndex;

        // We load the mesh for each vertex. This assumes that most of these will be unique anyway
        // That is, there is only one vertex sampled per mesh.
        // If this assumption changes later we'll have to create a second vector containing an index buffer
        // which mesh in a condensed vector to use
        std::vector<ShapeDescriptor::cpu::Mesh> meshes(meshCount);
        #pragma omp parallel for schedule(dynamic) default(none) shared(meshCount, dataset, startIndex, meshes, config, vertices)
        for(uint32_t i = 0; i < meshCount; i++) {
            const DatasetEntry& entry = dataset.at(vertices.at(startIndex + i).meshID);
            meshes.at(i) = readDatasetMesh(config, entry.meshFile, entry.computedObjectRadius);
        }

        return meshes;
    }

    inline void freeMeshRange(std::vector<ShapeDescriptor::cpu::Mesh>& meshes) {
        for(ShapeDescriptor::cpu::Mesh& mesh : meshes) {
            ShapeDescriptor::free(mesh);
        }
    }

    inline void freePointCloudRange(std::vector<ShapeDescriptor::cpu::PointCloud>& clouds) {
        for(ShapeDescriptor::cpu::PointCloud& cloud : clouds) {
            ShapeDescriptor::free(cloud);
        }
    }

    template<typename DescriptorType>
    void freeDescriptorVector(std::vector<ShapeDescriptor::gpu::array<DescriptorType>>& descriptorList) {
        for(ShapeDescriptor::gpu::array<DescriptorType>& descriptorArray : descriptorList) {
            ShapeDescriptor::free(descriptorArray);
        }
    }

    template<typename DescriptorType>
    void freeDescriptorVector(std::vector<ShapeDescriptor::cpu::array<DescriptorType>>& descriptorList) {
        for(ShapeDescriptor::cpu::array<DescriptorType>& descriptorArray : descriptorList) {
            ShapeDescriptor::free(descriptorArray);
        }
    }

    void printDistancesTable(const std::vector<DescriptorDistance> &distances,
                             uint32_t numberOfSampleDescriptors,
                             float supportRadiusStart,
                             float supportRadiusStep,
                             uint32_t supportRadiusCount) {
        std::stringstream outputBuffer;
        outputBuffer << "Radius index, radius, Min mean, Mean, Max mean, Variance min, Mean variance, max variance" << std::endl;
        std::vector<uint32_t> voteHistogram(supportRadiusCount);
        for(uint32_t radius = 0; radius < supportRadiusCount; radius++) {
            outputBuffer << radius << ", "
                         << (float(radius) * supportRadiusStep + supportRadiusStart) << ", ";
            float meanOfMeans = 0;
            float meanOfVariance = 0;
            uint32_t distancesStartIndex = radius * numberOfSampleDescriptors;
            float minMeans = distances.at(distancesStartIndex).mean;
            float maxMeans = distances.at(distancesStartIndex).mean;
            float minVariance = distances.at(distancesStartIndex).variance;
            float maxVariance = distances.at(distancesStartIndex).variance;
            for(uint32_t i = 0; i < numberOfSampleDescriptors; i++) {
                meanOfMeans += (distances.at(distancesStartIndex + i).mean - meanOfMeans) / float(i + 1);
                meanOfVariance += (distances.at(distancesStartIndex + i).variance - meanOfVariance) / float(i + 1);
                minMeans = std::min(minMeans, distances.at(distancesStartIndex + i).mean);
                maxMeans = std::max(maxMeans, distances.at(distancesStartIndex + i).mean);
                minVariance = std::min(minVariance, distances.at(distancesStartIndex + i).variance);
                maxVariance = std::max(maxVariance, distances.at(distancesStartIndex + i).variance);
            }
            outputBuffer << minMeans << ", " << meanOfMeans << ", " << maxMeans << ", " << minVariance << ", " << meanOfVariance << ", " << maxVariance << std::endl;
        }

        for(uint32_t i = 0; i < numberOfSampleDescriptors; i++) {
            uint32_t bestRadiusIndex = 0;
            float bestRadiusDistance = distances.at(0 * numberOfSampleDescriptors + i).mean;
            for(uint32_t radius = 0; radius < supportRadiusCount; radius++) {
                uint32_t distanceIndex = radius * numberOfSampleDescriptors + i;
                float currentDistance = distances.at(distanceIndex).mean;
                if(currentDistance > bestRadiusDistance) {
                    bestRadiusIndex = radius;
                    bestRadiusDistance = currentDistance;
                }
            }
            voteHistogram.at(bestRadiusIndex)++;
        }
        std::stringstream histogramBuffer;
        histogramBuffer << "Radius Index, Radius, Vote count" << std::endl;
        for(uint32_t radius = 0; radius < supportRadiusCount; radius++) {
            histogramBuffer << radius << ", " << (float(radius) * supportRadiusStep + supportRadiusStart) << ", " << voteHistogram.at(radius) << std::endl;
        }

        std::string unique = ShapeDescriptor::generateUniqueFilenameString();
        std::ofstream outputFile("support_radii_meanvariance_" + unique + ".txt");
        outputFile << outputBuffer.str();
        std::ofstream histogramFile("support_radii_votes_" + unique + ".txt");
        histogramFile << histogramBuffer.str();

        std::cout << outputBuffer.str() << std::endl << std::endl << histogramBuffer.str() << std::endl;
    }

    template<typename DescriptorMethod, typename DescriptorType>
    float estimateSupportRadius(const nlohmann::json& config, const Dataset& dataset, uint64_t randomSeed) {
        static_assert(std::is_base_of<Shapebench::Method<DescriptorType>, DescriptorMethod>::value, "The DescriptorMethod template type parameter must be an object inheriting from Shapebench::Method");

        uint32_t representativeSetSize = config.at("representativeSetObjectCount");
        std::mt19937_64 randomEngine(randomSeed);

        const nlohmann::json& supportRadiusConfig = config.at("parameterSelection").at("supportRadius");
        uint32_t sampleDescriptorSetSize = supportRadiusConfig.at("sampleDescriptorSetSize");
        float supportRadiusStart = supportRadiusConfig.at("radiusSearchStart");
        float supportRadiusStep = supportRadiusConfig.at("radiusSearchStep");
        uint32_t numberOfSupportRadiiToTry = supportRadiusConfig.at("numberOfSupportRadiiToTry");

        uint32_t referenceBatchSizeLimit =
                config.contains("limits") && config.at("limits").contains("representativeSetBatchSizeLimit")
                ? uint32_t(config.at("limits").at("representativeSetBatchSizeLimit"))
                : representativeSetSize;
        uint32_t sampleBatchSizeLimit =
                config.contains("limits") && config.at("limits").contains("sampleSetBatchSizeLimit")
                ? uint32_t(config.at("limits").at("sampleSetBatchSizeLimit"))
                : sampleDescriptorSetSize;
        std::cout << "    Batch sizes: representative -> " << referenceBatchSizeLimit << ", sample -> " << sampleBatchSizeLimit << std::endl;

        std::vector<ShapeDescriptor::cpu::array<DescriptorType>> sampleDescriptors;
        std::vector<ShapeDescriptor::cpu::array<DescriptorType>> referenceDescriptors;

        std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomEngine(), representativeSetSize);
        std::vector<VertexInDataset> sampleVerticesSet = dataset.sampleVertices(randomEngine(), sampleDescriptorSetSize);

        std::vector<DescriptorDistance> descriptorDistances(sampleDescriptorSetSize * numberOfSupportRadiiToTry);

        std::vector<float> supportRadiiToTry(numberOfSupportRadiiToTry);
        for(uint32_t radiusStep = 0; radiusStep < numberOfSupportRadiiToTry; radiusStep++) {
            supportRadiiToTry.at(radiusStep) = supportRadiusStart + float(radiusStep) * float(supportRadiusStep);
        }

        std::chrono::time_point start = std::chrono::steady_clock::now();

        for(uint32_t referenceStartIndex = 0; referenceStartIndex < representativeSetSize; referenceStartIndex += referenceBatchSizeLimit) {
            uint32_t referenceEndIndex = std::min<uint32_t>(referenceStartIndex + referenceBatchSizeLimit, representativeSetSize);
            std::cout << "    Processing reference batch " << (referenceStartIndex + 1) << "-" << referenceEndIndex << "/" << representativeSetSize << std::endl;
            std::cout << "    Loading meshes.." << std::endl;
            std::vector<ShapeDescriptor::cpu::Mesh> representativeSetMeshes = loadMeshRange(config, dataset,representativeSet,referenceStartIndex, referenceEndIndex);
            std::vector<ShapeDescriptor::cpu::PointCloud> representativeSetPointClouds;
            if(DescriptorMethod::usesPointCloudInput()) {
                std::cout << "    Sampling point clouds.." << std::endl;
                computePointClouds(representativeSetMeshes, representativeSetPointClouds, config, randomEngine());
            }

            std::cout << "    Computing reference descriptors.." << std::endl;
            referenceDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(
                    representativeSet, representativeSetMeshes, representativeSetPointClouds, config, randomSeed, supportRadiiToTry, referenceStartIndex, referenceEndIndex);

            for(uint32_t sampleStartIndex = 0; sampleStartIndex < sampleDescriptorSetSize; sampleStartIndex += sampleBatchSizeLimit) {
                uint32_t sampleEndIndex = std::min<uint32_t>(sampleStartIndex + sampleBatchSizeLimit, sampleDescriptorSetSize);
                std::cout << "    Computing ranks for sample " << (sampleStartIndex + 1) << "-" << sampleEndIndex << "/" << sampleDescriptorSetSize << " in representative vertex " << (referenceStartIndex + 1) << "-" << referenceEndIndex << "/" << representativeSetSize << std::endl;
                std::cout << "    Loading meshes.." << std::endl;
                std::vector<ShapeDescriptor::cpu::Mesh> sampleSetMeshes = loadMeshRange(config, dataset,sampleVerticesSet,sampleStartIndex, sampleEndIndex);
                std::vector<ShapeDescriptor::cpu::PointCloud> sampleSetPointClouds;
                if(DescriptorMethod::usesPointCloudInput()) {
                    std::cout << "    Sampling point clouds.." << std::endl;
                    //computePointClouds(sampleSetMeshes, sampleSetPointClouds, config, randomEngine());
                }
                std::cout << "    Computing sample descriptors.." << std::endl;
                sampleDescriptors = Shapebench::computeReferenceDescriptors<DescriptorMethod, DescriptorType>(
                        sampleVerticesSet, sampleSetMeshes, sampleSetPointClouds, config, randomSeed,supportRadiiToTry, sampleStartIndex, sampleEndIndex);

                std::cout << "    Computing distances.." << std::endl;
                for(uint32_t i = 0; i < supportRadiiToTry.size(); i++) {
                    // TODO: run this on GPU
                    ShapeDescriptor::cpu::array<DescriptorDistance> distances = computeReferenceSetDistance<DescriptorMethod, DescriptorType>(sampleDescriptors.at(i), referenceDescriptors.at(i));

                    uint32_t distancesStartIndex = i * sampleDescriptorSetSize + sampleStartIndex;
                    std::copy(distances.content, distances.content + distances.length, descriptorDistances.data() + distancesStartIndex);

                    ShapeDescriptor::free(distances);
                }
                freeDescriptorVector<DescriptorType>(sampleDescriptors);
                freePointCloudRange(sampleSetPointClouds);
                freeMeshRange(sampleSetMeshes);

                // Force LibC to clean up
                malloc_trim(0);
            }

            freeDescriptorVector<DescriptorType>(referenceDescriptors);
            freeMeshRange(representativeSetMeshes);
            freePointCloudRange(representativeSetPointClouds);

            // Force LibC to clean up
            malloc_trim(0);

            printDistancesTable(descriptorDistances,
                                sampleDescriptorSetSize,
                                supportRadiusStart,
                                supportRadiusStep,
                                numberOfSupportRadiiToTry);
        }

        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        // TODO: use descriptorDistances to determine a support radius


        return 0;
    }


}