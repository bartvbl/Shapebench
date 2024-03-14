#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <malloc.h>
#include "benchmarkCore/Dataset.h"
#include "json.hpp"
#include "benchmarkCore/Batch.h"
#include "methods/Method.h"
#include "benchmarkCore/common-procedures/descriptorGenerator.h"
#include "benchmarkCore/referenceSetDistanceKernel.cuh"
#include "benchmarkCore/common-procedures/pointCloudSampler.h"
#include "benchmarkCore/common-procedures/meshLoader.h"
#include "benchmarkCore/randomEngine.h"

namespace ShapeBench {
    inline ShapeDescriptor::cpu::Mesh loadMesh(const nlohmann::json& config, const ShapeBench::Dataset& dataset, ShapeBench::VertexInDataset vertex) {
        // We load the mesh for each vertex. This assumes that most of these will be unique anyway
        // That is, there is only one vertex sampled per mesh.
        // If this assumption changes later we'll have to create a second vector containing an index buffer
        // which mesh in a condensed vector to use
        const ShapeBench::DatasetEntry& entry = dataset.at(vertex.meshID);
        return readDatasetMesh(config, entry);
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

    struct DistanceStatistics {
        float meanOfMeans = 0;
        float meanOfVariance = 0;
        float minMeans = 0;
        float maxMeans = 0;
        float minVariance = 0;
        float maxVariance = 0;
    };

    template<typename DescriptorType>
    DistanceStatistics computeDistances(std::vector<ShapeBench::DescriptorDistance>& distances, uint32_t sampleDescriptorCount) {
        DistanceStatistics stats;
        stats.meanOfMeans = 0;
        stats.meanOfVariance = 0;
        stats.minMeans = distances.at(0).mean;
        stats.maxMeans = distances.at(0).mean;
        stats.minVariance = distances.at(0).variance;
        stats.maxVariance = distances.at(0).variance;
        for(uint32_t i = 0; i < sampleDescriptorCount; i++) {
            stats.meanOfMeans += (distances.at(i).mean - stats.meanOfMeans) / float(i + 1);
            stats.meanOfVariance += (distances.at(i).variance - stats.meanOfVariance) / float(i + 1);
            stats.minMeans = std::min(stats.minMeans, distances.at(i).mean);
            stats.maxMeans = std::max(stats.maxMeans, distances.at(i).mean);
            stats.minVariance = std::min(stats.minVariance, distances.at(i).variance);
            stats.maxVariance = std::max(stats.maxVariance, distances.at(i).variance);
        }
        return stats;
    }

    void printDistancesTable(const std::vector<ShapeBench::DescriptorDistance> &distances,
                             uint32_t numberOfSampleDescriptors,
                             float supportRadiusStart,
                             float supportRadiusStep,
                             uint32_t supportRadiusCount) {

     /*   for(uint32_t radius = 0; radius < supportRadiusCount; radius++) {

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

        std::cout << outputBuffer.str() << std::endl << std::endl << histogramBuffer.str() << std::endl;*/
    }

    template<typename DescriptorMethod, typename DescriptorType>
    float estimateSupportRadius(const nlohmann::json& config, const Dataset& dataset, uint64_t randomSeed) {
        static_assert(std::is_base_of<ShapeBench::Method<DescriptorType>, DescriptorMethod>::value, "The DescriptorMethod template type parameter must be an object inheriting from Shapebench::Method");


        ShapeBench::randomEngine randomEngine(randomSeed);

        const nlohmann::json& supportRadiusConfig = config.at("parameterSelection").at("supportRadius");
        uint32_t representativeSetSize = supportRadiusConfig.at("representativeSetObjectCount");
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

        std::vector<DescriptorType> sampleDescriptors(representativeSetSize * numberOfSupportRadiiToTry);
        std::vector<DescriptorType> referenceDescriptors(sampleDescriptorSetSize * numberOfSupportRadiiToTry);

        std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomEngine(), representativeSetSize, 1);
        std::vector<VertexInDataset> sampleVerticesSet = dataset.sampleVertices(randomEngine(), sampleDescriptorSetSize, 1);

        std::vector<float> supportRadiiToTry(numberOfSupportRadiiToTry);
        for(uint32_t radiusStep = 0; radiusStep < numberOfSupportRadiiToTry; radiusStep++) {
            supportRadiiToTry.at(radiusStep) = supportRadiusStart + float(radiusStep) * float(supportRadiusStep);
        }

        std::chrono::time_point start = std::chrono::steady_clock::now();


        std::cout << "    Computing reference descriptors.." << std::endl;
        uint64_t referencePointCloudSamplingSeed = randomEngine();
        uint64_t referenceDescriptorGenerationSeed = randomEngine();
        std::vector<DescriptorType> generatedDescriptors(supportRadiiToTry.size());
        int referenceCountProcessed = 0;
        #pragma omp parallel for default(none) shared(representativeSetSize, representativeSet, dataset, std::cout, config, referenceCountProcessed, referencePointCloudSamplingSeed, supportRadiiToTry, referenceDescriptorGenerationSeed, numberOfSupportRadiiToTry, referenceDescriptors, sampleDescriptorSetSize, sampleVerticesSet) firstprivate(generatedDescriptors) schedule(dynamic)
        for(uint32_t referenceIndex = 0; referenceIndex < representativeSetSize; referenceIndex++) {
            #pragma omp atomic
            referenceCountProcessed++;
            std::cout << "\r        Processing " + std::to_string(referenceCountProcessed) + "/" + std::to_string(representativeSetSize) << std::flush;
            VertexInDataset referenceVertex = representativeSet.at(referenceIndex);
            ShapeDescriptor::cpu::Mesh representativeSetMesh = loadMesh(config, dataset,referenceVertex);
            ShapeDescriptor::cpu::PointCloud representativeSetPointCloud;
            if (DescriptorMethod::usesPointCloudInput()) {
                representativeSetPointCloud = computePointCloud(representativeSetMesh, config, referencePointCloudSamplingSeed);
            }

            ShapeBench::computeDescriptorsForEachSupportRadii<DescriptorMethod, DescriptorType>(
                    referenceVertex, representativeSetMesh, representativeSetPointCloud, config, referenceDescriptorGenerationSeed,
                    supportRadiiToTry, generatedDescriptors);
            for(uint32_t i = 0; i < numberOfSupportRadiiToTry; i++) {
                referenceDescriptors.at(representativeSetSize * i + referenceIndex) = generatedDescriptors.at(i);
            }

            if(DescriptorMethod::usesPointCloudInput()) {
                ShapeDescriptor::free(representativeSetPointCloud);
            }
            ShapeDescriptor::free(representativeSetMesh);
        }
        std::cout << std::endl;

        // Force LibC to clean up
        malloc_trim(0);

        std::cout << "    Computing sample descriptors.." << std::endl;
        uint64_t samplePointCloudSamplingSeed = randomEngine();
        uint64_t sampleDescriptorGenerationSeed = randomEngine();
        int sampleCountProcessed = 0;
        #pragma omp parallel for default(none) shared(sampleDescriptorSetSize, sampleVerticesSet, sampleCountProcessed, std::cout, dataset, config, samplePointCloudSamplingSeed, sampleDescriptorGenerationSeed, supportRadiiToTry, numberOfSupportRadiiToTry, sampleDescriptors) firstprivate(generatedDescriptors) schedule(dynamic)
        for(uint32_t sampleIndex = 0; sampleIndex < sampleDescriptorSetSize; sampleIndex++) {
            #pragma omp atomic
            sampleCountProcessed++;
            std::cout << "\r        Processing " + std::to_string(sampleCountProcessed) + "/" + std::to_string(sampleDescriptorSetSize) << std::flush;
            VertexInDataset sampleVertex = sampleVerticesSet.at(sampleIndex);
            ShapeDescriptor::cpu::Mesh sampleSetMesh = loadMesh(config, dataset, sampleVertex);
            ShapeDescriptor::cpu::PointCloud sampleSetPointCloud;
            if (DescriptorMethod::usesPointCloudInput()) {
                sampleSetPointCloud = computePointCloud(sampleSetMesh, config, samplePointCloudSamplingSeed);
            }

            ShapeBench::computeDescriptorsForEachSupportRadii<DescriptorMethod, DescriptorType>(
                    sampleVertex, sampleSetMesh, sampleSetPointCloud, config, sampleDescriptorGenerationSeed,
                    supportRadiiToTry, generatedDescriptors);

            for(uint32_t i = 0; i < numberOfSupportRadiiToTry; i++) {
                sampleDescriptors.at(sampleDescriptorSetSize * i + sampleIndex) = generatedDescriptors.at(i);
            }

            if(DescriptorMethod::usesPointCloudInput()) {
                ShapeDescriptor::free(sampleSetPointCloud);
            }
            ShapeDescriptor::free(sampleSetMesh);
        }
        std::cout << std::endl;

        // Force LibC to clean up
        malloc_trim(0);

        std::cout << "    Computing distances.." << std::endl;

        std::stringstream outputBuffer;
        outputBuffer << "Radius index, radius, Min mean, Mean, Max mean, Variance min, Mean variance, max variance" << std::endl;
        std::vector<uint32_t> voteHistogram(numberOfSupportRadiiToTry);

        std::vector<DistanceStatistics> distanceStats(supportRadiiToTry.size());
        for(uint32_t i = 0; i < supportRadiiToTry.size(); i++) {
            std::cout << "\r        Processing " << (i+1) << "/" << supportRadiiToTry.size() << std::flush;
            ShapeDescriptor::cpu::array<DescriptorType> referenceArray = {representativeSetSize, referenceDescriptors.data() + i * representativeSetSize};
            ShapeDescriptor::cpu::array<DescriptorType> sampleArray = {sampleDescriptorSetSize, sampleDescriptors.data() + i * sampleDescriptorSetSize};
            std::vector<DescriptorDistance> supportRadiusDistances = computeReferenceSetDistance<DescriptorMethod, DescriptorType>(sampleArray, referenceArray);
            DistanceStatistics stats = computeDistances<DescriptorType>(supportRadiusDistances, sampleDescriptorSetSize);
            distanceStats.at(i) = stats;

            outputBuffer << i << ", " << (float(i) * supportRadiusStep + supportRadiusStart) << ", ";
            outputBuffer << stats.minMeans << ", " << stats.meanOfMeans << ", " << stats.maxMeans << ", "
                         << stats.minVariance << ", " << stats.meanOfVariance << ", " << stats.maxVariance << std::endl;
        }
        std::string unique = ShapeDescriptor::generateUniqueFilenameString();
        std::ofstream outputFile("support_radii_meanvariance_" + DescriptorMethod::getName() + "_" + unique + ".txt");
        outputFile << outputBuffer.str();

        std::cout << std::endl << outputBuffer.str() << std::endl;

        // Force LibC to clean up
        malloc_trim(0);


        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "    Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << std::endl;

        float highestMean = 0;
        float highestMeanSupportRadius = 0;

        for(int i = 0; i < distanceStats.size(); i++) {
            DistanceStatistics stats = distanceStats.at(i);
            if(stats.meanOfMeans > highestMean) {
                highestMean = stats.meanOfMeans;
                highestMeanSupportRadius = supportRadiiToTry.at(i);
            }
        }

        return highestMeanSupportRadius;
    }
}