#pragma once

#include <vector>
#include <iostream>
#include <random>
#include <malloc.h>
#include "dataset/Dataset.h"
#include "json.hpp"
#include "benchmarkCore/Batch.h"
#include "methods/Method.h"
#include "benchmarkCore/common-procedures/descriptorGenerator.h"
#include "benchmarkCore/referenceSetDistanceKernel.cuh"
#include "benchmarkCore/common-procedures/pointCloudSampler.h"
#include "benchmarkCore/common-procedures/meshLoader.h"
#include "benchmarkCore/randomEngine.h"
#include "utils/prettyprint.h"

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
    std::vector<DescriptorType> computeDescriptors(
            uint32_t numberOfSupportRadiiToTry,
            float supportRadiusStart,
            float supportRadiusStep,
            const Dataset& dataset,
            uint64_t referenceDescriptorGenerationSeed,
            uint64_t referencePointCloudSamplingSeed,
            const std::vector<VertexInDataset>& representativeSet,
            const nlohmann::json& config) {

        std::vector<DescriptorType> referenceDescriptors(representativeSet.size() * numberOfSupportRadiiToTry);
        uint32_t representativeSetSize = representativeSet.size();


        std::vector<float> supportRadiiToTry(numberOfSupportRadiiToTry);
        for(uint32_t radiusStep = 0; radiusStep < numberOfSupportRadiiToTry; radiusStep++) {
            supportRadiiToTry.at(radiusStep) = supportRadiusStart + float(radiusStep) * float(supportRadiusStep);
        }

        int referenceCountProcessed = 0;
        #pragma omp parallel for default(none) shared(representativeSetSize, representativeSet, dataset, std::cout, config, referenceCountProcessed, referencePointCloudSamplingSeed, supportRadiiToTry, referenceDescriptorGenerationSeed, numberOfSupportRadiiToTry, referenceDescriptors) schedule(dynamic)
        for(uint32_t referenceIndex = 0; referenceIndex < representativeSetSize; referenceIndex++) {
            std::vector<DescriptorType> generatedDescriptors(supportRadiiToTry.size());
            #pragma omp critical
            {
                referenceCountProcessed++;
                std::cout << "\r        Processing " + std::to_string(referenceCountProcessed) + "/" + std::to_string(representativeSetSize) << " ";
                ShapeBench::drawProgressBar(referenceCountProcessed, representativeSetSize);
                std::cout << std::flush;
            };
            VertexInDataset referenceVertex = representativeSet.at(referenceIndex);
            ShapeDescriptor::cpu::Mesh representativeSetMesh = loadMesh(config, dataset,referenceVertex);
            ShapeDescriptor::cpu::PointCloud representativeSetPointCloud;
            if (DescriptorMethod::usesPointCloudInput()) {
                representativeSetPointCloud = computePointCloud<DescriptorMethod>(representativeSetMesh, config, referencePointCloudSamplingSeed);
            }

            ShapeDescriptor::OrientedPoint originPoint = {representativeSetMesh.vertices[referenceVertex.vertexIndex], representativeSetMesh.normals[referenceVertex.vertexIndex]};
            std::vector<ShapeDescriptor::OrientedPoint> orientedPoints(supportRadiiToTry.size(), originPoint);
            ShapeBench::computeDescriptors<DescriptorMethod, DescriptorType>(
                    representativeSetMesh, representativeSetPointCloud, {orientedPoints.size(), orientedPoints.data()}, config,
                    supportRadiiToTry, referenceDescriptorGenerationSeed, generatedDescriptors);

            for(int i = 0; i < generatedDescriptors.size(); i++) {
                for(float content : generatedDescriptors[i].contents) {
                    if(std::isnan(content)) {
                        throw std::runtime_error("NaN detected");
                    }
                }
            }

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

        if(ShapeBench::hasConfigValue(config, DescriptorMethod::getName(), "normaliseDescriptorWhenComputingSupportRadius")
           && ShapeBench::readDescriptorConfigValue<bool>(config, DescriptorMethod::getName(), "normaliseDescriptorWhenComputingSupportRadius")) {
            std::cout << "    Normalising descriptors.." << std::endl;
            const uint32_t floatsPerDescriptor = (sizeof(DescriptorType) / sizeof(float));

            //#pragma omp parallel for
            for(DescriptorType& descriptor: referenceDescriptors) {
                for(uint32_t i = 0; i < floatsPerDescriptor; i++) {
                    if(std::isnan(descriptor.contents[i])) {
                        throw std::runtime_error("Found a NaN");
                    }
                }
                float maxElement = *std::max_element(descriptor.contents, descriptor.contents + floatsPerDescriptor);
                if(maxElement != 0) {
                    for(uint32_t i = 0; i < floatsPerDescriptor; i++) {
                        descriptor.contents[i] /= maxElement;
                    }
                }
                for(uint32_t i = 0; i < floatsPerDescriptor; i++) {
                    if(std::isnan(maxElement) || std::isnan(descriptor.contents[i])) {
                        throw std::runtime_error("Found a NaN");
                    }
                }
            }
        }

        return referenceDescriptors;
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

        std::vector<VertexInDataset> representativeSet = dataset.sampleVertices(randomEngine(), representativeSetSize, 1);
        std::vector<VertexInDataset> sampleVerticesSet = dataset.sampleVertices(randomEngine(), sampleDescriptorSetSize, 1);

        std::chrono::time_point start = std::chrono::steady_clock::now();


        std::cout << "    Computing reference descriptors.." << std::endl;
        uint64_t referencePointCloudSamplingSeed = randomEngine();
        uint64_t referenceDescriptorGenerationSeed = randomEngine();

        std::vector<DescriptorType> referenceDescriptors = computeDescriptors<DescriptorMethod, DescriptorType>(
                numberOfSupportRadiiToTry,
                supportRadiusStart,
                supportRadiusStep,
                dataset,
                referenceDescriptorGenerationSeed,
                referencePointCloudSamplingSeed,
                representativeSet,
                config);


        std::cout << "    Computing sample descriptors.." << std::endl;
        uint64_t samplePointCloudSamplingSeed = randomEngine();
        uint64_t sampleDescriptorGenerationSeed = randomEngine();

        std::vector<DescriptorType> sampleDescriptors = computeDescriptors<DescriptorMethod, DescriptorType>(
                numberOfSupportRadiiToTry,
                supportRadiusStart,
                supportRadiusStep,
                dataset,
                sampleDescriptorGenerationSeed,
                samplePointCloudSamplingSeed,
                sampleVerticesSet,
                config);



        std::cout << "    Computing distances.." << std::endl;

        std::stringstream outputBuffer;
        outputBuffer << "Radius index, radius, Min mean, Mean, Max mean, Variance min, Mean variance, max variance" << std::endl;
        std::vector<uint32_t> voteHistogram(numberOfSupportRadiiToTry);

        std::vector<DistanceStatistics> distanceStats(numberOfSupportRadiiToTry);
        std::vector<std::string> outputFileContents(numberOfSupportRadiiToTry);
        uint32_t nextToProcess = 0;

        std::cout << "\r        Completed " + std::to_string(nextToProcess) + "/" + std::to_string(numberOfSupportRadiiToTry) << " ";
        ShapeBench::drawProgressBar(nextToProcess + 1, numberOfSupportRadiiToTry);
        std::cout << std::flush;

        bool useGPU = supportRadiusConfig.at("useGPU");

        #pragma omp parallel for schedule(dynamic) default(none) shared(supportRadiusStart, std::cout, nextToProcess, numberOfSupportRadiiToTry, representativeSetSize, referenceDescriptors, sampleDescriptorSetSize, sampleDescriptors, useGPU, distanceStats, supportRadiusStep, outputFileContents)
        for(uint32_t i = 0; i < numberOfSupportRadiiToTry; i++) {
            ShapeDescriptor::cpu::array<DescriptorType> referenceArray = {representativeSetSize, referenceDescriptors.data() + i * representativeSetSize};
            ShapeDescriptor::cpu::array<DescriptorType> sampleArray = {sampleDescriptorSetSize, sampleDescriptors.data() + i * sampleDescriptorSetSize};
            std::vector<DescriptorDistance> supportRadiusDistances = computeReferenceSetDistance<DescriptorMethod, DescriptorType>(sampleArray, referenceArray, useGPU);
            DistanceStatistics stats = computeDistances<DescriptorType>(supportRadiusDistances, sampleDescriptorSetSize);
            distanceStats.at(i) = stats;

            std::stringstream outputLine;
            outputLine << i << ", " << (float(i) * supportRadiusStep + supportRadiusStart) << ", ";
            outputLine << stats.minMeans << ", " << stats.meanOfMeans << ", " << stats.maxMeans << ", "
                       << stats.minVariance << ", " << stats.meanOfVariance << ", " << stats.maxVariance << std::endl;
            outputFileContents.at(i) = outputLine.str();

            #pragma omp critical
            {
                nextToProcess++;
                std::cout << "\r        Completed " + std::to_string(nextToProcess+1) + "/" + std::to_string(numberOfSupportRadiiToTry) << " ";
                ShapeBench::drawProgressBar(nextToProcess, numberOfSupportRadiiToTry);
                std::cout << std::flush;
            }
        }
        for(uint32_t i = 0; i < numberOfSupportRadiiToTry; i++) {
            outputBuffer << outputFileContents.at(i);
        }
        std::string unique = ShapeDescriptor::generateUniqueFilenameString();
        std::filesystem::path outputDirectory = std::filesystem::path(std::string(config.at("resultsDirectory"))) / "support_radius_estimation";
        if(!std::filesystem::exists(outputDirectory)) {
            std::filesystem::create_directories(outputDirectory);
        }
        std::filesystem::path outputPath = outputDirectory / ("support_radii_meanvariance_" + DescriptorMethod::getName() + "_" + unique + ".txt");
        std::ofstream outputFile(outputPath);
        outputFile << outputBuffer.str();

        std::cout << std::endl << outputBuffer.str() << std::endl;

        // Force LibC to clean up
        malloc_trim(0);


        std::chrono::time_point end = std::chrono::steady_clock::now();
        std::cout << std::endl << "    Time taken: ";
        ShapeBench::printDuration(end - start);
        std::cout << std::endl;

        float highestMean = 0;
        float highestMeanSupportRadius = 0;

        for(int i = 0; i < distanceStats.size(); i++) {
            DistanceStatistics stats = distanceStats.at(i);
            if(stats.meanOfMeans > highestMean) {
                highestMean = stats.meanOfMeans;
                highestMeanSupportRadius = float(i) * supportRadiusStep + supportRadiusStart;
            }
        }

        return highestMeanSupportRadius;
    }
}
