#pragma once

#include "json.hpp"
#include "results/ExperimentResult.h"
#include <tabulate/table.hpp>

namespace ShapeBench {
    template<typename T>
    struct SinglePropertyStatistics {
        uint32_t count = 0;
        uint32_t currentCountIndentical = 0;
        double currentAverage = 0;
        double currentSum = 0;
        double currentMax = -std::numeric_limits<double>::max();

        double max() {
            return currentMax;
        }
        double average() {
            return currentAverage;
        }
        double sum() {
            return currentSum;
        }
        std::string countIdentical() {
            return std::to_string(currentCountIndentical) + " / " + std::to_string(count);
        }

        void registerValue(double deviation) {
            if(deviation == 0) {
                currentCountIndentical++;
            }
            count++;
            currentSum += deviation;
            currentMax = std::max<double>(currentMax, deviation);
            currentAverage += (double(deviation) - currentAverage) / double(count);
        }

        void registerValue(T value1, T value2) {
            double deviation = double(value1) - double(value2);
            registerValue(deviation);
        }
    };

    template<typename DescriptorMethod>
    void checkReplicatedExperimentResults(const nlohmann::json& config,
                                          const std::string& experimentName,
                                          const ShapeBench::ExperimentResult& replicatedResults,
                                          const nlohmann::json& previouslyComputedResults) {
        std::cout << "Validating replicated results.." << std::endl;

        uint32_t validatedResultCount = 0;
        SinglePropertyStatistics<double> clutterStats;
        SinglePropertyStatistics<double> occlusionStats;
        SinglePropertyStatistics<uint32_t> descriptorRankStats;
        SinglePropertyStatistics<double> meshIDCorrectStats;
        SinglePropertyStatistics<double> nearestNeighbourPRCStats;
        SinglePropertyStatistics<double> secondNearestNeighbourPRCStats;

        std::vector<SinglePropertyStatistics<double>> filterProperties;
        std::vector<std::string> filterPropertyNames;

        std::unordered_map<uint32_t, uint32_t> resultIndexConversionMap;
        for(uint32_t originalIndex = 0; originalIndex < previouslyComputedResults.size(); originalIndex++) {
            resultIndexConversionMap.insert({previouslyComputedResults.at(originalIndex).at("resultID"), originalIndex});
        }

        std::filesystem::path reportsDirectory = std::filesystem::path(config.at("resultsDirectory")) / "replication-reports";
        std::filesystem::create_directories(reportsDirectory);
        std::filesystem::path reportFilePath = reportsDirectory / (DescriptorMethod::getName() + "-" + experimentName + "-" + ShapeDescriptor::generateUniqueFilenameString() + ".csv");
        std::ofstream reportFile(reportFilePath);

        reportFile << "Result ID, Clutter,,, Occlusion,,, DDI,,, Model ID,,, PRC: distance to nearest neighbour,,, PRC: distance to second nearest neighbour" << std::endl;
        reportFile << ", author, replicated, identical?, author, replicated, identical?, author, replicated, identical?, author, replicated, identical?, author, replicated, identical?, author, replicated, identical?" << std::endl;

        for(uint32_t vertexIndex = 0; vertexIndex < replicatedResults.vertexResults.size(); vertexIndex++) {
            const ExperimentResultsEntry& replicatedResult = replicatedResults.vertexResults.at(vertexIndex);

            if(!replicatedResult.included) {
                continue;
            }

            if(!resultIndexConversionMap.contains(vertexIndex)) {
                throw std::runtime_error("Replication failed: previously computed results is missing one of the results that was replicated (index " + std::to_string(vertexIndex));
            }
            uint32_t mappedIndex = resultIndexConversionMap.at(vertexIndex);
            const nlohmann::json& originalResult = previouslyComputedResults.at(mappedIndex);



            // Detecting properties that any filters might have added
            uint32_t propertyIndex = 0;
            auto originalResultFilterOutputIterator = originalResult["filterOutput"].begin();
            for(const auto& [key, value] : replicatedResult.filterOutput.items()) {
                if(!value.is_number()) {
                    continue;
                }
                if(validatedResultCount == 0) {
                    filterProperties.emplace_back();
                    filterPropertyNames.push_back("Filter Specific Property \"" + key + "\"");
                }
                filterProperties.at(propertyIndex).registerValue(value, originalResultFilterOutputIterator.value());
                propertyIndex++;
                originalResultFilterOutputIterator++;
            }

            validatedResultCount++;

            reportFile << vertexIndex << ", ";
            clutterStats.registerValue(originalResult["fractionAddedNoise"], replicatedResult.fractionAddedNoise);
            reportFile << originalResult["fractionAddedNoise"] << ", " << replicatedResult.fractionAddedNoise << ", " << (originalResult["fractionAddedNoise"] == replicatedResult.fractionAddedNoise ? "yes" : "no") << ", ";
            occlusionStats.registerValue(originalResult["fractionSurfacePartiality"], replicatedResult.fractionSurfacePartiality);
            reportFile << originalResult["fractionSurfacePartiality"] << ", " << replicatedResult.fractionSurfacePartiality << ", " << (originalResult["fractionSurfacePartiality"] == replicatedResult.fractionSurfacePartiality ? "yes" : "no") << ", ";
            descriptorRankStats.registerValue(originalResult["filteredDescriptorRank"], replicatedResult.filteredDescriptorRank);
            reportFile << originalResult["filteredDescriptorRank"] << ", " << replicatedResult.filteredDescriptorRank << ", " << (originalResult["filteredDescriptorRank"] == replicatedResult.filteredDescriptorRank ? "yes" : "no") << ", ";
            meshIDCorrectStats.registerValue(originalResult["meshID"] == replicatedResult.sourceVertex.meshID ? 0 : 1);
            reportFile << originalResult["meshID"] << ", " << replicatedResult.sourceVertex.meshID << ", " << (originalResult["meshID"] == replicatedResult.sourceVertex.meshID ? "yes" : "no") << ", ";
            nearestNeighbourPRCStats.registerValue(originalResult["PRC"]["distanceToNearestNeighbour"], replicatedResult.prcMetadata.distanceToNearestNeighbour);
            reportFile << originalResult["PRC"]["distanceToNearestNeighbour"] << ", " << replicatedResult.prcMetadata.distanceToNearestNeighbour << ", " << (originalResult["PRC"]["distanceToNearestNeighbour"] == replicatedResult.prcMetadata.distanceToNearestNeighbour ? "yes" : "no") << ", ";
            secondNearestNeighbourPRCStats.registerValue(originalResult["PRC"]["distanceToSecondNearestNeighbour"], replicatedResult.prcMetadata.distanceToSecondNearestNeighbour);
            reportFile << originalResult["PRC"]["distanceToSecondNearestNeighbour"] << ", " << replicatedResult.prcMetadata.distanceToSecondNearestNeighbour << ", " << (originalResult["PRC"]["distanceToSecondNearestNeighbour"] == replicatedResult.prcMetadata.distanceToSecondNearestNeighbour ? "yes" : "no") << ", ";
            reportFile << std::endl;
            /*
             * entryJson["fractionAddedNoise"] = entry.fractionAddedNoise;
             * entryJson["fractionSurfacePartiality"] = entry.fractionSurfacePartiality;
             * entryJson["filteredDescriptorRank"] = entry.filteredDescriptorRank;
            entryJson["originalVertex"] = toJSON(entry.originalVertexLocation.vertex);
            entryJson["originalNormal"] = toJSON(entry.originalVertexLocation.normal);
            entryJson["filteredVertex"] = toJSON(entry.filteredVertexLocation.vertex);
            entryJson["filteredNormal"] = toJSON(entry.filteredVertexLocation.normal);
            entryJson["meshID"] = entry.sourceVertex.meshID;
            entryJson["vertexIndex"] = entry.sourceVertex.vertexIndex;
            entryJson["filterOutput"] = entry.filterOutput;
            if(isPRCEnabled) {
                entryJson["PRC"]["distanceToNearestNeighbour"] = entry.prcMetadata.distanceToNearestNeighbour;
                entryJson["PRC"]["distanceToSecondNearestNeighbour"] = entry.prcMetadata.distanceToSecondNearestNeighbour;
                entryJson["PRC"]["modelPointMeshID"] = entry.prcMetadata.modelPointMeshID;
                entryJson["PRC"]["scenePointMeshID"] = entry.prcMetadata.scenePointMeshID;
                entryJson["PRC"]["nearestNeighbourVertexModel"] = toJSON(entry.prcMetadata.nearestNeighbourVertexModel);
                entryJson["PRC"]["nearestNeighbourVertexScene"] = toJSON(entry.prcMetadata.nearestNeighbourVertexScene);
            }*/


        }

        std::cout << "    Table: Overview over the extent to which the replicated and original values deviate from each other." << std::endl;
        std::cout << "           Total deviation is the sum of differences between all replicated values." << std::endl << std::endl;


        tabulate::Table outputTable;
        outputTable.add_row({"Result", "Total Deviation", "Average Deviation", "Maximum Deviation", "Number of Identical Results"});

        outputTable.add_row(tabulate::RowStream{} << "Clutter" << clutterStats.sum() << clutterStats.average() << clutterStats.max() << clutterStats.countIdentical());
        outputTable.add_row(tabulate::RowStream{} << "Occlusion" << occlusionStats.sum() << occlusionStats.average() << occlusionStats.max() << occlusionStats.countIdentical());
        outputTable.add_row(tabulate::RowStream{} << "Descriptor Distance Index" << descriptorRankStats.sum() << descriptorRankStats.average() << descriptorRankStats.max() << descriptorRankStats.countIdentical());
        outputTable.add_row(tabulate::RowStream{} << "Distance to Nearest Neighbour (PRC)" << nearestNeighbourPRCStats.sum() << nearestNeighbourPRCStats.average() << nearestNeighbourPRCStats.max() << nearestNeighbourPRCStats.countIdentical());
        outputTable.add_row(tabulate::RowStream{} << "Distance to Second Nearest Neighbour (PRC)" << secondNearestNeighbourPRCStats.sum() << secondNearestNeighbourPRCStats.average() << secondNearestNeighbourPRCStats.max() << secondNearestNeighbourPRCStats.countIdentical());
        for(uint32_t i = 0; i < filterProperties.size(); i++) {
            outputTable.add_row(tabulate::RowStream{} << filterPropertyNames.at(i) << filterProperties.at(i).sum() << filterProperties.at(i).average() << filterProperties.at(i).max() << filterProperties.at(i).countIdentical());
        }


        std::cout << outputTable << std::endl << std::endl;

        std::cout << "    Replication complete." << std::endl;
        std::cout << "    A CSV file containing a comparison between all individual results has been written to:" << std::endl;
        std::cout << "        " << reportFilePath.string() << std::endl;
    }
}
