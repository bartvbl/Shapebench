#pragma once

#include "json.hpp"
#include "results/ExperimentResult.h"
#include <tabulate/table.hpp>

namespace ShapeBench {
    inline void checkReplicatedExperimentResults(const ShapeBench::ExperimentResult& replicatedResults,
                                                 const nlohmann::json& previouslyComputedResults) {
        std::cout << "Validating replicated results.." << std::endl;

        uint32_t validatedResultCount = 0;
        double fractionAddedNoise_deviationSum = 0;
        double fractionAddedNoise_deviationAverage = 0;

        for(uint32_t vertexIndex = 0; vertexIndex < replicatedResults.vertexResults.size(); vertexIndex++) {
            const ExperimentResultsEntry& replicatedResult = replicatedResults.vertexResults.at(vertexIndex);
            const nlohmann::json& originalResult = previouslyComputedResults.at(vertexIndex);

            if(!replicatedResult.included) {
                continue;
            }

            validatedResultCount++;

            double fractionAddedNoise_deviation = double(originalResult["fractionAddedNoise"]) - double(replicatedResult.fractionAddedNoise);
            fractionAddedNoise_deviationSum += fractionAddedNoise_deviation;
            fractionAddedNoise_deviationAverage += (fractionAddedNoise_deviation - fractionAddedNoise_deviationAverage) / double(validatedResultCount);

            /*entryJson["fractionAddedNoise"] = entry.fractionAddedNoise;
            entryJson["fractionSurfacePartiality"] = entry.fractionSurfacePartiality;
            entryJson["filteredDescriptorRank"] = entry.filteredDescriptorRank;
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

        std::cout << "    Validation complete. Results:" << std::endl << std::endl;

        tabulate::Table outputTable;
        outputTable.add_row({"Result", "Total Deviation", "Average Deviation"});
        outputTable.add_row({"Clutter", std::to_string(fractionAddedNoise_deviationSum), std::to_string(fractionAddedNoise_deviationAverage)});
        std::cout << outputTable << std::endl << std::endl;

        std::cout << "    Validated " << validatedResultCount << " results." << std::endl;
    }
}
