#include "ResultDumper.h"
#include <cuda_runtime.h>
#include <git.h>

nlohmann::json getGPUInfo() {
    cudaDeviceProp device_information;
    int32_t deviceID;
    cudaGetDevice(&deviceID);
    nlohmann::json deviceInfo = {};
    cudaGetDeviceProperties(&device_information, deviceID);
    deviceInfo["gpuName"] = std::string(device_information.name);
    deviceInfo["gpuClock"] = device_information.clockRate;
    deviceInfo["gpuVRAM"] = device_information.totalGlobalMem / (1024 * 1024);
    return deviceInfo;
}

nlohmann::json toJSON(ShapeDescriptor::cpu::float3 in) {
    nlohmann::json out;
    out.push_back(in.x);
    out.push_back(in.y);
    out.push_back(in.z);
    return out;
}

void writeExperimentResults(const ShapeBench::ExperimentResult &results, std::filesystem::path outputBaseDirectory, bool isFinalResult) {

    // 1: Initial version
    // 1.1: Added information about each filter


    nlohmann::json jsonOutput;
    jsonOutput["version"] = "1.1";

    jsonOutput["buildinfo"] = {};
    jsonOutput["buildinfo"]["commit"] = GitMetadata::CommitSHA1();
    jsonOutput["buildinfo"]["commitAuthor"] = GitMetadata::AuthorName();
    jsonOutput["buildinfo"]["commitDate"] = GitMetadata::CommitDate();
    jsonOutput["buildinfo"]["uncommittedChanges"] = GitMetadata::AnyUncommittedChanges();

    jsonOutput["gpuInfo"] = getGPUInfo();

    jsonOutput["experiment"]["index"] = results.experimentIndex;
    jsonOutput["experiment"]["randomSeed"] = results.experimentRandomSeed;

    jsonOutput["method"]["name"] = results.methodName;
    jsonOutput["method"]["metadata"] = results.methodMetadata;

    jsonOutput["configuration"] = results.usedConfiguration;
    jsonOutput["computedConfiguration"] = results.usedComputedConfiguration.toJSON();

    jsonOutput["results"];
    for(uint32_t i = 0; i < results.vertexResults.size(); i++) {
        nlohmann::json entryJson;
        const ShapeBench::ExperimentResultsEntry& entry = results.vertexResults.at(i);

        if(!entry.included) {
            continue;
        }

        entryJson["fractionAddedNoise"] = entry.fractionAddedNoise;
        entryJson["fractionSurfacePartiality"] = entry.fractionSurfacePartiality;
        entryJson["filteredDescriptorRank"] = entry.filteredDescriptorRank;
        entryJson["originalVertex"] = toJSON(entry.originalVertexLocation.vertex);
        entryJson["originalNormal"] = toJSON(entry.originalVertexLocation.normal);
        entryJson["filteredVertex"] = toJSON(entry.filteredVertexLocation.vertex);
        entryJson["filteredNormal"] = toJSON(entry.filteredVertexLocation.normal);
        entryJson["meshID"] = entry.sourceVertex.meshID;
        entryJson["vertexIndex"] = entry.sourceVertex.vertexIndex;
        entryJson["filterOutput"] = entry.filterOutput;

        jsonOutput["results"].push_back(entryJson);
    }

    std::string experimentName = results.usedConfiguration.at("experimentsToRun").at(results.experimentIndex).at("name");
    std::string uniqueString = ShapeDescriptor::generateUniqueFilenameString();
    std::string fileName = experimentName + "-" + results.methodName + "-" + uniqueString + ".json";
    std::filesystem::path outputDirectory = outputBaseDirectory / experimentName;
    if(!isFinalResult) {
        outputDirectory /= "intermediate";
    }
    std::filesystem::create_directories(outputDirectory);
    std::filesystem::path outputFilePath = outputDirectory / fileName;

    std::ofstream outputStream(outputFilePath);
    outputStream << jsonOutput.dump(4) << std::flush;
}
