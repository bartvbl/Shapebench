#include "arrrgh.hpp"
#include <shapeDescriptor/shapeDescriptor.h>
#include "benchmarkCore/MissingBenchmarkConfigurationException.h"
#include "benchmarkCore/constants.h"
#include "benchmarkCore/CompressedDatasetCreator.h"
#include "benchmarkCore/Dataset.h"
#include "methods/QUICCIMethod.h"
#include "methods/SIMethod.h"
#include "benchmarkCore/ComputedConfig.h"
#include "filters/additiveNoise/additiveNoiseFilter.h"
#include "benchmarkCore/experimentRunner.h"
#include "methods/3DSCMethod.h"
#include "methods/RoPSMethod.h"
#include "methods/RICIMethod.h"
#include "methods/USCMethod.h"


int main(int argc, const char** argv) {
    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& configurationFile = parser.add<std::string>(
            "configuration-file", "Location of the file from which to read the experimental configuration", '\0', arrrgh::Optional, "../cfg/config.json");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Use the GPU with the given ID (as shown in nvidia-smi)", '\0', arrrgh::Optional, 0);

    try
    {
        parser.parse(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        parser.show_usage(std::cerr);
        exit(1);
    }

    // Show help if desired
    if(showHelp.value())
    {
        return 0;
    }

    if(!ShapeDescriptor::isCUDASupportAvailable()) {
        throw std::runtime_error("This benchmark requires CUDA support to operate.");
    }

    //ShapeDescriptor::createCUDAContext(forceGPU.value());

    // ---------------------------------------------------------

    // Special case meshes used as sanity checks
    //ShapeDescriptor::cpu::Mesh inMesh = ShapeDescriptor::loadMesh("/mnt/DATASETS/objaverse/hf-objaverse-v1/glbs/000-031/006eccb258c94370bdfd26205491d135.glb");
    //ShapeDescriptor::writeOBJ(inMesh, "smallMesh.obj");
    //inMesh = ShapeDescriptor::loadMesh("/mnt/DATASETS/objaverse/hf-objaverse-v1/glbs/000-000/0554eaf578284d2b8fa3493a1f1d56c6.glb");
    //ShapeDescriptor::writeOBJ(inMesh, "alarm.obj");

    /*ShapeDescriptor::cpu::Mesh inMesh = ShapeDescriptor::loadMesh("/home/bart/projects/shapebench/current-datasets/raw/Armadillo_vres2_small_scaled_0.ply", ShapeDescriptor::RecomputeNormals::ALWAYS_RECOMPUTE);
    //ShapeDescriptor::cpu::Mesh inMesh = ShapeDescriptor::loadMesh("/home/bart/projects/shapebench/current-datasets/raw/cheff_0.ply", ShapeDescriptor::RecomputeNormals::ALWAYS_RECOMPUTE);
    std::cout << "Mesh has " << inMesh.vertexCount << " vertices" << std::endl;
    //ShapeDescriptor::cpu::PointCloud cloud = ShapeDescriptor::sampleMesh(inMesh, 10000000, 5453464);

    ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> points = ShapeDescriptor::generateUniqueSpinOriginBuffer(inMesh);
    float supportRadius_armadillo = 0.05;
    //float supprotRadius_chef = 25.0;
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptors = ShapeDescriptor::generateRadialIntersectionCountImages(ShapeDescriptor::copyToGPU(inMesh), points.copyToGPU(), supportRadius_armadillo).copyToCPU();
    std::cout << "Writing image.." << std::endl;
    ShapeDescriptor::writeDescriptorImages(descriptors, "armadillo.png", false, 50, 1000);
    //ShapeDescriptor::writeDescriptorImages(descriptors, "chef.png", false, 50, 1000);
    ShapeDescriptor::cpu::array<ShapeDescriptor::RICIDescriptor> descriptorZeroDuplicates(descriptors.length);
    ShapeDescriptor::cpu::float3 referenceVertex = {-0.05154, -0.05256, -0.00618};
    //ShapeDescriptor::cpu::float3 referenceVertex = {-99.67633, -3.19730, -588.92334};
    ShapeDescriptor::cpu::float3 closestPoint;
    int bestIndex = 0;
    float bestDistance = 10;
    for(int i = 0; i < inMesh.vertexCount; i++) {
        if(length(referenceVertex - inMesh.vertices[i]) < bestDistance) {
            bestIndex = i;
            bestDistance = length(referenceVertex - inMesh.vertices[i]);
        }
    }
    std::cout << "Found vertex: " << inMesh.vertices[bestIndex] << " at index " << bestIndex << std::endl;
    for(int i = 0; i < descriptors.length; i++) {
        descriptorZeroDuplicates[i] = descriptors[bestDistance];
    }
    ShapeDescriptor::cpu::array<int32_t> distances = ShapeDescriptor::computeRICIElementWiseModifiedSquareSumDistances(descriptors.copyToGPU(), descriptorZeroDuplicates.copyToGPU());
    ShapeDescriptor::cpu::array<float2> coordinates(distances.length);
    int max = 0;
    for(int i = 0; i < distances.length; i++) {
        max = std::max<int>(max, distances[i]);
    }
    for(int i = 0; i < distances.length; i++) {
        //std::cout << distances[i] << std::endl;
        coordinates[i] = {std::min<float>(1.0, std::max<float>(0.005, float(1.0f - (float(distances[i]) / 400.0f))) * 1.3), 0};
    }
    //inMesh.vertices[bestIndex].y += 0.1;
    ShapeDescriptor::writeOBJ(inMesh, "armadillo.obj", coordinates, "gradient.png");
    //ShapeDescriptor::writeOBJ(inMesh, "chef.obj", coordinates, "gradient.png");
    return 0;*/


    std::cout << "Reading configuration.." << std::endl;
    const std::filesystem::path configurationFileLocation(configurationFile.value());
    if(!std::filesystem::exists(configurationFileLocation)) {
        throw std::runtime_error("The specified configuration file was not found at: " + std::filesystem::absolute(configurationFileLocation).string());
    }
    std::ifstream inputStream(configurationFile.value());
    const nlohmann::json configuration = nlohmann::json::parse(inputStream);


    if(!configuration.contains("cacheDirectory")) {
        throw ShapeBench::MissingBenchmarkConfigurationException("cacheDirectory");
    }
    const std::filesystem::path cacheDirectory = configuration.at("cacheDirectory");
    if(!std::filesystem::exists(cacheDirectory)) {
        std::cout << "    Cache directory was not found. Creating a new one at: " << cacheDirectory.string() << std::endl;
        std::filesystem::create_directories(cacheDirectory);
    }


    if(!configuration.contains("compressedDatasetRootDir")) {
        throw ShapeBench::MissingBenchmarkConfigurationException("compressedDatasetRootDir");
    }
    const std::filesystem::path baseDatasetDirectory = configuration.at("objaverseDatasetRootDir");
    const std::filesystem::path derivedDatasetDirectory = configuration.at("compressedDatasetRootDir");
    const std::filesystem::path datasetCacheFile = cacheDirectory / ShapeBench::datasetCacheFileName;

    ShapeBench::Dataset dataset = ShapeBench::Dataset::computeOrLoadCached(baseDatasetDirectory, derivedDatasetDirectory, datasetCacheFile);

    uint64_t randomSeed = configuration.at("randomSeed");
    const nlohmann::json& methodSettings = configuration.at("methodSettings");

    if(methodSettings.at(ShapeBench::QUICCIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::QUICCIMethod, ShapeDescriptor::QUICCIDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    }
    if(methodSettings.at(ShapeBench::RICIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::RICIMethod, ShapeDescriptor::RICIDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    }
    if(methodSettings.at(ShapeBench::RoPSMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::RoPSMethod, ShapeDescriptor::RoPSDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    }
    if(methodSettings.at(ShapeBench::SIMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::SIMethod, ShapeDescriptor::SpinImageDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    }
    if(methodSettings.at(ShapeBench::USCMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::USCMethod, ShapeDescriptor::UniqueShapeContextDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    }
    if(methodSettings.at(ShapeBench::ShapeContextMethod::getName()).at("enabled")) {
        testMethod<ShapeBench::ShapeContextMethod, ShapeDescriptor::ShapeContextDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
    }

    // WIP:
    //testMethod<ShapeBench::FPFHMethod, ShapeDescriptor::FPFHDescriptor>(configuration, configurationFile.value(), dataset, randomSeed);
}
