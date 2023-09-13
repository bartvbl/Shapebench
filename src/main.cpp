

#include "arrrgh.hpp"
#include "shapeDescriptor/utilities/CUDAContextCreator.h"
#include "shapeDescriptor/utilities/CUDAAvailability.h"
#include "shapeDescriptor/utilities/read/GLTFLoader.h"
#include "shapeDescriptor/utilities/dump/meshDumper.h"

int main(int argc, const char** argv) {
    arrrgh::parser parser("shapebench", "Benchmark tool for 3D local shape descriptors");
    const auto& showHelp = parser.add<bool>(
            "help", "Show this help message", 'h', arrrgh::Optional, false);
    const auto& datasetDirectory = parser.add<std::string>(
            "dataset-directory", "Directory containing OBJ, PLY, or OFF files", '\0', arrrgh::Optional, "");
    const auto& randomSeed = parser.add<std::string>(
            "random-seed", "Random seed to use for experiments", '\0', arrrgh::Optional, "");
    const auto& experiment = parser.add<std::string>(
            "experiment", "Which experiment to use", '\0', arrrgh::Optional, "");
    const auto& method = parser.add<std::string>(
            "method", "Which algorithm/method to test", '\0', arrrgh::Optional, "");
    const auto& distanceFunction = parser.add<std::string>(
            "distance-function", "Which distance function to use for comparing descriptors (must be compatible with the chosen method)", '\0', arrrgh::Optional, "");
    const auto& forceGPU = parser.add<int>(
            "force-gpu", "Use the GPU with the given ID (as shown in nvidia-smi)", '\0', arrrgh::Optional, -1);

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

    if(forceGPU.value() != -1) {
        std::cout << "Forcing GPU " << forceGPU.value() << std::endl;
        ShapeDescriptor::utilities::createCUDAContext(forceGPU.value());
    }

    if(!ShapeDescriptor::isCUDASupportAvailable()) {
        throw std::runtime_error("This benchmark requires CUDA support to operate.");
    }

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::utilities::loadGLTFMesh("/mnt/VOID/datasets/objaverse/hf-objaverse-v1/glbs/000-023/cd19be32833148cbbcfd78453b5d49e7.glb", ShapeDescriptor::RecomputeNormals::DO_NOT_RECOMPUTE);
    ShapeDescriptor::dump::mesh(mesh, "output.obj");

    // Generate representative descriptor set
        // Compute descriptor set
        // Compute distance list

    // Determine optimal support radius
        // For a range of support radii:
            // While distance between descriptor pair has not exceeded noise floor
                // increase offset between support radii and regenerate descriptor pair
    //
}
