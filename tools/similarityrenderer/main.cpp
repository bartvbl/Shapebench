

#include <shapeDescriptor/shapeDescriptor.h>

int main(int argc, char** argv) {
    std::string objectFilePath(argv[1]);
    ShapeDescriptor::cpu::Mesh inMesh = ShapeDescriptor::loadMesh(objectFilePath, ShapeDescriptor::RecomputeNormals::ALWAYS_RECOMPUTE);
    std::cout << "Mesh read. It has " << inMesh.vertexCount << " vertices" << std::endl;

    std::cout << "Locating vertex on mesh surface.." << std::endl;
    ShapeDescriptor::cpu::array <ShapeDescriptor::OrientedPoint> points = ShapeDescriptor::generateUniqueSpinOriginBuffer(inMesh);
    float supportRadius_armadillo = 0.05;
    ShapeDescriptor::cpu::array <ShapeDescriptor::RICIDescriptor> descriptors = ShapeDescriptor::generateRadialIntersectionCountImages(
            ShapeDescriptor::copyToGPU(inMesh), points.copyToGPU(), supportRadius_armadillo).copyToCPU();
    ShapeDescriptor::cpu::array <ShapeDescriptor::RICIDescriptor> descriptorZeroDuplicates(descriptors.length);
    ShapeDescriptor::cpu::float3 referenceVertex = {-0.05154, -0.05256, -0.00618};
    int bestIndex = 0;
    float bestDistance = 10;
    for (int i = 0; i < inMesh.vertexCount; i++) {
        if (length(referenceVertex - inMesh.vertices[i]) < bestDistance) {
            bestIndex = i;
            bestDistance = length(referenceVertex - inMesh.vertices[i]);
        }
    }
    for (int i = 0; i < descriptors.length; i++) {
        descriptorZeroDuplicates[i] = descriptors[bestDistance];
    }
    std::cout << "Computing descriptor distances.." << std::endl;
    ShapeDescriptor::cpu::array <int32_t> distances = ShapeDescriptor::computeRICIElementWiseModifiedSquareSumDistances(descriptors.copyToGPU(), descriptorZeroDuplicates.copyToGPU());
    ShapeDescriptor::cpu::array <float2> coordinates(distances.length);
    int max = 0;
    for (int i = 0; i < distances.length; i++) {
        max = std::max<int>(max, distances[i]);
    }
    for (int i = 0; i < distances.length; i++) {
        coordinates[i] = {std::min<float>(1.0, std::max<float>(0.005, float(1.0f - (float(distances[i]) / 400.0f)) * 1.3)), 0};
    }

    std::cout << "Writing mesh file.." << std::endl;
    ShapeDescriptor::writeOBJ(inMesh, "armadillo.obj", coordinates, "gradient.png");

    std::cout << "Done." << std::endl << std::endl;
    return 0;
}