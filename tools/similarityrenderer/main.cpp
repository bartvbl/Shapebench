

#include <shapeDescriptor/shapeDescriptor.h>

int main() {
    ShapeDescriptor::cpu::Mesh inMesh = ShapeDescriptor::loadMesh("/home/bart/projects/shapebench/current-datasets/raw/Armadillo_vres2_small_scaled_0.ply", ShapeDescriptor::RecomputeNormals::ALWAYS_RECOMPUTE);
    std::cout << "Mesh has " << inMesh.vertexCount << " vertices" << std::endl;

    ShapeDescriptor::cpu::array <ShapeDescriptor::OrientedPoint> points = ShapeDescriptor::generateUniqueSpinOriginBuffer(inMesh);
    float supportRadius_armadillo = 0.05;
    ShapeDescriptor::cpu::array <ShapeDescriptor::RICIDescriptor> descriptors = ShapeDescriptor::generateRadialIntersectionCountImages(
            ShapeDescriptor::copyToGPU(inMesh), points.copyToGPU(), supportRadius_armadillo).copyToCPU();
    std::cout << "Writing image.." << std::endl;
    ShapeDescriptor::writeDescriptorImages(descriptors, "armadillo.png", false, 50, 1000);
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
    std::cout << "Found vertex: " << inMesh.vertices[bestIndex] << " at index " << bestIndex << std::endl;
    for (int i = 0; i < descriptors.length; i++) {
        descriptorZeroDuplicates[i] = descriptors[bestDistance];
    }
    ShapeDescriptor::cpu::array <int32_t> distances = ShapeDescriptor::computeRICIElementWiseModifiedSquareSumDistances(descriptors.copyToGPU(), descriptorZeroDuplicates.copyToGPU());
    ShapeDescriptor::cpu::array <float2> coordinates(distances.length);
    int max = 0;
    for (int i = 0; i < distances.length; i++) {
        max = std::max<int>(max, distances[i]);
    }
    for (int i = 0; i < distances.length; i++) {
        coordinates[i] = {std::min<float>(1.0, std::max<float>(0.005, float(1.0f - (float(distances[i]) / 400.0f))) * 1.3), 0};
    }

    ShapeDescriptor::writeOBJ(inMesh, "armadillo.obj", coordinates, "gradient.png");
    return 0;
}