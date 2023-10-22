#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

struct DescriptorDistance {
    float mean = 0;
    float variance = 0;
};

namespace internal {

}

// Kernel assumes 1D blocks with multiples of 32 in width
template<typename DescriptorMethod, typename DescriptorType>
__global__ void referenceSetDistanceKernel(
        ShapeDescriptor::gpu::array<DescriptorType> sampleDescriptors,
        ShapeDescriptor::gpu::array<DescriptorType> referenceDescriptors,
        ShapeDescriptor::gpu::array<DescriptorDistance> distances) {
    __shared__ DescriptorType sampleDescriptor;
    const uint32_t sampleDescriptorIndex = blockIdx.x;
    static_assert(sizeof(DescriptorType) % sizeof(float) == 0, "This descriptor does not appear to use integers or floats as its storage type. Please implement a special case for it.");
    for(uint32_t i = threadIdx.x; i < (sizeof(DescriptorType) / sizeof(float)); i += blockDim.x) {
        sampleDescriptor.contents[i] = sampleDescriptors[sampleDescriptorIndex].contents[i];
    }

    __syncthreads();

    uint32_t runningCount = 0;
    float runningMean = 0;
    float runningSumOfSquaredDifferences = 0;

    for(uint32_t referenceDescriptorIndex = 0; referenceDescriptorIndex < referenceDescriptors.length; referenceDescriptorIndex++) {
        float similarity = DescriptorMethod::computeDescriptorDistance(sampleDescriptor,
                                                                       referenceDescriptors[referenceDescriptorIndex]);
        // Using Welford's algorithm for computing the mean and variance
        // Updating the running mean and standard deviation values
        runningCount++;
        float delta = similarity - runningMean;
        runningMean += delta / float(runningCount);
        float delta2 = similarity - runningMean;
        runningSumOfSquaredDifferences += delta * delta2;
    }

    // Computing the variance
    if(threadIdx.x == 0) {
        DescriptorDistance distance;
        distance.mean = runningMean;
        // Note: will be NaN if runningCount <= 1
        distance.variance = runningSumOfSquaredDifferences / float(runningCount - 1);

        distances[sampleDescriptorIndex] = distance;
    }
};

template<typename DescriptorMethod, typename DescriptorType>
ShapeDescriptor::cpu::array<DescriptorDistance> computeReferenceSetDistance(
        ShapeDescriptor::gpu::array<DescriptorType> sampleDescriptors,
        ShapeDescriptor::gpu::array<DescriptorType> referenceDescriptors) {

    ShapeDescriptor::gpu::array<DescriptorDistance> device_descriptorDistances(sampleDescriptors.length);

    referenceSetDistanceKernel<DescriptorMethod, DescriptorType><<<sampleDescriptors.length, 32>>>(sampleDescriptors, referenceDescriptors, device_descriptorDistances);
    checkCudaErrors(cudaDeviceSynchronize());

    ShapeDescriptor::cpu::array<DescriptorDistance> descriptorDistances = device_descriptorDistances.copyToCPU();
    ShapeDescriptor::free(device_descriptorDistances);

    return descriptorDistances;
}