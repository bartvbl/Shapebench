#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    struct DescriptorDistance {
        float mean = 0;
        float variance = 0;
    };

    constexpr uint32_t warpsPerBlock = 16;

    // Kernel assumes 1D blocks with multiples of 32 in width
    template<typename DescriptorMethod, typename DescriptorType>
    __global__ void referenceSetDistanceKernel(
            ShapeDescriptor::gpu::array<DescriptorType> sampleDescriptors,
            ShapeDescriptor::gpu::array<DescriptorType> referenceDescriptors,
            ShapeDescriptor::gpu::array<DescriptorDistance> distances) {
        __shared__ DescriptorType sampleDescriptor;
        __shared__ float runningAverages[warpsPerBlock];
        __shared__ uint32_t averageCount[warpsPerBlock];
        __shared__ float runningSquaredSums[warpsPerBlock];
        const uint32_t sampleDescriptorIndex = blockIdx.x;
        static_assert(sizeof(DescriptorType) % sizeof(float) == 0, "This descriptor does not appear to use integers or floats as its storage type. Please implement a special case for it.");
        uint32_t threadIndex = threadIdx.y * blockDim.x + threadIdx.x;
        for(uint32_t i = threadIndex; i < (sizeof(DescriptorType) / sizeof(float)); i += blockDim.x * blockDim.y) {
            sampleDescriptor.contents[i] = sampleDescriptors[sampleDescriptorIndex].contents[i];
        }

        __syncthreads();

        uint32_t runningCount = 0;
        float runningMean = 0;
        float runningSumOfSquaredDifferences = 0;

        for(uint32_t referenceDescriptorIndex = threadIdx.y; referenceDescriptorIndex < referenceDescriptors.length; referenceDescriptorIndex += blockDim.y) {
            float similarity = DescriptorMethod::computeDescriptorDistanceGPU(sampleDescriptor, referenceDescriptors[referenceDescriptorIndex], __FLT_MAX__);
            // Using Welford's algorithm for computing the mean and variance
            // Updating the running mean and standard deviation values
            runningCount++;
            float delta = similarity - runningMean;
            runningMean += delta / float(runningCount);
            float delta2 = similarity - runningMean;
            runningSumOfSquaredDifferences += delta * delta2;
        }

        if(threadIdx.x == 0) {
            runningAverages[threadIdx.y] = runningMean;
            averageCount[threadIdx.y] = runningCount;
            runningSquaredSums[threadIdx.y] = runningSumOfSquaredDifferences;
        }

        __syncthreads();

        // Computing the variance
        if(threadIdx.x == 0 && threadIdx.y == 0) {
            float globalMean = 0;
            float globalSquaredSum = 0;
            for(int i = 0; i < blockDim.y; i++) {
                float fractionOfTotal = float(averageCount[i]) / float(referenceDescriptors.length);
                globalMean += fractionOfTotal * runningAverages[i];
                globalSquaredSum += runningSquaredSums[i];
            }

            DescriptorDistance distance;
            distance.mean = globalMean;
            distance.variance = globalSquaredSum / float(referenceDescriptors.length - 1);
            distances.content[sampleDescriptorIndex] = distance;
        }
    };

    template<typename DescriptorMethod, typename DescriptorType>
    void referenceSetDistanceKernelCPU(
            ShapeDescriptor::cpu::array<DescriptorType> sampleDescriptors,
            ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors,
            ShapeDescriptor::cpu::array<DescriptorDistance> distances) {

        #pragma omp parallel for schedule(dynamic)
        for(uint32_t sampleDescriptorIndex = 0; sampleDescriptorIndex < sampleDescriptors.length; sampleDescriptorIndex++) {
            uint32_t runningCount = 0;
            float runningMean = 0;
            float runningSumOfSquaredDifferences = 0;

            for(uint32_t referenceDescriptorIndex = 0; referenceDescriptorIndex < referenceDescriptors.length; referenceDescriptorIndex++) {
                float similarity = DescriptorMethod::computeDescriptorDistance(sampleDescriptors[sampleDescriptorIndex],
                                                                               referenceDescriptors[referenceDescriptorIndex], std::numeric_limits<float>::max());
                // Using Welford's algorithm for computing the mean and variance
                // Updating the running mean and standard deviation values
                runningCount++;
                float delta = similarity - runningMean;
                runningMean += delta / float(runningCount);
                float delta2 = similarity - runningMean;
                runningSumOfSquaredDifferences += delta * delta2;
            }

            // Computing the variance
            DescriptorDistance distance;
            distance.mean = runningMean;
            // Note: will be NaN if runningCount <= 1
            distance.variance = runningSumOfSquaredDifferences / float(runningCount - 1);

            distances.content[sampleDescriptorIndex] = distance;
        }
    }

    template<typename DescriptorMethod, typename DescriptorType>
    void referenceSetDistanceKernelGPU(
            ShapeDescriptor::cpu::array<DescriptorType> sampleDescriptors,
            ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors,
            ShapeDescriptor::cpu::array<DescriptorDistance> distances) {
        ShapeDescriptor::gpu::array<DescriptorType> device_sampleDescriptors = sampleDescriptors.copyToGPU();
        ShapeDescriptor::gpu::array<DescriptorType> device_referenceDescriptors = referenceDescriptors.copyToGPU();
        ShapeDescriptor::gpu::array<DescriptorDistance> device_distances = distances.copyToGPU();

        dim3 blockDimensions = {32, warpsPerBlock, 1};
        referenceSetDistanceKernel<DescriptorMethod, DescriptorType><<<sampleDescriptors.length, blockDimensions>>>(device_sampleDescriptors, device_referenceDescriptors, device_distances);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaMemcpy(distances.content, device_distances.content, distances.length * sizeof(DescriptorDistance), cudaMemcpyDeviceToHost));

        ShapeDescriptor::free(device_sampleDescriptors);
        ShapeDescriptor::free(device_referenceDescriptors);
        ShapeDescriptor::free(device_distances);
    }

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

    template<typename DescriptorMethod, typename DescriptorType>
    std::vector<DescriptorDistance> computeReferenceSetDistance(
            ShapeDescriptor::cpu::array<DescriptorType> sampleDescriptors,
            ShapeDescriptor::cpu::array<DescriptorType> referenceDescriptors,
            bool useGPU) {

        std::vector<DescriptorDistance> descriptorDistances(sampleDescriptors.length);

        if(useGPU) {
            referenceSetDistanceKernelGPU<DescriptorMethod, DescriptorType>(sampleDescriptors, referenceDescriptors, {sampleDescriptors.length, descriptorDistances.data()});
        } else {
            referenceSetDistanceKernelCPU<DescriptorMethod, DescriptorType>(sampleDescriptors, referenceDescriptors, {sampleDescriptors.length, descriptorDistances.data()});
        }

        return descriptorDistances;
    }
}
