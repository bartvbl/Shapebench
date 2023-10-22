#pragma once

#include "Method.h"
#include <shapeDescriptor/shapeDescriptor.h>

namespace Shapebench {
    struct QUICCIMethod : public Shapebench::Method<ShapeDescriptor::QUICCIDescriptor> {
        __device__ static __inline__ float computeDescriptorDistance(
                const ShapeDescriptor::QUICCIDescriptor& descriptor,
                const ShapeDescriptor::QUICCIDescriptor& otherDescriptor) {

            const uint32_t totalBitsInDescriptor = spinImageWidthPixels * spinImageWidthPixels;
            const uint32_t chunkCount32Bit = totalBitsInDescriptor / 32;

            uint32_t totalNeedleSetBitCount = 0;
            uint32_t totalHaystackSetBitCount = 0;
            uint32_t totalMissingSetPixelCount = 0;
            uint32_t totalMissingUnsetPixelCount = 0;
            for(unsigned int i = threadIdx.x; i < chunkCount32Bit; i+= blockDim.x) {
                uint32_t needleChunk = descriptor.contents[i];
                uint32_t haystackChunk = otherDescriptor.contents[i];

                totalNeedleSetBitCount += __popc(needleChunk);
                totalHaystackSetBitCount += __popc(haystackChunk);
                totalMissingSetPixelCount += __popc((needleChunk ^ haystackChunk) & needleChunk);
                totalMissingUnsetPixelCount += __popc((~needleChunk ^ ~haystackChunk) & ~needleChunk);
            }

            uint32_t combinedNeedleSetBitCount = ShapeDescriptor::warpAllReduceSum(totalNeedleSetBitCount);
            uint32_t combinedHaystackSetBitCount = ShapeDescriptor::warpAllReduceSum(totalHaystackSetBitCount);

            // Needle image is black, fall back to Hamming distance
            if(combinedNeedleSetBitCount == 0) {
                return float(combinedHaystackSetBitCount);
            }

            uint32_t combinedSetPixelCount = ShapeDescriptor::warpAllReduceSum(totalNeedleSetBitCount);
            uint32_t combinedUnsetPixelCount = totalBitsInDescriptor - combinedSetPixelCount;

            uint32_t combinedMissingSetCount = ShapeDescriptor::warpAllReduceSum(totalMissingSetPixelCount);
            uint32_t combinedMissingUnsetCount = ShapeDescriptor::warpAllReduceSum(totalMissingUnsetPixelCount);

            float missingSetBitPenalty = float(totalBitsInDescriptor) / float(max(combinedSetPixelCount, 1));
            float missingUnsetBitPenalty = float(totalBitsInDescriptor) / float(max(combinedUnsetPixelCount, 1));

            return float(combinedMissingSetCount) * missingSetBitPenalty +
                   float(combinedMissingUnsetCount) * missingUnsetBitPenalty;
        }

        static bool usesMeshInput() {
            return true;
        }
        static bool usesPointCloudInput() {
            return false;
        }
        static bool hasGPUKernels() {
            return true;
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            return ShapeDescriptor::generateQUICCImages(mesh, descriptorOrigins, supportRadius);
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            return ShapeDescriptor::generateQUICCImages(mesh, descriptorOrigins, supportRadius);
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwIncompatibleException();
            return {};
        }
        static std::string getName() {
            return "QUICCI";
        }

        static ShapeDescriptor::cpu::array<uint32_t> computeDescriptorRanks(
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> needleDescriptors,
                ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> haystackDescriptors) {
            return ShapeDescriptor::computeQUICCImageSearchResultRanks(needleDescriptors, haystackDescriptors);
        }
    };
}