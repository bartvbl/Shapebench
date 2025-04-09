#pragma once

#include "Method.h"
#include "utils/methodUtils/commonSupportVolumeIntersectionTests.h"
#include <shapeDescriptor/shapeDescriptor.h>
#include <bitset>

namespace ShapeBench {
    struct QUICCIMethod {
        static void init(const nlohmann::json &config) {

        }

        static void destroy() {

        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const ShapeDescriptor::QUICCIDescriptor& descriptor,
                const ShapeDescriptor::QUICCIDescriptor& otherDescriptor,
                float earlyExitThreshold) {
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

            uint32_t combinedNeedleUnsetPixelCount = totalBitsInDescriptor - combinedNeedleSetBitCount;

            uint32_t combinedMissingSetCount = ShapeDescriptor::warpAllReduceSum(totalMissingSetPixelCount);
            uint32_t combinedMissingUnsetCount = ShapeDescriptor::warpAllReduceSum(totalMissingUnsetPixelCount);

            float missingSetBitPenalty = float(totalBitsInDescriptor) / float(max(combinedNeedleSetBitCount, 1));
            float missingUnsetBitPenalty = float(totalBitsInDescriptor) / float(max(combinedNeedleUnsetPixelCount, 1));

            return float(combinedMissingSetCount) * missingSetBitPenalty +
                   float(combinedMissingUnsetCount) * missingUnsetBitPenalty;
        }

        static inline float computeDescriptorDistance(
                const ShapeDescriptor::QUICCIDescriptor& descriptor,
                const ShapeDescriptor::QUICCIDescriptor& otherDescriptor,
                float earlyExitThreshold) {

            const uint32_t totalBitsInDescriptor = spinImageWidthPixels * spinImageWidthPixels;
            const uint32_t chunkCount32Bit = totalBitsInDescriptor / 32;

            uint32_t combinedNeedleSetPixelCount = 0;
            uint32_t combinedHaystackSetBitCount = 0;
            uint32_t combinedMissingSetCount = 0;
            uint32_t combinedMissingUnsetCount = 0;
            for(unsigned int i = 0; i < chunkCount32Bit; i++) {
                uint32_t needleChunk = descriptor.contents[i];
                uint32_t haystackChunk = otherDescriptor.contents[i];

                combinedNeedleSetPixelCount += std::bitset<32>(needleChunk).count();
                combinedHaystackSetBitCount += std::bitset<32>(haystackChunk).count();
                combinedMissingSetCount += std::bitset<32>((needleChunk ^ haystackChunk) & needleChunk).count();
                combinedMissingUnsetCount += std::bitset<32>((~needleChunk ^ ~haystackChunk) & ~needleChunk).count();
            }

            // Needle image is black, fall back to Hamming distance
            if(combinedNeedleSetPixelCount == 0) {
                return float(combinedHaystackSetBitCount);
            }

            uint32_t combinedUnsetPixelCount = totalBitsInDescriptor - combinedNeedleSetPixelCount;

            float missingSetBitPenalty = float(totalBitsInDescriptor) / float(std::max<uint32_t>(combinedNeedleSetPixelCount, 1));
            float missingUnsetBitPenalty = float(totalBitsInDescriptor) / float(std::max<uint32_t>(combinedUnsetPixelCount, 1));

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
            return false;
        }
        static bool shouldUseGPUKernel() {
            return false;
        }
        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            Method::throwUnimplementedException();
            return 0;
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
        static ShapeDescriptor::gpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            return ShapeDescriptor::generatePartialityResistantQUICCImagesMultiRadius(mesh, descriptorOrigins, supportRadii);
        }
        static ShapeDescriptor::cpu::array<ShapeDescriptor::QUICCIDescriptor> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static std::string getName() {
            return "QUICCI";
        }

        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::FAST_CYLINDRICAL;
        }

        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            return isPointInCylindricalVolume(descriptorOrigin, supportRadius, supportRadius, samplePoint);
        }

        static nlohmann::json getMetadata() {
            nlohmann::json metadata;
            metadata["resolution"] = spinImageWidthPixels;
            metadata["distanceFunction"] = "Weighted Hamming";
            return metadata;
        }
    };
}