#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <nlohmann/json.hpp>

namespace ShapeBench {
    template<typename Type>
    inline Type readDescriptorConfigValue(const nlohmann::json& config, std::string methodName, std::string configEntryName) {
        return config.at("methodSettings").at(methodName).at(configEntryName);
    }
    inline bool hasConfigValue(const nlohmann::json& config, std::string methodName, std::string configEntryName) {
        return config.at("methodSettings").at(methodName).contains(configEntryName);
    }

    template<typename DescriptorType>
    struct Method {
    private:
        static void throwUnimplementedException() {
            throw std::logic_error("This method is required but has not been implemented.");
        }
    protected:
        static void throwIncompatibleException() {
            throw std::runtime_error("This method does not support this method");
        }
    public:
        static void init(const nlohmann::json& config) {

        }

        static bool usesMeshInput() {
            throwUnimplementedException();
            return false;
        }
        static bool usesPointCloudInput() {
            throwUnimplementedException();
            return false;
        }
        static bool hasGPUKernels() {
            throwUnimplementedException();
            return false;
        }
        static bool shouldUseGPUKernel() {
            throwUnimplementedException();
            return false;
        }
        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh& mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud& cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint>& device_descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh& mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud& cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint>& descriptorOrigins,
                const nlohmann::json& config,
                const std::vector<float>& supportRadii,
                uint64_t randomSeed) {
            throwUnimplementedException();
            return {};
        }
        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            throwUnimplementedException();
            return false;
        }
        static std::string getName() {
            return "METHOD NAME NOT SPECIFIED";
        }
        static nlohmann::json getMetadata() {
            return {};
        }
    };
}
