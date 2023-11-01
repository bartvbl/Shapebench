#pragma once

#include <shapeDescriptor/shapeDescriptor.h>

namespace Shapebench {
    template<typename Type>
    inline Type readDescriptorConfigValue(const nlohmann::json& config, std::string methodName, std::string configEntryName) {
        for(const nlohmann::json& entry : config.at("methodsToTest")) {
            if(entry.at("name") == methodName) {
                return Type(entry.at(configEntryName));
            }
        }
        throw std::runtime_error("Config entry \"" + configEntryName + "\" for method " + methodName + " not found in config file");
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
        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                ShapeDescriptor::gpu::Mesh mesh,
                ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> device_descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                ShapeDescriptor::cpu::Mesh mesh,
                ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> descriptorOrigins,
                const nlohmann::json& config,
                float supportRadius) {
            throwUnimplementedException();
            return {};
        }
        static ShapeDescriptor::cpu::array<uint32_t> computeDescriptorRanks(
                ShapeDescriptor::gpu::array<DescriptorType> needleDescriptors,
                ShapeDescriptor::gpu::array<DescriptorType> haystackDescriptors
                ) {
            throwUnimplementedException();
            return {};
        }
        static std::string getName() {
            return "METHOD NAME NOT SPECIFIED";
        }
    };




}
