#pragma once

#include "nlohmann/json.hpp"
#include "shapeDescriptor/shapeDescriptor.h"

namespace ShapeBench {
    namespace internal {
        void initPython(const std::string& pythonMethodName, const nlohmann::json& config);
        void destroyPython();
        nlohmann::json getPythonMetadata();

        ShapeDescriptor::cpu::array<float> computePythonDescriptors(
                const ShapeDescriptor::cpu::Mesh &mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed,
                uint32_t entriesPerDescriptor);

        ShapeDescriptor::cpu::array<float> computePythonDescriptors(
                const ShapeDescriptor::cpu::PointCloud &cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed,
                uint32_t entriesPerDescriptor);
    }

    template<typename DescriptorType, typename ContentsType = float>
    class PythonAdapter {
    public:
        static void init(std::string pythonMethodName, const nlohmann::json& config) {
            internal::initPython(pythonMethodName, config);
        }

        static void destroy() {
            internal::destroyPython();
        }

        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh &mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<float> rawDescriptors = internal::computePythonDescriptors(mesh, descriptorOrigins, config, supportRadii, randomSeed, sizeof(DescriptorType) / sizeof(ContentsType));
            ShapeDescriptor::cpu::array<DescriptorType> convertedDescriptors = {descriptorOrigins.length, reinterpret_cast<DescriptorType*>(rawDescriptors.content)};
            return convertedDescriptors;
        }

        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud &cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<float> rawDescriptors = internal::computePythonDescriptors(cloud, descriptorOrigins, config, supportRadii, randomSeed, sizeof(DescriptorType) / sizeof(ContentsType));
            ShapeDescriptor::cpu::array<DescriptorType> convertedDescriptors = {descriptorOrigins.length, reinterpret_cast<DescriptorType*>(rawDescriptors.content)};
            return convertedDescriptors;
        }

        static nlohmann::json getMetadata() {
            return internal::getPythonMetadata();
        }
    };
}