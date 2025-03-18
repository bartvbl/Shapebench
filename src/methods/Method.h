#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <nlohmann/json.hpp>
#include "types/IntersectingAreaParameters.h"
#include "types/IntersectingAreaEstimationStrategy.h"

namespace ShapeBench {
    template<typename Type>
    inline Type readDescriptorConfigValue(const nlohmann::json& config, std::string methodName, std::string configEntryName) {
        return config.at("methodSettings").at(methodName).at(configEntryName);
    }
    inline bool hasConfigValue(const nlohmann::json& config, std::string methodName, std::string configEntryName) {
        return config.at("methodSettings").at(methodName).contains(configEntryName);
    }

    struct Method {
        static void throwUnimplementedException() {
            throw std::logic_error("This method is required but has not been implemented.");
        }
        static void throwIncompatibleException() {
            throw std::runtime_error("This method does not support this method");
        }
    };
}
