#pragma once

#include "nlohmann/json.hpp"

namespace ShapeBench {
    class ComputedConfig {
        nlohmann::json configValues;
        std::filesystem::path configFilePath;

        void save();

    public:
        nlohmann::json toJSON() const;

        explicit ComputedConfig(const std::filesystem::path& configFileLocation);
        explicit ComputedConfig() = default;

        float getFloat(std::string methodName, std::string valueName);
        int32_t getInt(std::string methodName, std::string valueName);
        std::string getString(std::string methodName, std::string valueName);

        bool containsKey(std::string methodName, std::string valueName);

        void setFloatAndSave(std::string methodName, std::string valueName, float value);
        void setIntAndSave(std::string methodName, std::string valueName, int32_t value);
        void setStringAndSave(std::string methodName, std::string valueName, std::string value);
    };
}
