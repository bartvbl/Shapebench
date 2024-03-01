#include <fstream>
#include "ComputedConfig.h"

void ShapeBench::ComputedConfig::save() {
    // Create backup of previous version of file (or include it?)
    if(std::filesystem::exists(configFilePath)) {
        int backupVersion = 1;
        std::filesystem::path chosenPath = configFilePath;

        while(true) {
            std::filesystem::path numberedPath = chosenPath.replace_extension(".bak" + std::to_string(backupVersion));
            if(!std::filesystem::exists(numberedPath)) {
                chosenPath = numberedPath;
                break;
            }
            backupVersion++;
        }
        // Save new version of file
        std::filesystem::copy_file(configFilePath, chosenPath);
    }

    std::ofstream outStream{configFilePath};
    outStream << configValues.dump(4);
}

ShapeBench::ComputedConfig::ComputedConfig(const std::filesystem::path &configFileLocation) : configFilePath{configFileLocation} {
    if(!std::filesystem::exists(configFileLocation)) {
        // Create empty config file
        std::ofstream emptyConfigFileStream{configFileLocation};
        emptyConfigFileStream << "{" << std::endl <<
                                 "    \"version\": 1.0" << std::endl <<
                                 "}";
    }
    std::ifstream inStream{configFileLocation};
    configValues = nlohmann::json::parse(inStream);
}

void ensureMethodIsPresent(nlohmann::json& config, std::string methodName) {
    if(!config.contains(methodName)) {
        config[methodName] = {};
    }
}

float ShapeBench::ComputedConfig::getFloat(std::string methodName, std::string valueName) {
    ensureMethodIsPresent(configValues, methodName);
    return configValues.at(methodName).at(valueName);
}

int32_t ShapeBench::ComputedConfig::getInt(std::string methodName, std::string valueName) {
    ensureMethodIsPresent(configValues, methodName);
    return configValues.at(methodName).at(valueName);
}

std::string ShapeBench::ComputedConfig::getString(std::string methodName, std::string valueName) {
    ensureMethodIsPresent(configValues, methodName);
    return configValues.at(methodName).at(valueName);
}

void ShapeBench::ComputedConfig::setFloatAndSave(std::string methodName, std::string valueName, float value) {
    ensureMethodIsPresent(configValues, methodName);
    configValues.at(methodName)[valueName] = value;
    save();
}

void ShapeBench::ComputedConfig::setIntAndSave(std::string methodName, std::string valueName, int32_t value) {
    ensureMethodIsPresent(configValues, methodName);
    configValues.at(methodName)[valueName] = value;
    save();
}

void ShapeBench::ComputedConfig::setStringAndSave(std::string methodName, std::string valueName, std::string value) {
    ensureMethodIsPresent(configValues, methodName);
    configValues.at(methodName)[valueName] = value;
    save();
}

bool ShapeBench::ComputedConfig::containsKey(std::string methodName, std::string valueName) {
    return configValues.contains(methodName) && configValues.at(methodName).contains(valueName);
}

nlohmann::json ShapeBench::ComputedConfig::toJSON() const {
    return configValues;
}
