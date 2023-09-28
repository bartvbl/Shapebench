#pragma once

#include <string>
#include <stdexcept>

namespace Shapebench {
    class MissingBenchmarkConfigurationException : public std::runtime_error {
    public:
        explicit MissingBenchmarkConfigurationException(std::string missingKeyName) : std::runtime_error("The specified configuration file did not contain the entry \"" + missingKeyName + "\", which is required.") {}
    };
}
