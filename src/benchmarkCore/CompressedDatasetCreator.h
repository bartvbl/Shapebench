#pragma once

#include <filesystem>
#include "json.hpp"

namespace ShapeBench {
    // Takes in a dataset of file formats supported by the libShapeDescriptor library and compresses it using the library's compact mesh format
    // Can optionally produce a JSON file with dataset metadata
    nlohmann::json computeOrReadDatasetCache(const std::filesystem::path& originalDatasetDirectory, const std::filesystem::path& compressedDatasetDirectory, const std::filesystem::path& writeMetadataFilePath = "NOT_SPECIFIED");
}