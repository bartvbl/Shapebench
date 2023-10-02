#pragma once

#include <filesystem>

namespace Shapebench {
    // Takes in a dataset of file formats supported by the libShapeDescriptor library and compresses it using the library's compact mesh format
    // Can optionally produce a JSON file with dataset metadata
    void computeCompressedDataSet(const std::filesystem::path& originalDatasetDirectory, const std::filesystem::path& compressedDatasetDirectory, std::filesystem::path writeMetadataFilePath = "NOT_SPECIFIED");
}