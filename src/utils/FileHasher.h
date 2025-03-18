#pragma once

#include <string>
#include <filesystem>
#include "shapeDescriptor/shapeDescriptor.h"

namespace ShapeBench {
    std::string computeFileHash(const std::filesystem::path& filePath);
    std::string computePointCloudHash(const ShapeDescriptor::cpu::PointCloud& cloud);
    std::string computeMeshHash(const ShapeDescriptor::cpu::Mesh& mesh);
}

