#pragma once


#include <cassert>
#include <fstream>
#include <memory>
#include <string>
#include "Shader.h"
#include <filesystem>

namespace ShapeBench {
    Shader loadShader(const std::filesystem::path& directory, const std::string& fileName);
}
