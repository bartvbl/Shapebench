#pragma once

#include "utils/FileCache.h"

namespace ShapeBench {
    class RemoteDatasetCache : public FileCache {
        virtual void load(const std::filesystem::path& filePath) override;
    };
}