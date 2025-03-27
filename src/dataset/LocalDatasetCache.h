#pragma once

#include "utils/FileCache.h"
#include "curl/curl.h"

namespace ShapeBench {
    class LocalDatasetCache : public FileCache {
        CURL *curl = nullptr;
        std::string datasetBaseURL;
        std::filesystem::path temporaryDownloadFile;

        virtual void load(const std::filesystem::path& filePathInDataset, const std::filesystem::path& downloadURL, const std::string& expectedFileHash) override;

    public:
        LocalDatasetCache(const std::filesystem::path& localCacheDirectory,
                          std::string  datasetBaseURL,
                          size_t cacheDirectorySizeLimitBytes,
                          bool enableFileIntegrityVerification);
        ~LocalDatasetCache() override;
    };
}