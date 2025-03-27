#include "LocalDatasetCache.h"

#include <utility>
#include "curl/curl.h"
#include "fmt/format.h"
#include "utils/FileHasher.h"

void ShapeBench::LocalDatasetCache::load(const std::filesystem::path& filePathInDataset,
                                         const std::filesystem::path& downloadPath,
                                         const std::string& expectedFileHash) {
    double directorySizeGB = double(totalDirectorySize) / double(1024 * 1024 * 1024);
    double directorySizeLimitGB = double(totalDirectorySizeLimit) / double(1024 * 1024 * 1024);
    std::cout << "Downloading: " + downloadPath.string() + " -> " + filePathInDataset.string() << " (cache capacity used: " << directorySizeGB << "GB / " << directorySizeLimitGB << "GB)" << std::endl;

    std::string downloadURL = fmt::format(fmt::runtime(datasetBaseURL), downloadPath.string());
    FILE* temporaryFile = fopen(temporaryDownloadFile.c_str(), "wb");
    curl_easy_setopt(curl, CURLOPT_URL, downloadURL.c_str());

    curl_easy_setopt(curl, CURLOPT_WRITEDATA, temporaryFile);
    curl_easy_setopt(curl, CURLOPT_CA_CACHE_TIMEOUT, 604800L);

    CURLcode res = curl_easy_perform(curl);
    if(res != CURLE_OK) {
        throw std::logic_error("FATAL: failed to download file: " + downloadPath.string());
    }
    fclose(temporaryFile);

    if(verifyFileIntegrity) {
        std::string fileHash = ShapeBench::computeFileHash(temporaryDownloadFile);
        if(fileHash != expectedFileHash) {
            throw std::runtime_error("Error: hash of downloaded file did not match the one on record. File: " + filePathInDataset.string());
        }
    }

    std::filesystem::path filePathOnDisk = cacheRootDirectory / filePathInDataset;
    std::filesystem::create_directories(filePathOnDisk.parent_path());

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(temporaryDownloadFile);
    ShapeDescriptor::writeCompressedGeometryFile(mesh, filePathOnDisk, true);
    ShapeDescriptor::free(mesh);

    std::filesystem::remove(temporaryDownloadFile);
}

ShapeBench::LocalDatasetCache::LocalDatasetCache(const std::filesystem::path &localCacheDirectory,
                                                 std::string datasetBaseURL_,
                                                 size_t cacheDirectorySizeLimitBytes,
                                                 bool enableFileIntegrityVerification)
                                                 : FileCache(localCacheDirectory, cacheDirectorySizeLimitBytes,
                                                             enableFileIntegrityVerification),
                                                 datasetBaseURL(datasetBaseURL_){
    temporaryDownloadFile = localCacheDirectory / "download.glb";
    if(std::filesystem::exists(temporaryDownloadFile)) {
        std::filesystem::remove(temporaryDownloadFile);
    }
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if(!curl) {
        throw std::logic_error("FATAL: Failed to initialise curl. File downloads will not be possible.");
    }

    curl_easy_setopt(curl, CURLOPT_MAXREDIRS , 5);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, true);
    curl_easy_setopt(curl, CURLOPT_COOKIEFILE, "");
}

ShapeBench::LocalDatasetCache::~LocalDatasetCache() {
    curl_easy_cleanup(curl);
    curl_global_cleanup();
    if(std::filesystem::exists(temporaryDownloadFile)) {
        std::filesystem::remove(temporaryDownloadFile);
    }
}
