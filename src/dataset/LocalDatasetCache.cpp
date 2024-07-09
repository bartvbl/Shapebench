#include "LocalDatasetCache.h"

#include <utility>
#include "curl/curl.h"
#include "fmt/format.h"

void ShapeBench::LocalDatasetCache::load(const std::filesystem::path& filePathInDataset, const std::filesystem::path& downloadPath) {
    std::cout << "Downloading: " + downloadPath.string() + " -> " + filePathInDataset.string() << std::endl;

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

    std::filesystem::path filePathOnDisk = cacheRootDirectory / filePathInDataset;
    std::filesystem::create_directories(filePathOnDisk.parent_path());

    ShapeDescriptor::cpu::Mesh mesh = ShapeDescriptor::loadMesh(temporaryDownloadFile);
    ShapeDescriptor::writeCompressedGeometryFile(mesh, filePathOnDisk, true);
    ShapeDescriptor::free(mesh);

    std::filesystem::remove(temporaryDownloadFile);
}

ShapeBench::LocalDatasetCache::LocalDatasetCache(const std::filesystem::path &localCacheDirectory,
                                                 std::string datasetBaseURL_,
                                                 size_t cacheDirectorySizeLimitBytes)
                                                 : FileCache(localCacheDirectory, cacheDirectorySizeLimitBytes),
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
