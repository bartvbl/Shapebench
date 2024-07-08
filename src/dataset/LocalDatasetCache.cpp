#include "LocalDatasetCache.h"
#include "curl/curl.h"

void ShapeBench::LocalDatasetCache::load(const std::filesystem::path &filePath) {

    CURLcode res;

    if(curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "https://example.com/");
    }

    /* cache the CA cert bundle in memory for a week */
    curl_easy_setopt(curl, CURLOPT_CA_CACHE_TIMEOUT, 604800L);

    /* Perform the request, res gets the return code */
    res = curl_easy_perform(curl);
    /* Check for errors */
    if(res != CURLE_OK)
        fprintf(stderr, "curl_easy_perform() failed: %s\n",
                curl_easy_strerror(res));
}

ShapeBench::LocalDatasetCache::LocalDatasetCache(const std::filesystem::path &localCacheDirectory,
                                                 size_t cacheDirectorySizeLimitBytes)
                                                 : FileCache(localCacheDirectory, cacheDirectorySizeLimitBytes) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
}

ShapeBench::LocalDatasetCache::~LocalDatasetCache() {
    curl_easy_cleanup(curl);
    curl_global_cleanup();
}
