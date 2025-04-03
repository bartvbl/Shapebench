#include "FileCache.h"


ShapeBench::FileCache::FileCache(const std::filesystem::path& cacheDirectory, size_t totalDirectorySizeLimit, bool enableFileIntegrityVerification)
    : cacheRootDirectory(cacheDirectory), totalDirectorySizeLimit(totalDirectorySizeLimit) {
    randomAccessMap.reserve(1000);
    std::vector<std::filesystem::path> filesInDirectory = ShapeDescriptor::listDirectoryAndSubdirectories(cacheDirectory);
    for(const std::filesystem::path& filePath : filesInDirectory) {
        // Download URL can be empty because file exists
        if(std::filesystem::is_directory(filePath)) {
            continue;
        }
        verifyFileIntegrity = false;
        insertFile(std::filesystem::absolute(filePath), "", "NOT_SPECIFIED_ON_PURPOSE");
        verifyFileIntegrity = enableFileIntegrityVerification;
    }
}

void ShapeBench::FileCache::deleteLeastRecentlyUsedFile() {
    statistics.evictions++;
    typename std::list<CachedFile>::reverse_iterator leastRecentlyUsedItem;

    bool foundEntry = false;
    for(auto entry = lruItemQueue.rbegin(); entry != lruItemQueue.rend(); ++entry) {
        if(entry->usedByThreadCount == 0) {
            leastRecentlyUsedItem = entry;
            foundEntry = true;
            break;
        }
    }

    if(!foundEntry) {
        return;
    }

    // Make a copy so we don't rely on the iterator
    CachedFile evictedItem = *leastRecentlyUsedItem;
    assert(evictedItem.usedByThreadCount == 0);

    // Delete file from disk
    std::cout << "Deleting: " << evictedItem.filePath.string() << std::endl;
    std::filesystem::remove(evictedItem.filePath);

    // Remove entry from the cache
    typename std::list<CachedFile>::iterator it = std::next(leastRecentlyUsedItem).base();
    totalDirectorySize -= evictedItem.fileSizeInBytes;
    this->lruItemQueue.erase(it);
    this->randomAccessMap.erase(std::filesystem::absolute(evictedItem.filePath).string());
    assert(randomAccessMap.find(std::filesystem::absolute(evictedItem.filePath).string()) == randomAccessMap.end());
}

void ShapeBench::FileCache::insertFile(const std::filesystem::path& filePathInDataset,
                                       const std::filesystem::path& downloadURL, const std::string& expectedFileHash) {
    CachedFile cachedItem;
    cachedItem.usedByThreadCount = 0;
    cachedItem.filePath = filePathInDataset;

    std::string fileIdentifier = std::filesystem::absolute(cachedItem.filePath).string();

    if(randomAccessMap.contains(fileIdentifier)) {
        // File already exists
        return;
    }


    assert(totalDirectorySizeLimit > cachedItem.fileSizeInBytes);

    // If our cache directory exceeds our size limit, we need to first delete enough files to make enough space
    while(totalDirectorySize > (totalDirectorySizeLimit - cachedItem.fileSizeInBytes)) {
        deleteLeastRecentlyUsedFile();
    }

    // We now get hold of the file we want to add into the cache
    if(!std::filesystem::exists(cachedItem.filePath)) {
        load(filePathInDataset, downloadURL, expectedFileHash);
    }

    totalDirectorySize += std::filesystem::file_size(cachedItem.filePath);
    cachedItem.fileSizeInBytes = std::filesystem::file_size(cachedItem.filePath);

    // When the node is inserted, it is by definition the most recently used one
    // We therefore put it in the front of the queue right away
    lruItemQueue.emplace_front(cachedItem);
    randomAccessMap[fileIdentifier] = lruItemQueue.begin();

    statistics.insertions++;
}

// Mark an item present in the cache as most recently used
void ShapeBench::FileCache::touchFileEntry(const std::filesystem::path& filePathOnDisk) {
    // Move the desired node to the front of the LRU queue
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator it = randomAccessMap.find(filePathOnDisk);
    assert(it != randomAccessMap.end());
    assert(it->second->filePath == filePathOnDisk);
    lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, it->second);
}

void ShapeBench::FileCache::acquireFile(const std::filesystem::path& filePathInDataset, const std::filesystem::path& downloadURL, const std::string& expectedFileHash) {
    if(verifyFileIntegrity && expectedFileHash == "NO HASH SPECIFIED") {
        throw std::runtime_error("File cache has enabled verification of file integrity, but no hash was specified.");
    }
    std::filesystem::path filePathOnDisk = std::filesystem::absolute(filePathInDataset);
    std::unique_lock<std::mutex> mainLock(queueLock);
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator
            it = randomAccessMap.find(filePathOnDisk.string());

    if(it != randomAccessMap.end())
    {
        // FileCache hit
        statistics.hits++;
        touchFileEntry(filePathOnDisk);
    } else {
        // FileCache miss. Load the item into the cache instead
        statistics.misses++;
        insertFile(filePathOnDisk, downloadURL, expectedFileHash);
        it = randomAccessMap.find(std::filesystem::absolute(filePathOnDisk).string());
    }
    it->second->usedByThreadCount++;
}

void ShapeBench::FileCache::returnFile(const std::filesystem::path& filePathInDataset) {
    std::unique_lock<std::mutex> mainLock(queueLock);
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator
            it = randomAccessMap.find(std::filesystem::absolute(filePathInDataset).string());
    assert(it != randomAccessMap.end());
    assert(it->second->usedByThreadCount > 0);
    it->second->usedByThreadCount--;
}

size_t ShapeBench::FileCache::getCurrentCachedDirectorySize() const {
    return totalDirectorySize;
}