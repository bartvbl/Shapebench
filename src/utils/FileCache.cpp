#include "FileCache.h"


ShapeBench::FileCache::FileCache(const std::filesystem::path& cacheDirectory, size_t totalDirectorySizeLimit) : cacheRootDirectory(cacheDirectory), totalDirectorySizeLimit(totalDirectorySizeLimit) {
    randomAccessMap.reserve(1000);
    std::vector<std::filesystem::path> filesInDirectory = ShapeDescriptor::listDirectory(cacheDirectory);
    for(const std::filesystem::path& filePath : filesInDirectory) {
        insertFile(filePath);
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
    std::filesystem::remove(evictedItem.filePath);

    // Remove entry from the cache
    typename std::list<CachedFile>::iterator it = std::next(leastRecentlyUsedItem).base();
    totalDirectorySize -= evictedItem.fileSizeInBytes;
    this->lruItemQueue.erase(it);
    this->randomAccessMap.erase(std::filesystem::absolute(evictedItem.filePath).string());
    assert(randomAccessMap.find(std::filesystem::absolute(evictedItem.filePath).string()) == randomAccessMap.end());
}

void ShapeBench::FileCache::insertFile(const std::filesystem::path& filePathInDataset) {
    CachedFile cachedItem;
    cachedItem.usedByThreadCount = 0;
    cachedItem.filePath = cacheRootDirectory / filePathInDataset;
    cachedItem.fileSizeInBytes = std::filesystem::file_size(cachedItem.filePath);

    assert(totalDirectorySizeLimit > cachedItem.fileSizeInBytes);

    // If our cache directory exceeds our size limit, we need to first delete enough files to make enough space
    while(totalDirectorySize > (totalDirectorySizeLimit - cachedItem.fileSizeInBytes)) {
        deleteLeastRecentlyUsedFile();
    }

    // We now get hold of the file we want to add into the cache
    if(!std::filesystem::exists(cachedItem.filePath)) {
        load(filePathInDataset);
        totalDirectorySize += cachedItem.fileSizeInBytes;
    }

    // When the node is inserted, it is by definition the most recently used one
    // We therefore put it in the front of the queue right away
    lruItemQueue.emplace_front(cachedItem);
    randomAccessMap[std::filesystem::absolute(cachedItem.filePath).string()] = lruItemQueue.begin();

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

void ShapeBench::FileCache::acquireFile(const std::filesystem::path& filePathInDataset) {
    std::filesystem::path filePathOnDisk = std::filesystem::absolute(cacheRootDirectory / filePathInDataset);
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
        insertFile(filePathInDataset);
        it = randomAccessMap.find(std::filesystem::absolute(filePathOnDisk).string());
    }
    it->second->usedByThreadCount++;
}

void ShapeBench::FileCache::returnFile(const std::filesystem::path& filePathInDataset) {
    std::unique_lock<std::mutex> mainLock(queueLock);
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator
            it = randomAccessMap.find(std::filesystem::absolute(cacheRootDirectory / filePathInDataset).string());
    assert(it != randomAccessMap.end());
    assert(it->second->usedByThreadCount > 0);
    it->second->usedByThreadCount--;
}

size_t ShapeBench::FileCache::getCurrentCachedDirectorySize() const {
    return totalDirectorySize;
}

