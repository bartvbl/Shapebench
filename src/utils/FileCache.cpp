#include "FileCache.h"


ShapeBench::FileCache::FileCache(const std::filesystem::path& cacheDirectory, size_t totalDirectorySizeLimit) : totalDirectorySizeLimit(totalDirectorySizeLimit) {
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

void ShapeBench::FileCache::insertFile(const std::filesystem::path& filePath) {
    CachedFile cachedItem;
    cachedItem.usedByThreadCount = 0;
    cachedItem.fileSizeInBytes = std::filesystem::file_size(filePath);
    cachedItem.filePath = filePath;

    assert(totalDirectorySizeLimit > cachedItem.fileSizeInBytes);

    // If our cache directory exceeds our size limit, we need to first delete enough files to make enough space
    while(totalDirectorySize > (totalDirectorySizeLimit - cachedItem.fileSizeInBytes)) {
        deleteLeastRecentlyUsedFile();
    }

    // We now get hold of the file we want to add into the cache
    if(!std::filesystem::exists(filePath)) {
        load(filePath);
        totalDirectorySize += cachedItem.fileSizeInBytes;
    }

    // When the node is inserted, it is by definition the most recently used one
    // We therefore put it in the front of the queue right away
    lruItemQueue.emplace_front(cachedItem);
    randomAccessMap[std::filesystem::absolute(filePath).string()] = lruItemQueue.begin();

    statistics.insertions++;
}

// Mark an item present in the cache as most recently used
void ShapeBench::FileCache::touchFileEntry(const std::filesystem::path& path) {
    // Move the desired node to the front of the LRU queue
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator it = randomAccessMap.find(path);
    assert(it != randomAccessMap.end());
    assert(it->second->filePath == path);
    lruItemQueue.splice(lruItemQueue.begin(), lruItemQueue, it->second);
}

void ShapeBench::FileCache::acquireFile(const std::filesystem::path& filePath) {
    std::unique_lock<std::mutex> mainLock(queueLock);
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator
            it = randomAccessMap.find(std::filesystem::absolute(filePath).string());

    if(it != randomAccessMap.end())
    {
        // FileCache hit
        statistics.hits++;
        touchFileEntry(filePath);
    } else {
        // FileCache miss. Load the item into the cache instead
        statistics.misses++;
        insertFile(filePath);
        it = randomAccessMap.find(std::filesystem::absolute(filePath).string());
    }
    it->second->usedByThreadCount++;
}

void ShapeBench::FileCache::returnFile(const std::filesystem::path& filePath) {
    std::unique_lock<std::mutex> mainLock(queueLock);
    typename std::unordered_map<std::string, typename std::list<CachedFile>::iterator>::iterator
            it = randomAccessMap.find(std::filesystem::absolute(filePath).string());
    assert(it != randomAccessMap.end());
    assert(it->second->usedByThreadCount > 0);
    it->second->usedByThreadCount--;
}

size_t ShapeBench::FileCache::getCurrentCachedDirectorySize() {
    return totalDirectorySize;
}
