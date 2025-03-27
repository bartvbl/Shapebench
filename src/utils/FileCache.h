#pragma once

#include <mutex>
#include <list>
#include <unordered_map>
#include <cassert>
#include <omp.h>
#include <iostream>
#include <condition_variable>
#include <thread>
#include <malloc.h>
#include <vector>
#include <filesystem>
#include <shapeDescriptor/shapeDescriptor.h>

namespace ShapeBench {
    // The cached nodes are stored as pointers to avoid accidental copies being created
    struct CachedFile {
        uint32_t usedByThreadCount = 0;
        size_t fileSizeInBytes = 0;
        std::filesystem::path filePath = "";

        CachedFile() = default;
    };

    struct CacheStatistics {
        size_t misses = 0;
        size_t hits = 0;
        size_t evictions = 0;
        size_t insertions = 0;

        void reset() {
            misses = 0;
            hits = 0;
            evictions = 0;
            insertions = 0;
        }
    };

    class FileCache {
    protected:
        // Nodes are evicted on a Least Recently Used basis
        // This is most efficiently done by using a doubly linked list
        std::list<CachedFile> lruItemQueue;

        // These hash tables allow efficient fetching of nodes from the cache
        std::unordered_map<std::string, typename std::list<CachedFile>::iterator> randomAccessMap;

        // Lock used for modification of the cache data structures.
        std::mutex queueLock;
        std::condition_variable queueConditionVariable;

        const size_t totalDirectorySizeLimit;
        size_t totalDirectorySize = 0;
        CacheStatistics statistics;
        bool verifyFileIntegrity = false;



        explicit FileCache(const std::filesystem::path& cacheDirectory, size_t totalDirectorySizeLimit, bool enableFileIntegrityVerification);

        void deleteLeastRecentlyUsedFile();
        void insertFile(const std::filesystem::path& filePathInDataset, const std::filesystem::path& downloadURL, const std::string& expectedFileHash);
        void touchFileEntry(const std::filesystem::path& filePathOnDisk);

        // What needs to happen when a cache miss or eviction occurs depends on the specific use case
        // Since this class is a general implementation, a subclass needs to implement this functionality.

        // May be called by multiple threads simultaneously
        virtual void load(const std::filesystem::path& filePathInDataset, const std::filesystem::path& downloadURL, const std::string& expectedFileHash) = 0;

    protected:
        const std::filesystem::path cacheRootDirectory;

    public:
        void acquireFile(const std::filesystem::path& filePathInDataset, const std::filesystem::path& downloadURL, const std::string& expectedFileHash = std::string("NO HASH SPECIFIED"));
        void returnFile(const std::filesystem::path& filePathInDataset);

        size_t getCurrentCachedDirectorySize() const;

        virtual ~FileCache() = default;
    };
}
