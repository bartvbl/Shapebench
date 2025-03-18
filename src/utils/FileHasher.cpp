#include "FileHasher.h"
#include "sha1.hpp"
#include "shapeDescriptor/shapeDescriptor.h"

void appendToHash(SHA1& checksum, const char* buffer, size_t length) {
    // How this should be solved in the future:
    // std::span<char> bufferSpan(decompressedFileContents);
    // std::spanstream stream(bufferSpan);
    std::string temporaryCopy(buffer, buffer + length);
    checksum.update(temporaryCopy);
}

std::string ShapeBench::computeFileHash(const std::filesystem::path& filePath) {
    return SHA1::from_file(filePath.string());
}

std::string ShapeBench::computePointCloudHash(const ShapeDescriptor::cpu::PointCloud &cloud) {
    SHA1 checksum;
    appendToHash(checksum, reinterpret_cast<const char*>(cloud.vertices), cloud.pointCount * sizeof(ShapeDescriptor::cpu::float3));
    if(cloud.hasVertexNormals) {
        appendToHash(checksum, reinterpret_cast<const char*>(cloud.normals), cloud.pointCount * sizeof(ShapeDescriptor::cpu::float3));
    }
    if(cloud.hasVertexColours) {
        appendToHash(checksum, reinterpret_cast<const char*>(cloud.vertexColours), cloud.pointCount * sizeof(ShapeDescriptor::cpu::uchar4));
    }

    return checksum.final();
}

std::string ShapeBench::computeMeshHash(const ShapeDescriptor::cpu::Mesh &mesh) {
    SHA1 checksum;
    appendToHash(checksum, reinterpret_cast<const char*>(mesh.vertices), mesh.vertexCount * sizeof(ShapeDescriptor::cpu::float3));
    if(mesh.normals != nullptr) {
        appendToHash(checksum, reinterpret_cast<const char*>(mesh.normals), mesh.vertexCount * sizeof(ShapeDescriptor::cpu::float3));
    }
    if(mesh.vertexColours != nullptr) {
        appendToHash(checksum, reinterpret_cast<const char*>(mesh.vertexColours), mesh.vertexCount * sizeof(ShapeDescriptor::cpu::uchar4));
    }

    return checksum.final();
}