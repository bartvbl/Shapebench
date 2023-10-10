#include "RepresentativeSet.h"
#include "shapeDescriptor/cpu/types/Mesh.h"
#include <shapeDescriptor/utilities/read/CompressedGeometryFile.h>

template<typename DescriptorMethod, typename DescriptorType>
ShapeDescriptor::gpu::array<DescriptorMethod> Shapebench::computeRepresentativeSet(const Dataset &dataset, uint32_t count, uint64_t randomSeed, float supportRadius) {
    std::vector<VertexInDataset> sampledVertices = dataset.sampleVertices(randomSeed, count);

    std::filesystem::path currentMeshPath = sampledVertices.at(0).meshFile;
    ShapeDescriptor::cpu::Mesh currentMesh = ShapeDescriptor::readMeshFromCompressedGeometryFile(currentMeshPath);

    return ShapeDescriptor::gpu::array<DescriptorMethod>();
}