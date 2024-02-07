#include "FilteredMeshPair.h"

void ShapeBench::FilteredMeshPair::free() {
    ShapeDescriptor::free(originalMesh);
    ShapeDescriptor::free(filteredSampleMesh);
    ShapeDescriptor::free(filteredAdditiveNoise);
}
