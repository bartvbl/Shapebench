#pragma once

#include "shapeDescriptor/gpu/types/array.h"
#include "Dataset.h"

namespace Shapebench {
    template<typename DescriptorMethod, typename DescriptorType>
    ShapeDescriptor::gpu::array<DescriptorMethod> computeRepresentativeSet(const Dataset& dataset, uint32_t count, uint64_t randomSeed, float supportRadius);




}