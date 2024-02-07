#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "filters/FilteredMeshPair.h"

namespace ShapeBench {
    struct AreaEstimate {
        float sampleObjectArea = 0;
        float clutterArea = 0;
        float totalArea = 0;
    };

    template<typename DescriptorMethod>
    AreaEstimate estimateAreaInSupportVolume(ShapeBench::FilteredMeshPair& meshes, ShapeDescriptor::OrientedPoint referencePoint, float supportRadius) {

    }
}

