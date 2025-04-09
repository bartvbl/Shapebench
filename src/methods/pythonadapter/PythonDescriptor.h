#pragma once

#include <stdint.h>

namespace ShapeBench {
    template<uint32_t numberOfEntries>
    struct PythonDescriptor {
        float contents[numberOfEntries];
    };
}