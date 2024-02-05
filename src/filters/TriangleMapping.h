#pragma once

#include <vector>
#include <cstdint>

namespace Shapebench {
    enum class TriangleState {
        UNCHANGED, ALTERED, DELETED
    };

    struct SingleTriangleInfo {
        TriangleState state = TriangleState::UNCHANGED;
        uint32_t indexInModifiedMesh = 0;
    };

    struct TriangleMapping {
        std::vector<SingleTriangleInfo> mapping;
    };
}