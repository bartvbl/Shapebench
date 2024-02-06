#pragma once

#include <iostream>

namespace ShapeBench {
    inline void drawProgressBar(uint32_t completed, uint32_t total) {
        const int barSteps = 16;
        float progress = float(completed) / float(total);
        int stepsToDraw = int(barSteps * progress);
        std::cout << "[";
        for(int i = 0; i < stepsToDraw; i++) {
            std::cout << "=";
        }
        for(int i = 0; i < barSteps - stepsToDraw; i++) {
            std::cout << " ";
        }
        std::cout << "]";
    }
}
