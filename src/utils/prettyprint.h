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

    inline void printDuration(std::chrono::duration<uint64_t, std::nano> duration) {
        uint32_t timeInSeconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
        uint32_t timeInHours = timeInSeconds / 3600;
        timeInSeconds -= timeInHours * 3600;
        uint32_t timeInMinutes = timeInSeconds / 60;
        timeInSeconds -= timeInMinutes * 60;
        std::cout << timeInHours << ":" << timeInMinutes << ":" << timeInSeconds << std::endl;
    }

}
