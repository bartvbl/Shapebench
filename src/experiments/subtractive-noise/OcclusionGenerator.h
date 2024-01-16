#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include "json.hpp"
#include "benchmark-core/ComputedConfig.h"
#include "benchmark-core/Dataset.h"
#include "GLFW/glfw3.h"
#include "utils/gl/Shader.h"
#include "utils/gl/GeometryBuffer.h"

class OccludedSceneGenerator {
    GLFWwindow* window = nullptr;
    bool isDestroyed = false;
    Shader objectIDShader;
    Shader fullscreenQuadShader;
    uint32_t frameBufferID = 0;
    uint32_t renderBufferID = 0;
    uint32_t renderTextureID = 0;
    std::vector<unsigned char> localFramebufferCopy;
    GeometryBuffer screenQuadVAO;

public:
    OccludedSceneGenerator();
    ~OccludedSceneGenerator();
    ShapeDescriptor::cpu::Mesh computeOccludedMesh(const ShapeDescriptor::cpu::Mesh mesh, uint64_t seed);
    void init();
    void destroy();
};

