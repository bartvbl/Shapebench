#pragma once

struct GeometryBuffer {
    unsigned int vaoID = -1;

    unsigned int indexCount = 0;

    unsigned int vertexBufferID = -1;
    unsigned int normalBufferID = -1;
    unsigned int textureBufferID = -1;
    unsigned int indexBufferID = -1;

    void destroy();
};