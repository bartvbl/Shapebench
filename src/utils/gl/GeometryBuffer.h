#pragma once

namespace ShapeBench {
    struct GeometryBuffer {
        unsigned int vaoID = 0xFFFFFFFF;

        unsigned int indexCount = 0;

        unsigned int vertexBufferID = 0xFFFFFFFF;
        unsigned int normalBufferID = 0xFFFFFFFF;
        unsigned int textureBufferID = 0xFFFFFFFF;
        unsigned int colourBufferID = 0xFFFFFFFF;
        unsigned int indexBufferID = 0xFFFFFFFF;

        void destroy();
    };
}
