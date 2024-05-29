#include "glad/gl.h"
#include "GeometryBuffer.h"
#include "GLUtils.h"

void ShapeBench::GeometryBuffer::destroy() {
    if(vaoID != 0xFFFFFFFF) {
        glDeleteVertexArrays(1, &vaoID);
        vaoID = 0xFFFFFFFF;
    }
    if(vertexBufferID != 0xFFFFFFFF) {
        glDeleteBuffers(1, &vertexBufferID);
        vertexBufferID = 0xFFFFFFFF;
    }
    if(normalBufferID != 0xFFFFFFFF) {
        glDeleteBuffers(1, &normalBufferID);
        normalBufferID = 0xFFFFFFFF;
    }
    if(textureBufferID != 0xFFFFFFFF) {
        glDeleteBuffers(1, &textureBufferID);
        textureBufferID = 0xFFFFFFFF;
    }
    if(colourBufferID != 0xFFFFFFFF) {
        glDeleteBuffers(1, &colourBufferID);
        colourBufferID = 0xFFFFFFFF;
    }
    if(indexBufferID != 0xFFFFFFFF) {
        glDeleteBuffers(1, &indexBufferID);
        indexBufferID = 0xFFFFFFFF;
    }
    ShapeBench::printGLError(__FILE__, __LINE__);
}
