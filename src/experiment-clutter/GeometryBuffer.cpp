#include <glad/gl.h>
#include "GeometryBuffer.h"

void GeometryBuffer::destroy() {
    glDeleteVertexArrays(1, &vaoID);
    glDeleteBuffers(1, &vertexBufferID);
    glDeleteBuffers(1, &normalBufferID);
    glDeleteBuffers(1, &textureBufferID);
    glDeleteBuffers(1, &indexBufferID);
}
