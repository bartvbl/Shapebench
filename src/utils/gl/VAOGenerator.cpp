#include <glad/gl.h>
#include <vector>
#include "VAOGenerator.h"

BufferObject generateVertexArray(
        ShapeDescriptor::cpu::float3 *vertices,
        ShapeDescriptor::cpu::float3 *normals,
        ShapeDescriptor::cpu::float3* colours,
        unsigned int vertexCount) {
    BufferObject buffer;

    glGenVertexArrays(1, &buffer.VAOID);
    glBindVertexArray(buffer.VAOID);

    glGenBuffers(1, &buffer.vertexBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, buffer.vertexBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 3, vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);
    glEnableVertexAttribArray(0);

    if (normals != nullptr) {
        glGenBuffers(1, &buffer.normalBufferID);
        glBindBuffer(GL_ARRAY_BUFFER, buffer.normalBufferID);
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 3, normals, GL_STATIC_DRAW);
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);
        glEnableVertexAttribArray(2);
    }

    glGenBuffers(1, &buffer.colourBufferID);
    glBindBuffer(GL_ARRAY_BUFFER, buffer.colourBufferID);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertexCount * 3, colours, GL_STATIC_DRAW);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 3, 0);
    glEnableVertexAttribArray(3);


    std::vector<unsigned int> indexBuffer;
    indexBuffer.resize(vertexCount);

    for(int i = 0; i < vertexCount; i++) {
        indexBuffer.at(i) = i;
    }

    glGenBuffers(1, &buffer.indexBufferID);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer.indexBufferID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indexBuffer.size(), indexBuffer.data(), GL_STATIC_DRAW);

    return buffer;
}
