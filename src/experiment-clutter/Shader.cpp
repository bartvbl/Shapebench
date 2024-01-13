#include <glad/gl.h>
#include <iostream>
#include "Shader.h"

void Shader::use() {
    glUseProgram(programID);
}

void Shader::destroy() {
    glDeleteProgram(programID);
}

unsigned int Shader::get() {
    return programID;
}

void Shader::setUniform(unsigned int ID, float *matrix) {
    glUniformMatrix4fv(ID, 1, GL_FALSE, matrix);
}

void Shader::setUniform(unsigned int ID, float x, float y, float z, float w) {
    glUniform4f(ID, x, y, z, w);
}
