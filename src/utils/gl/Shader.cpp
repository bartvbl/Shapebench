#include "glad/gl.h"
#include <iostream>
#include "Shader.h"

void ShapeBench::Shader::use() {
    glUseProgram(programID);
}

void ShapeBench::Shader::destroy() {
    glDeleteProgram(programID);
}

unsigned int ShapeBench::Shader::get() {
    return programID;
}

void ShapeBench::Shader::setUniform(unsigned int ID, float *matrix) {
    glUniformMatrix4fv(ID, 1, GL_FALSE, matrix);
}

void ShapeBench::Shader::setUniformMat3(unsigned int ID, float *matrix) {
    glUniformMatrix3fv(ID, 1, GL_FALSE, matrix);
}

void ShapeBench::Shader::setUniform(unsigned int ID, float x, float y, float z) {
    glUniform3f(ID, x, y, z);
}

void ShapeBench::Shader::setUniform(unsigned int ID, float x, float y, float z, float w) {
    glUniform4f(ID, x, y, z, w);
}