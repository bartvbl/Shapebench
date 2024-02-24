#pragma once


namespace ShapeBench {
    class Shader {
    public:
        unsigned int programID = 0;
    public:
        Shader(unsigned int shaderProgramID) : programID(shaderProgramID) {}
        Shader() = default;
        void use();
        unsigned int get();
        void setUniformMat3(unsigned int ID, float *matrix);
        void setUniform(unsigned int ID, float* matrix);
        void setUniform(unsigned int ID, float x, float y, float z);
        void setUniform(unsigned int ID, float x, float y, float z, float w);
        void destroy();
    };
}
