#include <iostream>
#include "glad/gl.h"
#include "ShaderLoader.h"
#include "GLUtils.h"

struct ShaderSource {
    std::string vertexShaderSource;
    std::string fragmentShaderSource;
    std::string geometryShaderSource;
    std::string computeShaderSource;
    std::string tesselationControlShaderSource;
    std::string tesselationEvaluationShaderSource;
};

enum class ShaderType {
    VERTEX_SHADER,
    FRAGMENT_SHADER,
    GEOMETRY_SHADER,
    COMPUTE_SHADER,
    MESH_SHADER,
    TESSELATION_CONTROL_SHADER,
    TESSELATION_EVALUATION_SHADER
};

unsigned int createShaderObject(ShaderType type) {
    switch (type) {
        case ShaderType::VERTEX_SHADER:
            return glCreateShader(GL_VERTEX_SHADER);
        case ShaderType::FRAGMENT_SHADER:
            return glCreateShader(GL_FRAGMENT_SHADER);
        case ShaderType::GEOMETRY_SHADER:
            return glCreateShader(GL_GEOMETRY_SHADER);
        case ShaderType::COMPUTE_SHADER:
            return glCreateShader(GL_COMPUTE_SHADER);
        case ShaderType::MESH_SHADER:
            throw std::runtime_error("mesh shaders are currently not supported");
        case ShaderType::TESSELATION_CONTROL_SHADER:
            return glCreateShader(GL_TESS_CONTROL_SHADER);
        case ShaderType::TESSELATION_EVALUATION_SHADER:
            return glCreateShader(GL_TESS_EVALUATION_SHADER);
        default:
            throw std::runtime_error("Attempted to create shader of unsupported type!");
    }
}

void loadSingleShader(unsigned int shaderProgramID, const std::string &shaderSource, ShaderType type)
{
    // Create shader object
    unsigned int shaderID = createShaderObject(type);
    const char* shaderSourceCString = shaderSource.c_str();
    glShaderSource(shaderID, 1, &shaderSourceCString, nullptr);
    glCompileShader(shaderID);

    // Display errors
    int compileStatus = 0;
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &compileStatus);
    if (!compileStatus)
    {
        int messageLength = 0;
        glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &messageLength);
        std::unique_ptr<char[]> buffer(new char[messageLength]);
        glGetShaderInfoLog(shaderID, messageLength, nullptr, buffer.get());
        std::cout << "SHADER COMPILATION FAILED\nShader source:\n" + shaderSource + "\n\nCompilation message:" + std::string(buffer.get()) + "\n" << std::flush;
    }

    assert(compileStatus);

    // Attach shader and free allocated memory
    glAttachShader(shaderProgramID, shaderID);
    glDeleteShader(shaderID);
}

void linkProgram(unsigned int programID) {
    glLinkProgram(programID);

    int linkStatus = 0;
    glGetProgramiv(programID, GL_LINK_STATUS, &linkStatus);
    if (!linkStatus) {
        int errorMessageLength = 0;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &errorMessageLength);
        std::unique_ptr<char[]> buffer(new char[errorMessageLength]);
        glGetProgramInfoLog(programID, errorMessageLength, nullptr, buffer.get());
        std::cerr << "Linking of program failed. Reason:\n" + std::string(buffer.get()) + "\n" << std::flush;
    }

    assert(linkStatus);
    ShapeBench::printGLError(__FILE__, __LINE__);
}

void validateProgram(unsigned int programID) {
    // Validate linked shader program
    glValidateProgram(programID);

    // Display errors
    int programStatus = 0;
    glGetProgramiv(programID, GL_VALIDATE_STATUS, &programStatus);
    if (!programStatus)
    {
        int errorMessageLength = 0;
        glGetProgramiv(programID, GL_INFO_LOG_LENGTH, &errorMessageLength);
        std::unique_ptr<char[]> buffer(new char[errorMessageLength]);
        glGetProgramInfoLog(programID, errorMessageLength, nullptr, buffer.get());
        std::cerr << "Program failed to validate. Reason:\n" + std::string(buffer.get()) + "\n" << std::flush;
        assert(false);
    }
    ShapeBench::printGLError(__FILE__, __LINE__);
}

ShapeBench::Shader createShader(ShaderSource* source) {
    unsigned int programID = glCreateProgram();
    ShapeBench::Shader shader(programID);
    uint32_t attachedCount = 0;

    if(!source->vertexShaderSource.empty()) {
        loadSingleShader(shader.get(), source->vertexShaderSource, ShaderType::VERTEX_SHADER);
    }
    if(!source->fragmentShaderSource.empty()) {
        loadSingleShader(shader.get(), source->fragmentShaderSource, ShaderType::FRAGMENT_SHADER);
    }
    if(!source->geometryShaderSource.empty()) {
        loadSingleShader(shader.get(), source->geometryShaderSource, ShaderType::GEOMETRY_SHADER);
    }
    if(!source->computeShaderSource.empty()) {
        loadSingleShader(shader.get(), source->computeShaderSource, ShaderType::COMPUTE_SHADER);
    }
    if(!source->tesselationControlShaderSource.empty()) {
        loadSingleShader(shader.get(), source->tesselationControlShaderSource, ShaderType::TESSELATION_CONTROL_SHADER);
    }
    if(!source->tesselationEvaluationShaderSource.empty()) {
        loadSingleShader(shader.get(), source->tesselationEvaluationShaderSource, ShaderType::TESSELATION_EVALUATION_SHADER);
    }

    linkProgram(shader.get());
    validateProgram(shader.get());
    return shader;
}

std::string tryFileLoad(const std::filesystem::path& file) {
    if(std::filesystem::exists(file)) {
        std::fstream inputStream(file);
        return std::string(std::istreambuf_iterator<char>(inputStream),(std::istreambuf_iterator<char>()));
    } else {
        return "";
    }
}

bool allSourcesEmpty(ShaderSource &sources) {
    return sources.computeShaderSource.empty()
        && sources.fragmentShaderSource.empty()
        && sources.geometryShaderSource.empty()
        && sources.tesselationControlShaderSource.empty()
        && sources.tesselationEvaluationShaderSource.empty()
        && sources.vertexShaderSource.empty();
}

ShapeBench::Shader ShapeBench::loadShader(const std::filesystem::path& directory, const std::string& fileName) {
    ShaderSource sources;
    sources.vertexShaderSource = tryFileLoad(directory / (fileName + ".vert"));
    sources.fragmentShaderSource = tryFileLoad(directory / (fileName + ".frag"));
    sources.geometryShaderSource = tryFileLoad(directory / (fileName + ".geom"));
    sources.computeShaderSource = tryFileLoad(directory / (fileName + ".comp"));
    sources.tesselationControlShaderSource = tryFileLoad(directory / (fileName + ".tcs"));
    sources.tesselationEvaluationShaderSource = tryFileLoad(directory / (fileName + ".tes"));
    if(allSourcesEmpty(sources)) {
        throw std::runtime_error("No shader sources found for shader " + fileName + " in directory " + directory.string());
    }
    return createShader(&sources);
}



