#pragma once

#include <GLFW/glfw3.h>
#include <string>

namespace ShapeBench {
    inline void printGLError(const char* fileName, int lineNumber) {
        int errorID = glGetError();

        if(errorID != GL_NO_ERROR) {
            std::string errorString;

            switch(errorID) {
                case GL_INVALID_ENUM:
                    errorString = "GL_INVALID_ENUM";
                    break;
                case GL_INVALID_OPERATION:
                    errorString = "GL_INVALID_OPERATION";
                    break;
                case GL_INVALID_FRAMEBUFFER_OPERATION:
                    errorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
                    break;
                case GL_OUT_OF_MEMORY:
                    errorString = "GL_OUT_OF_MEMORY";
                    break;
                case GL_STACK_UNDERFLOW:
                    errorString = "GL_STACK_UNDERFLOW";
                    break;
                case GL_STACK_OVERFLOW:
                    errorString = "GL_STACK_OVERFLOW";
                    break;
                default:
                    errorString = "[Unknown error ID]";
                    break;
            }

            fprintf(stderr, "An OpenGL error occurred in file %s:%i (%i): %s.\n",
                    fileName, lineNumber, errorID, errorString.c_str());
        }
    }

    GLFWwindow* GLinitialise(uint32_t windowWidth, uint32_t windowHeight, std::string windowTitle = "ShapeBench");
}
