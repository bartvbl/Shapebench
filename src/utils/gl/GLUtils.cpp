#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "glad/gl.h"
#include "GLUtils.h"




static void glfwErrorCallback(int error, const char *description)
{
    fprintf(stderr, "GLFW returned an error:\n\t%s (%i)\n", description, error);
}

GLFWwindow* ShapeBench::GLinitialise(uint32_t windowWidth, uint32_t windowHeight, std::string windowTitle)
{
    // Initialise GLFW
    if (!glfwInit())
    {
        fprintf(stderr, "Could not start GLFW\n");
        exit(EXIT_FAILURE);
    }

    // Set core window options (adjust version numbers if h github
    // needed)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Enable the GLFW runtime error callback function defined previously.
    glfwSetErrorCallback(glfwErrorCallback);

    // Set additional window options
    glfwWindowHint(GLFW_RESIZABLE, true);
    glfwWindowHint(GLFW_SAMPLES, 1);  // MSAA

    const GLFWvidmode * mode = glfwGetVideoMode(glfwGetPrimaryMonitor());

    // Create window using GLFW
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, windowTitle.c_str(), nullptr, nullptr);

    // Ensure the window is set up correctly
    if (!window)
    {
        fprintf(stderr, "Could not open GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Let the window be the current OpenGL context and initialise glad
    glfwMakeContextCurrent(window);

    int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        printf("Failed to initialize OpenGL context\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    // Print various OpenGL information to stdout
    //printf("%s: %s\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER));
    //printf("GLFW\t %s\n", glfwGetVersionString());
    std::cout << "    Created OpenGL context: " << glGetString(GL_VERSION) << std::endl;
    //printf("GLSL\t %s\n\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

    glClearColor(0.3, 0.3, 0.3, 1.0);
    ShapeBench::printGLError(__FILE__, __LINE__);

    return window;
}