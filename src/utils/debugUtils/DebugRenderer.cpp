

#include "DebugRenderer.h"
#include <glad/gl.h>
#include <glm/glm.hpp>
#include <thread>
#include <utility>
#include "utils/gl/GLUtils.h"
#include "glm/ext/matrix_transform.hpp"
#include "utils/gl/GeometryBuffer.h"
#include "utils/gl/VAOGenerator.h"
#include "utils/gl/ShaderLoader.h"
#include "glm/ext/matrix_clip_space.hpp"
#include "glm/gtc/type_ptr.hpp"
#include <thread>
#include <functional>

namespace ShapeBench {
    namespace internal {
        static std::chrono::steady_clock::time_point _previousTimePoint = std::chrono::steady_clock::now();

        double getTimeDeltaSeconds() {
            // Determine the current time
            std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();

            // Calculate the number of nanoseconds that elapsed since the previous call to this function
            long long timeDelta = std::chrono::duration_cast<std::chrono::nanoseconds>(currentTime - _previousTimePoint).count();
            // Convert the time delta in nanoseconds to seconds
            double timeDeltaSeconds = (double)timeDelta / 1000000000.0;

            // Store the previously measured current time
            _previousTimePoint = currentTime;

            // Return the calculated time delta in seconds
            return timeDeltaSeconds;
        }

        void handleInput(GLFWwindow* window, glm::vec3& cameraPosition, glm::vec3& cameraRotation, float& movementSpeed, float& rotationSpeed, float& sceneScale) {
            float frameTime = getTimeDeltaSeconds();

            int windowWidth, windowHeight;
            glfwGetWindowSize(window, &windowWidth, &windowHeight);

            glViewport(0, 0, windowWidth, windowHeight);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // --- Handle Input --
            float deltaX = 0;
            float deltaY = 0;

            if(glfwJoystickPresent(GLFW_JOYSTICK_1)) {
                int axisCount;
                const float *axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &axisCount);
                int buttonCount;
                const unsigned char *buttons = glfwGetJoystickButtons(GLFW_JOYSTICK_1, &buttonCount);

                deltaX += axes[0];
                deltaY += -1 * axes[1];
                float deltaRotationX = std::abs(axes[3]) > 0.3 ? axes[3] : 0;
                float deltaRotationY = std::abs(axes[4]) > 0.3 ? -1.0f * axes[4] : 0;

                if (std::abs(deltaRotationX) < 0.15) { deltaRotationX = 0; }
                if (std::abs(deltaRotationY) < 0.15) { deltaRotationY = 0; }
                if (std::abs(deltaX) < 0.15) { deltaX = 0; }
                if (std::abs(deltaY) < 0.15) { deltaY = 0; }

                if (std::abs(deltaX) < 0.1) {
                    deltaX = 0;
                }

                if (std::abs(deltaY) < 0.1) {
                    deltaY = 0;
                }

                cameraPosition.y += (((axes[2] + 1.0f) / 2.0f) - ((axes[5] + 1.0f) / 2.0f)) * movementSpeed * frameTime;

                cameraRotation.x += deltaRotationX * rotationSpeed * frameTime;
                cameraRotation.y -= deltaRotationY * rotationSpeed * frameTime;

                if(buttonCount >= 2) {
                    if(buttons[0] == GLFW_PRESS) {
                        movementSpeed *= 1.04f;
                        movementSpeed = std::min<float>(movementSpeed, 1000);
                    }
                    if(buttons[1] == GLFW_PRESS) {
                        movementSpeed *= 1.0f / 1.04f;
                        movementSpeed = std::max<float>(movementSpeed, 0.001);
                    }
                }
            }

            if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
                cameraRotation.x -= rotationSpeed * frameTime;
            }
            if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
                cameraRotation.x += rotationSpeed * frameTime;
            }
            if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
                cameraRotation.y -= rotationSpeed * frameTime;
            }
            if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
                cameraRotation.y += rotationSpeed * frameTime;
            }

            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                deltaX -= 1;
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                deltaX += 1;
            }
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                deltaY += 1;
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                deltaY -= 1;
            }
            if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
                cameraPosition.y += movementSpeed * frameTime;
            }
            if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
                cameraPosition.y -= movementSpeed * frameTime;
            }

            float angleYRadiansForward = cameraRotation.x;
            float angleYRadiansSideways = (cameraRotation.x + float(M_PI / 2.0));

            cameraPosition.x -= deltaY * std::sin(angleYRadiansForward) * movementSpeed * frameTime;
            cameraPosition.z += deltaY * std::cos(angleYRadiansForward) * movementSpeed * frameTime;

            cameraPosition.x -= deltaX * std::sin(angleYRadiansSideways) * movementSpeed * frameTime;;
            cameraPosition.z += deltaX * std::cos(angleYRadiansSideways) * movementSpeed * frameTime;


        }
    }
}

void ShapeBench::DebugRenderer::drawLine(ShapeDescriptor::cpu::float3 start, ShapeDescriptor::cpu::float3 end, ShapeDescriptor::cpu::float3 colour) {
    scenes.at(scenes.size() - 1).lineVertexPositions.push_back(start);
    scenes.at(scenes.size() - 1).lineVertexPositions.push_back(end);
    scenes.at(scenes.size() - 1).lineVertexColours.push_back(colour);
    scenes.at(scenes.size() - 1).lineVertexColours.push_back(colour);
}

void ShapeBench::DebugRenderer::drawCylinder(ShapeBench::Cylinder cylinder) {
    const int stepCount = 60;
    cylinder.normalisedDirection = normalize(cylinder.normalisedDirection);
    ShapeDescriptor::cpu::double3 centreTop = cylinder.centrePoint + cylinder.halfOfHeight * cylinder.normalisedDirection;
    ShapeDescriptor::cpu::double3 centreBottom = cylinder.centrePoint - cylinder.halfOfHeight * cylinder.normalisedDirection;
    for(int i = 0; i < stepCount; i++) {
        float angle = (float(i) / float(stepCount)) * 2.0f * float(M_PI);
        float previousAngle = (float(i-1) / float(stepCount)) * 2.0f * float(M_PI);
        ShapeDescriptor::cpu::double3 previousBottom = computeCylindricalCoordinate(cylinder, -cylinder.halfOfHeight, previousAngle, cylinder.radius);
        ShapeDescriptor::cpu::double3 previousTop = computeCylindricalCoordinate(cylinder, cylinder.halfOfHeight, previousAngle, cylinder.radius);
        ShapeDescriptor::cpu::double3 bottom = computeCylindricalCoordinate(cylinder, -cylinder.halfOfHeight, angle, cylinder.radius);
        ShapeDescriptor::cpu::double3 top = computeCylindricalCoordinate(cylinder, cylinder.halfOfHeight, angle, cylinder.radius);
        drawLine(bottom, top);
        drawLine(previousBottom, bottom);
        drawLine(previousTop, top);
        drawLine(bottom, centreBottom);
        drawLine(top, centreTop);
    }
}

void ShapeBench::DebugRenderer::drawSphere(ShapeDescriptor::cpu::float3 centre, float radius) {

}

void sleepToFrameRate(double targetFrameRate) {
    static bool hasBeenInitialised = false;
    static std::chrono::time_point<std::chrono::steady_clock> previousCallTime;

    if(!hasBeenInitialised) {
        hasBeenInitialised = true;
        previousCallTime = std::chrono::steady_clock::now();
        return;
    }

    std::chrono::time_point<std::chrono::steady_clock> currentCallTime = std::chrono::steady_clock::now();
    std::chrono::duration timeDifference = currentCallTime - previousCallTime;
    previousCallTime = currentCallTime;

    double elapsedTimeSincePreviousCall = double(std::chrono::duration_cast<std::chrono::nanoseconds>(timeDifference).count()) / 1000000000.0;
    double minFrameTime = 1.0 / targetFrameRate;
    if(elapsedTimeSincePreviousCall < minFrameTime) {
        // Wait some time if the frame was computed too quickly
        double timeToWaitSeconds = minFrameTime - elapsedTimeSincePreviousCall;
        std::this_thread::sleep_for(std::chrono::nanoseconds(uint64_t(1000000000.0 * timeToWaitSeconds)));
    }
}

void ShapeBench::DebugRenderer::show(std::string sceneTitle) {
    std::unique_lock<std::mutex> lock(drawLock);
    scenes.at(scenes.size() - 1).title = std::move(sceneTitle);

    currentScene = int(scenes.size());

    // Create a new scene
    createEmptyScene();
}

ShapeDescriptor::cpu::double3
ShapeBench::DebugRenderer::computeCylindricalCoordinate(ShapeBench::Cylinder cylinder, float distanceUp, float angle,
                                                        float distanceOut) {
    const ShapeDescriptor::cpu::double3 xAxis = {1, 0, 0};
    const ShapeDescriptor::cpu::double3 zAxis = {0, 0, 1};

    ShapeDescriptor::cpu::double3 orthogonalDirection = xAxis;
    if(cylinder.normalisedDirection != zAxis) {
        orthogonalDirection = cross(zAxis, normalize(cylinder.normalisedDirection));
    }
    orthogonalDirection = normalize(orthogonalDirection);

    glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0), angle, glm::vec3(cylinder.normalisedDirection.x, cylinder.normalisedDirection.y, cylinder.normalisedDirection.z));
    glm::vec3 rotatedDirection = rotationMatrix * glm::vec4(orthogonalDirection.x, orthogonalDirection.y, orthogonalDirection.z, 1.0);
    rotatedDirection = glm::normalize(rotatedDirection);

    ShapeDescriptor::cpu::double3 convertedDirection = {rotatedDirection.x, rotatedDirection.y, rotatedDirection.z};
    ShapeDescriptor::cpu::double3 outPoint = cylinder.centrePoint + distanceUp * cylinder.normalisedDirection + distanceOut * convertedDirection;
    return outPoint;
}

void ShapeBench::DebugRenderer::drawLine(glm::vec3 start, glm::vec3 end, ShapeDescriptor::cpu::float3 colour) {
    ShapeDescriptor::cpu::float3 convertedStart = {start.x, start.y, start.z};
    ShapeDescriptor::cpu::float3 convertedEnd = {end.x, end.y, end.z};
    drawLine(convertedStart, convertedEnd, colour);
}

void ShapeBench::DebugRenderer::drawLine(ShapeDescriptor::cpu::double3 start, ShapeDescriptor::cpu::double3 end,
                                         ShapeDescriptor::cpu::float3 colour) {
    ShapeDescriptor::cpu::float3 convertedStart = {(float)start.x, (float)start.y, (float)start.z};
    ShapeDescriptor::cpu::float3 convertedEnd = {(float)end.x, (float)end.y, (float)end.z};
    drawLine(convertedStart, convertedEnd, colour);
}

void ShapeBench::DebugRenderer::drawCylinder(glm::vec3 centrePoint, glm::vec3 normalisedDirection, float halfOfHeight, float radius) {
    ShapeDescriptor::cpu::float3 convertedCentrePoint = {centrePoint.x, centrePoint.y, centrePoint.z};
    ShapeDescriptor::cpu::float3 convertedDirection = {normalisedDirection.x, normalisedDirection.y, normalisedDirection.z};
    drawCylinder(ShapeBench::Cylinder{toDouble3(convertedCentrePoint), toDouble3(convertedDirection), halfOfHeight, radius});
}

void ShapeBench::DebugRenderer::drawTriangle(glm::vec3 vertex0, glm::vec3 vertex1, glm::vec3 vertex2,
                                             ShapeDescriptor::cpu::float3 colour) {
    drawLine(vertex0, vertex1, colour);
    drawLine(vertex1, vertex2, colour);
    drawLine(vertex2, vertex0, colour);
}

void ShapeBench::DebugRenderer::drawTriangle(ShapeDescriptor::cpu::float2 vertex0, ShapeDescriptor::cpu::float2 vertex1,
                                             ShapeDescriptor::cpu::float2 vertex2,
                                             ShapeDescriptor::cpu::float3 colour) {
    glm::vec3 vertex0_glm = {vertex0.x, vertex0.y, 0};
    glm::vec3 vertex1_glm = {vertex1.x, vertex1.y, 0};
    glm::vec3 vertex2_glm = {vertex2.x, vertex2.y, 0};
    drawTriangle(vertex0_glm, vertex1_glm, vertex2_glm, colour);
}

void ShapeBench::DebugRenderer::drawTriangle(ShapeDescriptor::cpu::double2 vertex0, ShapeDescriptor::cpu::double2 vertex1,
                                             ShapeDescriptor::cpu::double2 vertex2, ShapeDescriptor::cpu::float3 colour) {
    glm::vec3 vertex0_glm = {vertex0.x, vertex0.y, 0};
    glm::vec3 vertex1_glm = {vertex1.x, vertex1.y, 0};
    glm::vec3 vertex2_glm = {vertex2.x, vertex2.y, 0};
    drawTriangle(vertex0_glm, vertex1_glm, vertex2_glm, colour);
}

ShapeBench::DebugRenderer::DebugRenderer() : renderThread(std::bind(&ShapeBench::DebugRenderer::runRenderThread, this)) {
    createEmptyScene();
}

void ShapeBench::DebugRenderer::runRenderThread() {
    std::cout << "Render thread started" << std::endl;
    GLFWwindow* window = ShapeBench::GLinitialise(1024, 768, "[debug renderer]");

    ShapeBench::GeometryBuffer linesBuffer;
    ShapeBench::Shader colourShader = loadShader("../res/shaders/", "objectIDShader");
    colourShader.use();

    glm::vec3 cameraPosition = {0, 0, 0};
    glm::vec3 cameraRotation = {0, 0, 0};
    float sceneScale = 1;
    bool wasNextButtonPressed = false;
    bool wasPreviousButtonPressed = false;


    glClearColor(0.5, 0.5, 0.5, 1.0);
    float movementSpeed = 10;
    float rotationSpeed = 1;
    while(!glfwWindowShouldClose(window) && !closeRequested) {
        glfwPollEvents();

        if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            closeRequested = true;
        }
        bool isNextButtonPressed = glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS;
        if(wasNextButtonPressed && !isNextButtonPressed) {
            currentScene = std::min<int>(currentScene + 1, scenes.size() - 1);
        }
        wasNextButtonPressed = isNextButtonPressed;
        bool isPreviousButtonPressed = glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS;
        if(wasPreviousButtonPressed && !isPreviousButtonPressed) {
            currentScene = std::max<int>(currentScene - 1, 0);
        }
        wasPreviousButtonPressed = isPreviousButtonPressed;

        {
            std::unique_lock<std::mutex> lock(drawLock);
            if(currentShownScene != currentScene) {
                std::cout << "Updating scene.." << std::endl;
                if(currentShownScene != -1) {
                    destroyVertexArray(linesBuffer);
                }
                linesBuffer = ShapeBench::generateVertexArray(scenes.at(currentScene).lineVertexPositions.data(),
                                                              nullptr,
                                                              scenes.at(currentScene).lineVertexColours.data(),
                                                              scenes.at(currentScene).lineVertexPositions.size());
                glfwSetWindowTitle(window, scenes.at(currentScene).title.c_str());
                currentShownScene = currentScene;
            }
        }



        int windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);
        internal::handleInput(window, cameraPosition, cameraRotation, movementSpeed, rotationSpeed, sceneScale);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        ShapeBench::printGLError(__FILE__, __LINE__);


        float nearPlaneDistance = 0.001;
        float farPlaneDistance = 100.0;
        float fovy = M_PI / 2; // 90 degrees

        glm::mat4 objectProjection = glm::perspective(fovy, (float) windowWidth / (float) windowHeight, nearPlaneDistance, farPlaneDistance);
        glm::mat4x4 view(1.0);

        view *= glm::rotate(glm::mat4(1.0), cameraRotation.z, glm::vec3(0, 0, 1));
        view *= glm::rotate(glm::mat4(1.0), cameraRotation.y, glm::vec3(1, 0, 0));
        view *= glm::rotate(glm::mat4(1.0), cameraRotation.x, glm::vec3(0, 1, 0));
        view *= glm::translate(glm::mat4(1.0), cameraPosition);
        view *= glm::scale(glm::mat4(1.0), glm::vec3(sceneScale, sceneScale, sceneScale));

        glm::mat4x4 projection = glm::perspective(1.57f, (float) windowWidth / (float) windowHeight, 0.001f, 10000.0f); //glm::ortho(-1.0, 1.0, -1.0, 1.0, .00, 10000.0);//
        glm::mat4x4 viewProjection = projection * view;

        glUniformMatrix4fv(16, 1, false, glm::value_ptr(viewProjection));
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glEnable(GL_CULL_FACE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        if(currentShownScene != -1) {
            glBindVertexArray(linesBuffer.vaoID);
            glDrawElements(GL_LINES, linesBuffer.indexCount, GL_UNSIGNED_INT, nullptr);
        }
        ShapeBench::printGLError(__FILE__, __LINE__);

        glfwSwapBuffers(window);
        sleepToFrameRate(90);

    }

    glfwDestroyWindow(window);
    glfwTerminate();
    std::cout << "Debug renderer terminated" << std::endl;
}

ShapeBench::DebugRenderer::~DebugRenderer() {
    closeRequested = true;
    // Wait for thread to clean up and exit
    if(!destroyed) {
        renderThread.join();
    }
}

void ShapeBench::DebugRenderer::createEmptyScene() {
    // Create an empty scene
    scenes.emplace_back();

    // Draw primary axes
    drawLine(glm::vec3{-1000, 0, 0}, {1000, 0, 0}, {1, 0, 0});
    drawLine(glm::vec3{0, -1000, 0}, {0, 1000, 0}, {0, 1, 0});
    drawLine(glm::vec3{0, 0, -1000}, {0, 0, 1000}, {0, 0, 1});
}

void ShapeBench::DebugRenderer::waitForClose() {
    if(!destroyed) {
        renderThread.join();
    }
    destroyed = true;
}







