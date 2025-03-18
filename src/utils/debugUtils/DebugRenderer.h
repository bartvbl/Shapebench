#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include "utils/AreaCalculator.h"
#include <glm/glm.hpp>
#include <mutex>
#include <thread>

namespace ShapeBench {
    namespace internal {
        struct DebugRendererScene {
            std::string title;
            std::vector<ShapeDescriptor::cpu::float3> lineVertexPositions;
            std::vector<ShapeDescriptor::cpu::float3> lineVertexColours;
        };
    }

    class DebugRenderer {
        std::vector<internal::DebugRendererScene> scenes;
        int32_t currentShownScene = -1;
        int32_t currentScene = -1;
        std::mutex drawLock;
        bool closeRequested = false;
        bool destroyed = false;

        static ShapeDescriptor::cpu::double3 computeCylindricalCoordinate(ShapeBench::Cylinder cylinder, float distanceUp, float angle, float distanceOut);
        void runRenderThread();
        void createEmptyScene();
        std::thread renderThread;

    public:
        ~DebugRenderer();
        DebugRenderer();

        void drawLine(ShapeDescriptor::cpu::float3 start, ShapeDescriptor::cpu::float3 end, ShapeDescriptor::cpu::float3 colour = {0, 0, 0});
        void drawLine(ShapeDescriptor::cpu::double3 start, ShapeDescriptor::cpu::double3 end, ShapeDescriptor::cpu::float3 colour = {0, 0, 0});
        void drawLine(glm::vec3 start, glm::vec3 end, ShapeDescriptor::cpu::float3 colour = {0, 0, 0});
        void drawTriangle(glm::vec3 vertex0, glm::vec3 vertex1, glm::vec3 vertex2, ShapeDescriptor::cpu::float3 colour = {0, 0, 0});
        void drawTriangle(ShapeDescriptor::cpu::float2 vertex0, ShapeDescriptor::cpu::float2 vertex1, ShapeDescriptor::cpu::float2 vertex2, ShapeDescriptor::cpu::float3 colour = {0, 0, 0});
        void drawTriangle(ShapeDescriptor::cpu::double2 vertex0, ShapeDescriptor::cpu::double2 vertex1, ShapeDescriptor::cpu::double2 vertex2, ShapeDescriptor::cpu::float3 colour = {0, 0, 0});
        void drawCylinder(ShapeBench::Cylinder cylinder);
        void drawCylinder(glm::vec3 centrePoint, glm::vec3 normalisedDirection, float halfOfHeight, float radius);
        void drawSphere(ShapeDescriptor::cpu::float3 centre, float radius);
        void show(std::string title);
        void waitForClose();
    };
}