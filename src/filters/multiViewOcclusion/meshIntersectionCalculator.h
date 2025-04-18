#pragma once
#include <shapeDescriptor/shapeDescriptor.h>
#include <glm/glm.hpp>
#include "filters/FilteredMeshPair.h"

namespace ShapeBench {
    ShapeDescriptor::cpu::Mesh computeCommonTrianglesMesh(const ShapeBench::FilteredMeshPair& model,
                                                          const ShapeBench::FilteredMeshPair& scene);

    void transformMesh(ShapeDescriptor::cpu::Mesh &mesh, glm::mat4 vertexTransformation, glm::mat3 normalTransformation);
    void transformMesh(ShapeBench::FilteredMeshPair &mesh, glm::mat4 vertexTransformation, glm::mat3 normalTransformation);

    namespace internal {
        inline glm::vec3 toGLM(ShapeDescriptor::cpu::float3 in) {
            return {in.x, in.y, in.z};
        }

        inline ShapeDescriptor::cpu::float3 fromGLM(glm::vec3 in) {
            return {in.x, in.y, in.z};
        }
    }
}