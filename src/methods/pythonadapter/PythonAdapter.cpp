

#include "PythonAdapter.h"

#include "pybind11/pybind11.h"
#include "fmt/format.h"
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11_json.hpp>
#include <pybind11/numpy.h>
#include <iostream>


namespace py = pybind11;
using namespace pybind11::literals;

static std::string currentActivePythonMethod;
static py::object pythonScope;
static py::object adapterModule;
static py::object SimpleNamespace;

const std::string pythonSourceFileName = "method";
const std::filesystem::path pythonMethodsDirectory = "../scripts/pythonmethods";

void ShapeBench::internal::initPython(const std::string &pythonMethodName, const nlohmann::json &config) {
    if(!currentActivePythonMethod.empty()) {
        throw std::runtime_error("Failed to initialise Python method \"" + pythonMethodName + "\": another method is already initialised: \"" + currentActivePythonMethod + "\"");
    }

    currentActivePythonMethod = pythonMethodName;
    py::initialize_interpreter(true, 0, nullptr, false);

    // Add python method to path
    py::module_ sys = py::module_::import("sys");
    py::list path = sys.attr("path");
    path.append((pythonMethodsDirectory / pythonMethodName).string());
    sys.attr("path") = path;
    pythonScope = py::module_::import("__main__").attr("__dict__");
    adapterModule = py::module_::import(pythonSourceFileName.c_str());
    SimpleNamespace = py::module_::import("types").attr("SimpleNamespace");

    py::object initMethodFunction = adapterModule.attr("initMethod");
    py::dict configDict = config;
    initMethodFunction(configDict);
}

void ShapeBench::internal::destroyPython() {
    py::object destroyMethodFunction = adapterModule.attr("destroyMethod");
    destroyMethodFunction();

    if(currentActivePythonMethod.empty()) {
        throw std::runtime_error("Failed to destroy Python method: no method is active");
    }

    py::finalize_interpreter();
}

nlohmann::json ShapeBench::internal::getPythonMetadata() {
    py::gil_scoped_acquire GIL;
    py::object getMethodMetadata = adapterModule.attr("getMethodMetadata");
    py::dict metadata = getMethodMetadata();
    nlohmann::json convertedMetadata = metadata;
    return convertedMetadata;
}

ShapeDescriptor::cpu::array<float> convertDescriptors(const py::array_t<float>& descriptors, size_t expectedNumberOfDescriptors, size_t expectedElementsPerDescriptor) {
    if(descriptors.ndim() != 2) {
        throw std::runtime_error("Python descriptor method did not return the right numer of dimensions. Expected: dimension 0 for descriptors, dimension 1 for contents of each descriptor.");
    }

    size_t descriptorCount = descriptors.shape()[0];
    uint32_t elementsPerDescriptor = descriptors.shape()[1];

    if(elementsPerDescriptor != expectedElementsPerDescriptor) {
        throw std::runtime_error(fmt::format("Python method returned {} elements per descriptor, "
                                             "but the descriptor structure defined on the C++ side has {} elements. "
                                             "These need to match. Please correct the one that is incorrect.",
                                             elementsPerDescriptor, expectedElementsPerDescriptor));
    }

    if(descriptorCount != expectedNumberOfDescriptors) {
        throw std::runtime_error(fmt::format("Python method returned {} descriptors, but {} were requested.", descriptorCount, expectedNumberOfDescriptors));
    }

    ShapeDescriptor::cpu::array<float> convertedDescriptors(elementsPerDescriptor * expectedNumberOfDescriptors);
    size_t nextBufferIndex = 0;
    for(uint32_t descriptorIndex = 0; descriptorIndex < expectedNumberOfDescriptors; descriptorIndex++) {
        for(uint32_t elementIndex = 0; elementIndex < elementsPerDescriptor; elementIndex++) {
            convertedDescriptors.content[nextBufferIndex] = descriptors.at(descriptorIndex, elementIndex);
            nextBufferIndex++;
        }
    }

    return convertedDescriptors;
}

ShapeDescriptor::cpu::array<float>
ShapeBench::internal::computePythonDescriptors(const ShapeDescriptor::cpu::Mesh &mesh,
                                               const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                                               const nlohmann::json &config,
                                               const std::vector<float> &supportRadii,
                                               uint64_t randomSeed,
                                               uint32_t entriesPerDescriptor) {
    py::gil_scoped_acquire GIL;

    py::array_t<float> vertexArray({mesh.vertexCount, (size_t)3});

    for(size_t vertexIndex = 0; vertexIndex < mesh.vertexCount; vertexIndex++) {
        vertexArray.mutable_at(vertexIndex, 0) = mesh.vertices[vertexIndex].x;
        vertexArray.mutable_at(vertexIndex, 1) = mesh.vertices[vertexIndex].y;
        vertexArray.mutable_at(vertexIndex, 2) = mesh.vertices[vertexIndex].z;
    }

    py::array_t<float> normalArray({mesh.vertexCount, (size_t)3});

    for(size_t vertexIndex = 0; vertexIndex < mesh.vertexCount; vertexIndex++) {
        normalArray.mutable_at(vertexIndex, 0) = mesh.normals[vertexIndex].x;
        normalArray.mutable_at(vertexIndex, 1) = mesh.normals[vertexIndex].y;
        normalArray.mutable_at(vertexIndex, 2) = mesh.normals[vertexIndex].z;
    }

    py::array_t<float> descriptorOriginArray({descriptorOrigins.length, (size_t) 2, (size_t) 3});

    for(size_t originIndex = 0; originIndex < descriptorOrigins.length; originIndex++) {
        descriptorOriginArray.mutable_at(originIndex, 0, 0) = descriptorOrigins.content[originIndex].vertex.x;
        descriptorOriginArray.mutable_at(originIndex, 0, 1) = descriptorOrigins.content[originIndex].vertex.y;
        descriptorOriginArray.mutable_at(originIndex, 0, 2) = descriptorOrigins.content[originIndex].vertex.z;
        descriptorOriginArray.mutable_at(originIndex, 1, 0) = descriptorOrigins.content[originIndex].normal.x;
        descriptorOriginArray.mutable_at(originIndex, 1, 1) = descriptorOrigins.content[originIndex].normal.y;
        descriptorOriginArray.mutable_at(originIndex, 1, 2) = descriptorOrigins.content[originIndex].normal.z;
    }

    py::dict configurationDict = config;

    py::object meshObject = SimpleNamespace("vertices"_a=vertexArray, "normals"_a=normalArray);

    py::object computeDescriptors = adapterModule.attr("computeMeshDescriptors");
    py::array_t<float> descriptors;
    try {
        descriptors = computeDescriptors(meshObject, descriptorOriginArray, configurationDict, supportRadii, randomSeed);
    } catch (py::error_already_set &e) {
        py::object formatTraceBack = py::module_::import("traceback").attr("format_tb");
        std::string traceBack = py::str(formatTraceBack(e.trace()));

        throw std::runtime_error("Python method threw an exception during its execution.\n    Error: " + std::string(e.what()) + "\nTraceback:\n" + traceBack);
    }
    return convertDescriptors(descriptors, descriptorOrigins.length, entriesPerDescriptor);
}

ShapeDescriptor::cpu::array<float>
ShapeBench::internal::computePythonDescriptors(const ShapeDescriptor::cpu::PointCloud &cloud,
                                               const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                                               const nlohmann::json &config,
                                               const std::vector<float> &supportRadii,
                                               uint64_t randomSeed,
                                               uint32_t entriesPerDescriptor) {
    py::gil_scoped_acquire GIL;

    py::array_t<float> vertexArray({cloud.pointCount, (size_t)3});

    for(size_t vertexIndex = 0; vertexIndex < cloud.pointCount; vertexIndex++) {
        vertexArray.mutable_at(vertexIndex, 0) = cloud.vertices[vertexIndex].x;
        vertexArray.mutable_at(vertexIndex, 1) = cloud.vertices[vertexIndex].y;
        vertexArray.mutable_at(vertexIndex, 2) = cloud.vertices[vertexIndex].z;
    }

    py::array_t<float> normalArray({cloud.pointCount, (size_t)3});

    for(size_t vertexIndex = 0; vertexIndex < cloud.pointCount; vertexIndex++) {
        normalArray.mutable_at(vertexIndex, 0) = cloud.normals[vertexIndex].x;
        normalArray.mutable_at(vertexIndex, 1) = cloud.normals[vertexIndex].y;
        normalArray.mutable_at(vertexIndex, 2) = cloud.normals[vertexIndex].z;
    }

    py::array_t<float> descriptorOriginArray({descriptorOrigins.length, (size_t) 2, (size_t) 3});

    for(size_t originIndex = 0; originIndex < descriptorOrigins.length; originIndex++) {
        descriptorOriginArray.mutable_at(originIndex, 0, 0) = descriptorOrigins.content[originIndex].vertex.x;
        descriptorOriginArray.mutable_at(originIndex, 0, 1) = descriptorOrigins.content[originIndex].vertex.y;
        descriptorOriginArray.mutable_at(originIndex, 0, 2) = descriptorOrigins.content[originIndex].vertex.z;
        descriptorOriginArray.mutable_at(originIndex, 1, 0) = descriptorOrigins.content[originIndex].normal.x;
        descriptorOriginArray.mutable_at(originIndex, 1, 1) = descriptorOrigins.content[originIndex].normal.y;
        descriptorOriginArray.mutable_at(originIndex, 1, 2) = descriptorOrigins.content[originIndex].normal.z;
    }

    py::dict configurationDict = config;

    py::object cloudObject = SimpleNamespace("vertices"_a=vertexArray, "normals"_a=normalArray);

    py::object computeDescriptors = adapterModule.attr("computePointCloudDescriptors");
    py::array_t<float> descriptors;
    try {
        descriptors = computeDescriptors(cloudObject, descriptorOriginArray, configurationDict, supportRadii, randomSeed);
    } catch (py::error_already_set &e) {
        py::object formatTraceBack = py::module_::import("traceback").attr("format_tb");
        std::string traceBack = py::str(formatTraceBack(e.trace()));

        throw std::runtime_error("Python method threw an exception during its execution.\n    Error: " + std::string(e.what()) + "\nTraceback:\n" + traceBack);
    }

    return convertDescriptors(descriptors, descriptorOrigins.length, entriesPerDescriptor);
}
