#pragma once

#include <shapeDescriptor/shapeDescriptor.h>
#include <nlohmann/json.hpp>
#include "Method.h"
#include "types/IntersectingAreaParameters.h"
#include "types/IntersectingAreaEstimationStrategy.h"
#include "methods/pythonadapter/PythonAdapter.h"
#include "methods/pythonadapter/PythonDescriptor.h"

namespace ShapeBench {
    // This template allows you to add a new method that the benchmark can test.
    // Before you start, you should create a struct that contains a single descriptor
    // Next, create a copy of this file and rename it and the MethodName struct to something that represents your method
    // You should also replace all uses of DescriptorType in this template with the struct you defined
    // for representing a single descriptor.
    // Finally, to include this method in the benchmark, you must add a call to testMethod() in src/main.cu
    // with your MethodName as a template parameter. You will find a note in that file where to add this call.

    // As for this file, you might notice that it contains an object with a number of static methods.
    // The purpose of these methods is to be compiled into the benchmark code in specific places
    // The surrounding object allows them to be grouped such that they can be used as a template parameter



    struct MethodName {

        using DescriptorType = ShapeBench::PythonDescriptor<0>;

        // Called once when the benchmark starts up
        // Can be used to initialise some static variables
        // Does not need to do anything
        static void init(const nlohmann::json &config) {
            ShapeBench::PythonAdapter<DescriptorType>::init(getName(), config);
        }

        // Called when the benchmark is finished
        static void destroy() {
            ShapeBench::PythonAdapter<DescriptorType>::destroy();
        }

        // Used to tell the benchmark what input the benchmark should provide the method
        // The 3D surfaces that are represented by the descriptor method can either be a triangle mesh or point cloud
        // Either usesMeshInput() or usesPointCloudInput() must therefore return true
        static bool usesMeshInput() {
            return false;
        }

        static bool usesPointCloudInput() {
            return false;
        }



        // Depending on the settings above, implement the method that corresponds with the 3D surface input type
        // you selected above (triangle mesh or point cloud), and whether your method runs on the CPU or GPU
        // Do not delete the other ones


        // Triangle mesh, runs on the CPU
        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::Mesh &mesh,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<DescriptorType> descriptors = ShapeBench::PythonAdapter<DescriptorType>::computeDescriptors(mesh, descriptorOrigins, config, supportRadii, randomSeed);
            return descriptors;
        }

        // Point cloud, runs on the CPU
        static ShapeDescriptor::cpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::cpu::PointCloud &cloud,
                const ShapeDescriptor::cpu::array<ShapeDescriptor::OrientedPoint> &descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            ShapeDescriptor::cpu::array<DescriptorType> descriptors = ShapeBench::PythonAdapter<DescriptorType>::computeDescriptors(cloud, descriptorOrigins, config, supportRadii, randomSeed);
            return descriptors;
        }

        // The distance function that allows two descriptors to be compared
        // IMPORTANT: THE BENCHMARK ASSUMES LOWER DISTANCE IS BETTER
        // The only other requirement of the produced distances is that it must be possible to compare distances
        // produced by this function, where it may be assumed that 'otherDescriptor' is unchanged.
        // The earlyExitThreshold sets an upper threshold that the returned distance is expected to be
        // If while computing the distance it is detected that the total distance exceeds the specified threshold,
        // the distance computation may be ended early, and any distance greater than the threshold may be returned.
        static __inline__ float computeDescriptorDistance(
                const DescriptorType& descriptor,
                const DescriptorType& otherDescriptor,
                float earlyExitThreshold) {
            Method::throwUnimplementedException();

            // The benchmark provides the following functions that are potentially useful
            // Note: these functions assume that your descriptor ONLY contains
            // ShapeBench::computeEuclideanDistance(const DescriptorType& descriptor1, const DescriptorType& descriptor2, float earlyExitThreshold);
            // ShapeBench::computePearsonCorrelation(const DescriptorType& descriptor1, const DescriptorType& descriptor2)

            return 0;
        }



        // The benchmark needs to be able to estimate the amount of area of a mesh that is contained within the
        // support volume of your descriptor. That depends on the shape the support volume has.
        // - Cylindrical or spherical support volume: use the FAST_SPHERICAL or FAST_CYLINDRICAL enum values.
        //   The benchmark will handle everything for you in these cases.
        // - For a little more control over how area is computed, you can return the CUSTOM enum here instead.
        //   This requires that you implement the computeIntersectingAreaCustom() function below.
        // - For complex support volumes, a monte carlo based estimation method counting surface point samples
        //   intersecting the support volume can be used. Return in that case the SLOW_MONTE_CARLO_ESTIMATION enum
        //   Doing so requires that you implement the isPointInSupportVolume() function below.
        static ShapeBench::IntersectingAreaEstimationStrategy getIntersectingAreaEstimationStrategy() {
            return ShapeBench::IntersectingAreaEstimationStrategy::FAST_SPHERICAL;
        }

        // ONLY needed when getIntersectingAreaEstimationStrategy() returns the Monte Carlo based estimation method
        // Leave as-is otherwise
        // Returns true if the given samplePoint lies within the support volume of the descriptor that is computed
        // for the surface point descriptorOrigin, and support radius supportRadius.
        static bool isPointInSupportVolume(float supportRadius, ShapeDescriptor::OrientedPoint descriptorOrigin, ShapeDescriptor::cpu::float3 samplePoint) {
            Method::throwUnimplementedException();

            // Functions provided by the benchmark that may be useful here:
            // ShapeBench::isPointInCylindricalVolume(ShapeDescriptor::OrientedPoint referencePoint,
            //                                        float cylinderRadius, float cylinderHeight,
            //                                        ShapeDescriptor::cpu::float3 point)
            // ShapeBench::isPointInSphericalVolume(ShapeDescriptor::OrientedPoint referencePoint,
            //                                      float supportRadius,
            //                                      ShapeDescriptor::cpu::float3 point)

            return false;
        }

        // ONLY needed when getIntersectingAreaEstimationStrategy() indicates a custom method will be used
        // Leave as-is otherwise
        // Must return the area of the provided mesh that intersects with the support volume of the descriptor computed
        // for the given descriptorOrigin and supportRadius.
        static double computeIntersectingAreaCustom(const ShapeBench::IntersectionAreaParameters& parameters) {
            Method::throwUnimplementedException();

            // Functions provided by the benchmark that may be useful here:
            // ShapeBench::computeAreaInSphericalSupportVolume(const ShapeDescriptor::cpu::Mesh &mesh,
            //                                                 ShapeDescriptor::OrientedPoint referencePoint,
            //                                                 float supportRadius)
            // ShapeBench::computeAreaInCylindricalSupportVolume(const ShapeDescriptor::cpu::Mesh &mesh,
            //                                                   ShapeDescriptor::OrientedPoint referencePoint,
            //                                                   float supportRadius)

            return 0;
        }

        // A name that will be used to represent your method in file names and charts.
        // Must not contain any spaces
        static std::string getName() {
            return "METHODNAMEGOESHERE";
        }

        // This function should return a JSON object containing configuration values for your method
        // that are not already included in the benchmark's configuration file itself
        // These will be included in the output JSON file produced by the benchmark
        // This is useful from both a debugging perspective (ability to retrace steps when trying different parameter values),
        // and a replicability perspective (ability to validate results of different runs against each other)
        static nlohmann::json getMetadata() {
            nlohmann::json metadata = PythonAdapter<DescriptorType>::getMetadata();
            metadata["METHODConfigurationValue"] = 10;
            return metadata;
        }




        // ----- GPU functions, not supported by Python -----
        // They are mandatory for the benchmark, so we just put them here

        static bool hasGPUKernels() {
            return false;
        }

        __device__ static __inline__ float computeDescriptorDistanceGPU(
                const DescriptorType& descriptor,
                const DescriptorType& otherDescriptor,
                float earlyExitThreshold) {
            return 0;
        }

        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::Mesh &mesh,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> &device_descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }

        static ShapeDescriptor::gpu::array<DescriptorType> computeDescriptors(
                const ShapeDescriptor::gpu::PointCloud &cloud,
                const ShapeDescriptor::gpu::array<ShapeDescriptor::OrientedPoint> &device_descriptorOrigins,
                const nlohmann::json &config,
                const std::vector<float> &supportRadii,
                uint64_t randomSeed) {
            Method::throwIncompatibleException();
            return {};
        }
    };
}



