#include "additiveNoiseFilter.h"

#include <cstdio>
#include <stdarg.h>
#include <iostream>

#include "json.hpp"
#include "benchmarkCore/ComputedConfig.h"

#include "Jolt/Jolt.h"

#include "Jolt/RegisterTypes.h"
#include "Jolt/Core/Factory.h"
#include "Jolt/Core/TempAllocator.h"
#include "Jolt/Core/JobSystemThreadPool.h"
#include "Jolt/Physics/PhysicsSettings.h"
#include "Jolt/Physics/PhysicsSystem.h"
#include "Jolt/Physics/Collision/Shape/BoxShape.h"
#include "Jolt/Physics/Body/BodyCreationSettings.h"
#include "Jolt/Physics/Body/BodyActivationListener.h"
#include "Jolt/Physics/Collision/PhysicsMaterialSimple.h"
#include "Jolt/Physics/Collision/Shape/MeshShape.h"
#include "Jolt/Physics/Collision/Shape/ConvexHullShape.h"
#include "Jolt/Physics/Collision/Shape/CompoundShape.h"
#include "Jolt/Physics/Collision/Shape/StaticCompoundShape.h"
#include "OpenGLDebugRenderer.h"

#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtx/quaternion.hpp"
#include "glm/gtc/matrix_inverse.hpp"

#define ENABLE_VHACD_IMPLEMENTATION 1
#define VHACD_DISABLE_THREADING 0
#include "VHACD.h"
#include "Jolt/Geometry/ConvexHullBuilder.h"

static void TraceImpl(const char *inFMT, ...)
{
    va_list list;
    va_start(list, inFMT);
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), inFMT, list);
    va_end(list);
    std::cout << buffer << std::endl;
}

#ifdef JPH_ENABLE_ASSERTS
// Callback for asserts, connect this to your own assert handler if you have one
static bool AssertFailedImpl(const char *inExpression, const char *inMessage, const char *inFile, uint inLine)
{
    std::cout << inFile << ":" << inLine << ": (" << inExpression << ") " << (inMessage != nullptr? inMessage : "") << std::endl;
    return true;
};
#endif // JPH_ENABLE_ASSERTS

using namespace JPH::literals;

// Layer that objects can be in, determines which other objects it can collide with
// Typically you at least want to have 1 layer for moving bodies and 1 layer for static bodies, but you can have more
// layers if you want. E.g. you could have a layer for high detail collision (which is not used by the physics simulation
// but only if you do collision testing).
namespace Layers
{
    static constexpr JPH::ObjectLayer NON_MOVING = 0;
    static constexpr JPH::ObjectLayer MOVING = 1;
    static constexpr JPH::ObjectLayer NUM_LAYERS = 2;
};

/// Class that determines if two object layers can collide
class ObjectLayerPairFilterImpl : public JPH::ObjectLayerPairFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inObject1, JPH::ObjectLayer inObject2) const override
    {
        switch (inObject1)
        {
            case Layers::NON_MOVING:
                return inObject2 == Layers::MOVING; // Non moving only collides with moving
            case Layers::MOVING:
                return true; // Moving collides with everything
            default:
                JPH_ASSERT(false);
                return false;
        }
    }
};

// Each broadphase layer results in a separate bounding volume tree in the broad phase. You at least want to have
// a layer for non-moving and moving objects to avoid having to update a tree full of static objects every frame.
// You can have a 1-on-1 mapping between object layers and broadphase layers (like in this case) but if you have
// many object layers you'll be creating many broad phase trees, which is not efficient. If you want to fine tune
// your broadphase layers define JPH_TRACK_BROADPHASE_STATS and look at the stats reported on the TTY.
namespace BroadPhaseLayers
{
    static constexpr JPH::BroadPhaseLayer NON_MOVING(0);
    static constexpr JPH::BroadPhaseLayer MOVING(1);
    static constexpr uint NUM_LAYERS(2);
};

// BroadPhaseLayerInterface implementation
// This defines a mapping between object and broadphase layers.
class BPLayerInterfaceImpl final : public JPH::BroadPhaseLayerInterface
{
public:
    BPLayerInterfaceImpl()
    {
        // Create a mapping table from object to broad phase layer
        mObjectToBroadPhase[Layers::NON_MOVING] = BroadPhaseLayers::NON_MOVING;
        mObjectToBroadPhase[Layers::MOVING] = BroadPhaseLayers::MOVING;
    }

    virtual uint GetNumBroadPhaseLayers() const override
    {
        return BroadPhaseLayers::NUM_LAYERS;
    }

    virtual JPH::BroadPhaseLayer GetBroadPhaseLayer(JPH::ObjectLayer inLayer) const override
    {
        JPH_ASSERT(inLayer < Layers::NUM_LAYERS);
        return mObjectToBroadPhase[inLayer];
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    virtual const char* GetBroadPhaseLayerName(JPH::BroadPhaseLayer inLayer) const override
    {
        switch ((JPH::BroadPhaseLayer::Type)inLayer)
        {
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::NON_MOVING:
                return "NON_MOVING";
            case (JPH::BroadPhaseLayer::Type)BroadPhaseLayers::MOVING:
                return "MOVING";
            default: JPH_ASSERT(false); return "INVALID";
        }
    }
#endif // JPH_EXTERNAL_PROFILE || JPH_PROFILE_ENABLED

private:
    JPH::BroadPhaseLayer mObjectToBroadPhase[Layers::NUM_LAYERS];
};

/// Class that determines if an object layer can collide with a broadphase layer
class ObjectVsBroadPhaseLayerFilterImpl : public JPH::ObjectVsBroadPhaseLayerFilter
{
public:
    virtual bool ShouldCollide(JPH::ObjectLayer inLayer1, JPH::BroadPhaseLayer inLayer2) const override
    {
        switch (inLayer1)
        {
            case Layers::NON_MOVING:
                return inLayer2 == BroadPhaseLayers::MOVING;
            case Layers::MOVING:
                return true;
            default:
                JPH_ASSERT(false);
                return false;
        }
    }
};

inline JPH::StaticCompoundShapeSettings* convertMeshToConvexHulls(const ShapeDescriptor::cpu::Mesh& mesh, const ShapeBench::AdditiveNoiseFilterSettings& settings) {
    //std::cout << "Computing hulls for mesh with " << mesh.vertexCount << " vertices" << std::endl;
    VHACD::IVHACD *subdivider = VHACD::CreateVHACD();

    std::vector<double> meshVertices(3 * mesh.vertexCount);
    for(uint32_t i = 0; i < mesh.vertexCount; i++) {
        meshVertices.at(3 * i + 0) = mesh.vertices[i].x;
        meshVertices.at(3 * i + 1) = mesh.vertices[i].y;
        meshVertices.at(3 * i + 2) = mesh.vertices[i].z;
    }

    std::vector<uint32_t> indices(mesh.vertexCount);
    for(uint32_t i = 0; i < mesh.vertexCount; i++) {
        indices.at(i) = i;
    }

    VHACD::IVHACD::Parameters parameters;
    parameters.m_maxConvexHulls = settings.maxConvexHulls;//64;
    parameters.m_resolution = settings.convexHullGenerationResolution;//400000;
    parameters.m_maxRecursionDepth = settings.convexHullGenerationRecursionDepth;//64; // max allowed by the library
    parameters.m_maxNumVerticesPerCH = settings.convexHullGenerationMaxVerticesPerHull;//256; // Jolt physics limitation

    subdivider->Compute(meshVertices.data(), mesh.vertexCount, indices.data(), mesh.vertexCount / 3, parameters);

    assert(subdivider->GetNConvexHulls() > 0);

    JPH::StaticCompoundShapeSettings* convexHullContainer = new JPH::StaticCompoundShapeSettings();
    std::vector<JPH::Vec3> hullVertices;

    for(uint32_t i = 0; i < subdivider->GetNConvexHulls(); i++) {

        hullVertices.clear();
        VHACD::IVHACD::ConvexHull hull;
        subdivider->GetConvexHull(i, hull);
        hullVertices.reserve(hull.m_points.size());
        for(uint32_t j = 0; j < hull.m_points.size(); j++) {
            VHACD::Vertex vertex = hull.m_points.at(j);
            JPH::Vec3 converted(vertex.mX, vertex.mY, vertex.mZ);
            hullVertices.push_back(converted);
        }
        JPH::ConvexHullShapeSettings* convexShape = new JPH::ConvexHullShapeSettings(hullVertices.data(), hullVertices.size());

        const char *error = nullptr;
        JPH::ConvexHullBuilder builder(convexShape->mPoints);
        builder.Initialize(JPH::ConvexHullShape::cMaxPointsInHull, convexShape->mHullTolerance, error);
        JPH::Vec3 centerOfMass;
        float volume;
        builder.GetCenterOfMassAndVolume(centerOfMass, volume);
        if(volume == 0) {
            delete convexShape;
            continue;
        }

        convexHullContainer->AddShape(JPH::Vec3Arg(0, 0, 0), JPH::Quat::sIdentity(), convexShape, 0);

    }

    subdivider->Release();

    return convexHullContainer;
}

void moveMeshToOriginAndUnitSphere(ShapeDescriptor::cpu::Mesh &mesh, ShapeDescriptor::cpu::float3 sphereCentre, float sphereRadius) {
    float oneOverSphereRadius = 1.0f/sphereRadius;
    for(uint32_t i = 0; i < mesh.vertexCount; i++) {
        ShapeDescriptor::cpu::float3 vertex = mesh.vertices[i];
        vertex.x = (vertex.x - sphereCentre.x) * oneOverSphereRadius;
        vertex.y = (vertex.y - sphereCentre.y) * oneOverSphereRadius;
        vertex.z = (vertex.z - sphereCentre.z) * oneOverSphereRadius;
        mesh.vertices[i] = vertex;
    }
}

bool anyBodyActive(JPH::BodyInterface *bodyInterface, const std::vector<JPH::BodyID>& simulatedBodies) {
    for(const JPH::BodyID& body : simulatedBodies) {
        if(bodyInterface->IsActive(body)) {
            return true;
        }
    }
    return false;
}


void ShapeBench::initPhysics() {
    // Register allocation hook
    JPH::RegisterDefaultAllocator();

    // Install callbacks
    JPH::Trace = TraceImpl;
    JPH_IF_ENABLE_ASSERTS(JPH::AssertFailed = AssertFailedImpl;)

    // Create a factory
    JPH::Factory::sInstance = new JPH::Factory();

    // Register all Jolt physics types
    JPH::RegisterTypes();
}

std::vector<ShapeBench::Orientation> ShapeBench::runPhysicsSimulation(ShapeBench::AdditiveNoiseFilterSettings settings,
                                                                      const std::vector<ShapeDescriptor::cpu::Mesh>& meshes) {
    // We need a temp allocator for temporary allocations during the physics update. We're
    // pre-allocating 10 MB to avoid having to do allocations during the physics update.
    JPH::TempAllocatorImpl temp_allocator(settings.tempAllocatorSizeBytes);

    // We need a job system that will execute physics jobs on multiple threads. Typically
    // you would implement the JobSystem interface yourself and let Jolt Physics run on top
    // of your own job scheduler. JobSystemThreadPool is an example implementation.
    JPH::JobSystemThreadPool job_system(JPH::cMaxPhysicsJobs, JPH::cMaxPhysicsBarriers, std::thread::hardware_concurrency() - 1);

    std::vector<JPH::TriangleList> joltMeshes(meshes.size());
    std::vector<JPH::StaticCompoundShapeSettings*> meshHullReplacements(meshes.size());

    std::vector<bool> meshIncluded(meshes.size(), true);

#pragma omp parallel for
    for(uint32_t i = 0; i < meshes.size(); i++) {
        JPH::StaticCompoundShapeSettings* hullSettings = convertMeshToConvexHulls(meshes.at(i), settings);
        if(hullSettings->mSubShapes.size() > 0) {
            meshHullReplacements.at(i) = hullSettings;
        } else {
            // Mesh only has empty convex hull volumes and cannot be simulated. It is therefore excluded.
            if(i == 0) {
                throw std::runtime_error("Reference mesh has no convex hulls!");
            }
            meshIncluded.at(i) = false;
        }
    }

    const uint32_t cMaxBodies = 65536;

    // This determines how many mutexes to allocate to protect rigid bodies from concurrent access. Set it to 0 for the default settings.
    const uint32_t cNumBodyMutexes = 0;

    // This is the max amount of body pairs that can be queued at any time (the broad phase will detect overlapping
    // body pairs based on their bounding boxes and will insert them into a queue for the narrowphase). If you make this buffer
    // too small the queue will fill up and the broad phase jobs will start to do narrow phase work. This is slightly less efficient.
    const uint32_t cMaxBodyPairs = 65536;

    // This is the maximum size of the contact constraint buffer. If more contacts (collisions between bodies) are detected than this
    // number then these contacts will be ignored and bodies will start interpenetrating / fall through the world.
    // Note: This value is low because this is a simple test. For a real project use something in the order of 10240.
    const uint32_t cMaxContactConstraints = 10240;

    BPLayerInterfaceImpl broad_phase_layer_interface;
    ObjectVsBroadPhaseLayerFilterImpl object_vs_broadphase_layer_filter;
    ObjectLayerPairFilterImpl object_vs_object_layer_filter;

    // Now we can create the actual physics system.
    JPH::PhysicsSystem physics_system;
    physics_system.Init(cMaxBodies, cNumBodyMutexes, cMaxBodyPairs, cMaxContactConstraints, broad_phase_layer_interface, object_vs_broadphase_layer_filter, object_vs_object_layer_filter);

    // The main way to interact with the bodies in the physics system is through the body interface. There is a locking and a non-locking
    // variant of this. We're going to use the locking version (even though we're not planning to access bodies from multiple threads)
    JPH::BodyInterface &body_interface = physics_system.GetBodyInterface();

    // Next we can create a rigid body to serve as the floor, we make a large box
    // Create the settings for the collision volume (the shape).
    // Note that for simple shapes (like boxes) you can also directly construct a BoxShape.
    JPH::BoxShapeSettings floor_shape_settings(JPH::Vec3(100.0f, 1.0f, 100.0f));

    // Create the shape
    JPH::ShapeSettings::ShapeResult floor_shape_result = floor_shape_settings.Create();
    JPH::ShapeRefC floor_shape = floor_shape_result.Get(); // We don't expect an error here, but you can check floor_shape_result for HasError() / GetError()

    // Create the settings for the body itself. Note that here you can also set other properties like the restitution / friction.
    JPH::BodyCreationSettings floor_settings(floor_shape, JPH::RVec3(0.0_r, -1.0_r, 0.0_r), JPH::Quat::sIdentity(), JPH::EMotionType::Static, Layers::NON_MOVING);

    // Create the actual rigid body
    JPH::Body *floor = body_interface.CreateBody(floor_settings); // Note that if we run out of bodies this can return nullptr
    floor->SetFriction(settings.floorFriction);
    body_interface.AddBody(floor->GetID(), JPH::EActivation::DontActivate);

    // Adding sample objects to the scene
    std::vector<JPH::BodyID> simulatedBodies(meshes.size());
    for(uint32_t i = 0; i < meshes.size(); i++) {
        if(!meshIncluded.at(i)) {
            continue;
        }
        JPH::PhysicsMaterialList materials;
        materials.push_back(new JPH::PhysicsMaterialSimple("Default material", JPH::Color::sGetDistinctColor(i)));
        JPH::StaticCompoundShapeSettings* compoundSettings = meshHullReplacements.at(i);
        JPH::BodyID meshBodyID = body_interface.CreateAndAddBody(JPH::BodyCreationSettings(compoundSettings, JPH::RVec3(0, settings.initialObjectSeparation * float(i) + 1.0f, 0), JPH::Quat::sIdentity(), JPH::EMotionType::Dynamic, Layers::MOVING), JPH::EActivation::Activate);

        simulatedBodies.at(i) = meshBodyID;
    }

    const float cDeltaTime = 1.0f / float(settings.simulationFrameRate);

    ShapeBench::OpenGLDebugRenderer* renderer = nullptr;
    if(settings.enableDebugRenderer) {
        renderer = new ShapeBench::OpenGLDebugRenderer();
    }

    // Optional step: Before starting the physics simulation you can optimize the broad phase. This improves collision detection performance (it's pointless here because we only have 2 bodies).
    // You should definitely not call this every frame or when e.g. streaming in a new level section as it is an expensive operation.
    // Instead insert all new objects in batches instead of 1 at a time to keep the broad phase efficient.
    physics_system.OptimizeBroadPhase();

    uint32_t steps = 0;

    while ((steps < settings.simulationStepLimit) && anyBodyActive(&body_interface, simulatedBodies) || (settings.enableDebugRenderer && settings.runSimulationUntilManualExit && !renderer->windowShouldClose()))
    {
        steps++;

        JPH::RVec3 referenceObjectPosition = body_interface.GetCenterOfMassPosition(simulatedBodies.at(0));
        for(int i = 1; i < meshes.size(); i++) {
            if(!meshIncluded.at(i)) {
                continue;
            }
            JPH::RVec3 sampleObjectPosition = body_interface.GetCenterOfMassPosition(simulatedBodies.at(i));
            JPH::RVec3 deltaVector = referenceObjectPosition - sampleObjectPosition;
            deltaVector /= deltaVector.Length();
            JPH::RVec3 forceDirection = settings.objectAttractionForceMagnitude * deltaVector;

            // Only applying a force to the sample object will cause several objects to push the reference one indefinitely
            // It is therefore necessary to apply a separate force in the opposite direction for the simulation to converge
            body_interface.AddForce(simulatedBodies.at(i), forceDirection);
            body_interface.AddForce(simulatedBodies.at(0), -forceDirection);
        }

        // If you take larger steps than 1 / 60th of a second you need to do multiple collision steps in order to keep the simulation stable. Do 1 collision step per 1 / 60th of a second (round up).
        const int cCollisionSteps = 1;

        // Step the world
        physics_system.Update(cDeltaTime, cCollisionSteps, &temp_allocator, &job_system);

        if(settings.enableDebugRenderer) {
            JPH::BodyManager::DrawSettings settings;
            settings.mDrawShape = true;
            physics_system.DrawBodies(settings, renderer);
            renderer->nextFrame();
        }
    }

    if(settings.enableDebugRenderer) {
        std::cout << "    Simulation completed in " << steps << " steps." << std::endl;
    }

    std::vector<ShapeBench::Orientation> orientations(meshes.size());

    for(int i = 0; i < meshes.size(); i++) {
        if(!meshIncluded.at(i)) {
            float invalidValue = std::nanf("");
            orientations.at(i) = {{invalidValue, invalidValue, invalidValue}, {invalidValue, invalidValue, invalidValue, invalidValue}};
            continue;
        }

        JPH::RVec3 outputPosition = body_interface.GetPosition(simulatedBodies.at(i));
        JPH::Quat rotation = body_interface.GetRotation(simulatedBodies.at(i));

        orientations.at(i) = {
                {outputPosition.GetX(), outputPosition.GetY(), outputPosition.GetZ()},
                {rotation.GetX(), rotation.GetY(), rotation.GetZ(), rotation.GetW()}};
    }

    /*ShapeDescriptor::cpu::Mesh outputMesh(outputSampleMesh.vertexCount + outputAdditiveNoiseMesh.vertexCount);
    std::copy(outputSampleMesh.vertices, outputSampleMesh.vertices + outputSampleMesh.vertexCount, outputMesh.vertices);
    std::copy(outputSampleMesh.normals, outputSampleMesh.normals + outputSampleMesh.vertexCount, outputMesh.normals);
    std::copy(outputSampleMesh.vertexColours, outputSampleMesh.vertexColours + outputSampleMesh.vertexCount, outputMesh.vertexColours);
    std::copy(outputAdditiveNoiseMesh.vertices, outputAdditiveNoiseMesh.vertices + outputAdditiveNoiseMesh.vertexCount, outputMesh.vertices + outputSampleMesh.vertexCount);
    std::copy(outputAdditiveNoiseMesh.normals, outputAdditiveNoiseMesh.normals + outputAdditiveNoiseMesh.vertexCount, outputMesh.normals + outputSampleMesh.vertexCount);
    std::copy(outputAdditiveNoiseMesh.vertexColours, outputAdditiveNoiseMesh.vertexColours + outputAdditiveNoiseMesh.vertexCount, outputMesh.vertexColours + outputSampleMesh.vertexCount);

    ShapeDescriptor::writeOBJ(outputMesh, "clutterScene_" + ShapeDescriptor::generateUniqueFilenameString() + ".obj");

    ShapeDescriptor::free(outputMesh);*/

    if(settings.enableDebugRenderer) {
        renderer->destroy();
        delete renderer;
    }

    // Remove the sphere from the physics system. Note that the sphere itself keeps all of its state and can be re-added at any time.
    for(uint32_t i = 0; i < simulatedBodies.size(); i++) {
        if(!meshIncluded.at(i)) {
            continue;
        }
        body_interface.RemoveBody(simulatedBodies.at(i));
        body_interface.DestroyBody(simulatedBodies.at(i));
    }

    // Remove and destroy the floor
    body_interface.RemoveBody(floor->GetID());
    body_interface.DestroyBody(floor->GetID());

    // Unregisters all types with the factory and cleans up the default material
    JPH::UnregisterTypes();

    // Destroy the factory
    delete JPH::Factory::sInstance;
    JPH::Factory::sInstance = nullptr;

    return orientations;
}

void ShapeBench::runAdditiveNoiseFilter(AdditiveNoiseFilterSettings settings, ShapeBench::FilteredMeshPair& scene, const ShapeBench::Dataset& dataset, uint64_t randomSeed, AdditiveNoiseCache& cache) {
    std::filesystem::path datasetRootDir = settings.compressedDatasetRootDir;
    uint32_t clutterObjectCount = settings.addedClutterObjectCount;
    std::vector<ShapeBench::VertexInDataset> chosenVertices = dataset.sampleVertices(randomSeed, clutterObjectCount);
    std::vector<ShapeDescriptor::cpu::Mesh> meshes(chosenVertices.size() + 1);
    meshes.at(0) = scene.filteredSampleMesh;

    // Load meshes
#pragma omp parallel for
    for(uint32_t i = 1; i < meshes.size(); i++) {
        ShapeBench::DatasetEntry entry = dataset.at(chosenVertices.at(i - 1).meshID);
        std::filesystem::path meshFilePath = datasetRootDir / entry.meshFile.replace_extension(".cm");
        meshes.at(i) = ShapeDescriptor::loadMesh(meshFilePath);
        moveMeshToOriginAndUnitSphere(meshes.at(i), entry.computedObjectCentre, entry.computedObjectRadius);
    }

    // Compute orientations by doing a physics simulation or using a cached result
    std::vector<ShapeBench::Orientation> objectOrientations;
    if(!cache.contains(randomSeed)) {
        objectOrientations = runPhysicsSimulation(settings, meshes);
        cache.set(randomSeed, objectOrientations);
    } else {
        objectOrientations = cache.get(randomSeed);
    }

    // Constructing cluttered scene

    uint32_t totalAdditiveNoiseVertexCount = 0;
    for(int i = 1; i < meshes.size(); i++) {
        if(std::isnan(objectOrientations.at(i).position.x)) {
            continue;
        }
        totalAdditiveNoiseVertexCount += meshes.at(i).vertexCount;
    }

    ShapeDescriptor::cpu::Mesh outputSampleMesh(meshes.at(0).vertexCount);
    ShapeDescriptor::cpu::Mesh outputAdditiveNoiseMesh(totalAdditiveNoiseVertexCount);
    uint32_t nextVertexIndex = 0;

    for(int i = 0; i < meshes.size(); i++) {
        if(std::isnan(objectOrientations.at(i).position.x)) {
            continue;
        }

        ShapeBench::Orientation orientation = objectOrientations.at(i);

        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0), glm::vec3(orientation.position.x, orientation.position.y, orientation.position.z));
        glm::mat4 rotationMatrix = glm::toMat4(glm::qua(orientation.rotation.w, orientation.rotation.x, orientation.rotation.y, orientation.rotation.z));
        glm::mat4 transformationMatrix = translationMatrix * rotationMatrix;
        glm::mat4 normalMatrix = glm::inverseTranspose(transformationMatrix);

        ShapeDescriptor::cpu::Mesh& meshToWriteTo = (i == 0) ? outputSampleMesh : outputAdditiveNoiseMesh;

        const ShapeDescriptor::cpu::Mesh& mesh = meshes.at(i);
        for(uint32_t vertexIndex = 0; vertexIndex < mesh.vertexCount; vertexIndex++) {
            ShapeDescriptor::cpu::float3 vertex = mesh.vertices[vertexIndex];
            ShapeDescriptor::cpu::float3 normal = mesh.normals[vertexIndex];
            glm::vec4 transformedVertexGLM = transformationMatrix * glm::vec4(vertex.x, vertex.y, vertex.z, 1.0);
            meshToWriteTo.vertices[nextVertexIndex + vertexIndex] = ShapeDescriptor::cpu::float3(transformedVertexGLM.x, transformedVertexGLM.y, transformedVertexGLM.z);
            glm::vec3 transformedNormalGLM = glm::normalize(glm::vec3(normalMatrix * glm::vec4(normal.x, normal.y, normal.z, 1)));
            meshToWriteTo.normals[nextVertexIndex + vertexIndex] = ShapeDescriptor::cpu::float3(transformedNormalGLM.x, transformedNormalGLM.y, transformedNormalGLM.z);
        }
        if(i > 0) {
            nextVertexIndex += mesh.vertexCount;
        }
    }

    ShapeDescriptor::free(scene.filteredAdditiveNoise);
    scene.filteredAdditiveNoise = outputAdditiveNoiseMesh;
    ShapeDescriptor::free(scene.filteredSampleMesh);
    scene.filteredSampleMesh = outputSampleMesh;


    for(uint32_t i = 1; i < meshes.size(); i++) {
        ShapeDescriptor::free(meshes.at(i));
    }
}
