import numpy as np

def initMethod(config):
	print('GEDI method active!')

def destroyMethod():
	print('GEDI method destroyed')

def getMethodMetadata():
	return {}

def computeMeshDescriptors(mesh, descriptorOrigins, config, supportRadii, randomSeed):
	print("Processing mesh", mesh.vertices.shape, mesh.normals.shape, descriptorOrigins.shape)
	vertexCount = mesh.vertices.shape[0]
	numberOfDescriptorsToCompute = descriptorOrigins.shape[0]
	descriptorOriginVertex = descriptorOrigins[0][0]
	descriptorOriginNormal = descriptorOrigins[0][1]

	return np.zeros((descriptorOrigins.shape[0], 32), dtype=float)

def computePointCloudDescriptors(pointCloud, descriptorOrigins, config, supportRadii, randomSeed):
	print("Processing cloud", pointCloud.vertices.shape, pointCloud.normals.shape, descriptorOrigins.shape)
	vertexCount = pointCloud.vertices.shape[0]
	numberOfDescriptorsToCompute = descriptorOrigins.shape[0]
	descriptorOriginVertex = descriptorOrigins[0][0]
	descriptorOriginNormal = descriptorOrigins[0][1]

	return np.zeros((descriptorOrigins.shape[0], 32), dtype=float)