print("The bindings are working from the Python side as well!")

import sys
print(sys.version)

def functionIWantToCall():
	print('Adapter function was called!')
	descriptors = []
	for i in range(10):
		descriptor = [x for x in range(10)]
		descriptors.append(descriptor)
	return descriptors