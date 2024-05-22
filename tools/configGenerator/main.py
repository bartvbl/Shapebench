import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate configurations for each combination of ")
    parser.add_argument("basefile", help="Path to the base config file")
    parser.add_argument("directory", help="Path to the directory containing files")

    args = parser.parse_args()

    directory_path = args.directory
    os.makedirs(directory_path, exist_ok=True)

    filters = ['additive-noise-only',
               'subtractive-noise-only',
               'normal-noise-only',
               'repeated-capture-only',
               'support-radius-deviation-only',
               'depth-camera-capture-only',
               'gaussian-noise-only',
               'additive-and-subtractive-noise',
               'additive-and-gaussian-noise',
               'subtractive-and-gaussian-noise']
    filtersEnabled = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True
    ]

    methods = ['QUICCI', 'RICI', 'SI', 'USC', 'RoPS']
    methodsEnabled = [True, False, False, False, False]

    configIndex = 0

    for methodIndex, methodName in enumerate(methods):
        if not methodsEnabled[methodIndex]:
            continue
        for filterIndex, filterName in enumerate(filters):
            if not filtersEnabled[filterIndex] and not (not any(filtersEnabled) and filterIndex == 0):
                continue

            filePath = os.path.join(directory_path, "run_" + str(configIndex) + ".json")
            configIndex += 1

            with open(args.basefile, "r") as baseFile:
                outputData = json.load(baseFile)
            #outputData['includes'] = ['config.json']
            outputData['enableComparisonToPRC'] = True

            if filtersEnabled[filterIndex]:
                outputData['experimentsToRun'][methodIndex]['enabled'] = True

            outputData['methodSettings'][methodName]['enabled'] = True

            with open(filePath, 'w') as json_file:
                json.dump(outputData, json_file, indent=4)

if __name__ == "__main__":
    main()
