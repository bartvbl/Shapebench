import json
import math
import os
import argparse
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

def determineProcessingMode(mode, fileContents):
    if mode != "auto":
        return mode

    experimentID = fileContents["experiment"]["index"]
    lastFilterName = fileContents["configuration"]["experimentsToRun"][experimentID]["filters"][-1]['type']
    print(lastFilterName)
    if lastFilterName == "normal-noise":
        return "normal"
    elif lastFilterName == "subtractive-noise":
        return "subtractive"
    elif lastFilterName == "additive-noise":
        return "additive"
    else:
        raise Exception("Unknown filter name: " + lastFilterName)


def getXValue(result, mode):
    if mode == "normal":
        return result['filterOutput']['normal-noise-deviationAngle']
    elif mode == "additive":
        return result['fractionAddedNoise']
    elif mode == "subtractive":
        return result['fractionSurfacePartiality']


def getXAxisLabel(mode):
    if mode == "additive":
        return 'Fraction clutter area added'
    elif mode == "normal":
        return "Normal deviation (degrees)"
    elif mode == "subtractive":
        return "Fraction of area removed"


def processSingleFile(jsonContent, mode):
    chartDataSequence = {}
    chartDataSequence["name"] = jsonContent["method"]["name"]
    chartDataSequence["x"] = []
    chartDataSequence["y"] = []

    rawResults = []

    for result in jsonContent["results"]:
        rawResult = [getXValue(result, mode), result['filteredDescriptorRank']]
        rawResults.append(rawResult)
        chartDataSequence["x"].append(rawResult[0])
        chartDataSequence["y"].append(rawResult[1])

    return chartDataSequence, rawResults


def computeStackedHistogram(rawResults, config, mode):
    histogramMin = 0
    histogramMax = 1

    if mode == "normal":
        histogramMax = config['filterSettings']['normalVectorNoise']['maxAngleDeviationDegrees']

    binCount = 100
    delta = (histogramMax - histogramMin) / binCount

    representativeSetSize = config['commonExperimentSettings']['representativeSetSize']
    stepsPerBin = int(math.log10(representativeSetSize)) + 1

    histogram = []
    labels = []
    for i in range(stepsPerBin):
        histogram.append([0] * (binCount + 1))
        if i == 0:
            labels.append('0')
        elif i == 1:
            labels.append('1 - 10')
        elif i == stepsPerBin - 1:
            labels.append(str(int(representativeSetSize)))
        else:
            labels.append(str(int(10 ** (i-1)) + 1) + " - " + str(int(10 ** i)))

    xValues = [((float(x+1) * delta) + histogramMin) for x in range(binCount + 1)]

    removedCount = 0
    for rawResult in rawResults:
        if rawResult[0] < histogramMin or rawResult[0] > histogramMax:
            removedCount += 1
            continue
        binIndexX = int((rawResult[0] - histogramMin) / delta)
        binIndexY = int(0 if rawResult[1] == 0 else math.log10(rawResult[1]))
        histogram[binIndexY][binIndexX] += 1

    # Time to normalise
    for i in range(binCount + 1):
        stepSum = 0
        for j in range(stepsPerBin):
            stepSum += histogram[j][i]
        if stepSum != 0:
            for j in range(stepsPerBin):
                histogram[j][i] = float(histogram[j][i]) / float(stepSum)

    return xValues, histogram, labels



def createChart(results_directory, output_file, mode):
    # Find all JSON files in results directory
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

    if len(jsonFilePaths) == 0:
        print("No json files were found in this directory. Aborting.")
        return

    print("Found {} json files".format(len(jsonFilePaths)))
    for jsonFilePath in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFilePath)) as inFile:
            print('    Loading file: {}'.format(jsonFilePath))
            jsonContents = json.load(inFile)
            mode = determineProcessingMode(mode, jsonContents)
            dataSequence, rawResults = processSingleFile(jsonContents, mode)
            #figure = go.Figure()
            #figure.add_trace(go.Scatter(x=dataSequence["x"], y=dataSequence['y'], mode='markers', name=dataSequence['name']))
            xAxisLabel = getXAxisLabel(mode)
            #figure.update_layout(title='Title missing', xaxis_title=xAxisLabel, yaxis_title='Image rank')
            #figure.show()

            stackedXValues, stackedYValues, stackedLabels = computeStackedHistogram(rawResults, jsonContents["configuration"], mode)
            stackFigure = go.Figure()
            for index, yValueStack in enumerate(stackedYValues):
                stackFigure.add_trace(go.Scatter(x=stackedXValues, y=yValueStack, name=stackedLabels[index], stackgroup="main"))
            stackFigure.update_layout(title=jsonFilePath, xaxis_title=xAxisLabel, yaxis_title='Image rank')
            stackFigure.show()

def main():
    parser = argparse.ArgumentParser(description="Generates charts for the experiment results")
    parser.add_argument("--results-directory", help="Directory containing JSON output files produced by ShapeBench", required=True)
    parser.add_argument("--output-file", help="Where to write the chart image", required=True)
    parser.add_argument("--mode", help="Specifies what the x-axis of the chart should represent",
                        choices=["auto", "normal", "additive", "subtractive"],
                        default="auto", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.results_directory):
        print(f"The specified directory '{args.results_directory}' does not exist.")
        return
    if not os.path.isdir(args.results_directory):
        print(f"The specified directory '{args.results_directory}' is not a directory. You need to specify the "
              f"directory rather than individual JSON files.")
        return

    createChart(args.results_directory, args.output_file, args.mode)



if __name__ == "__main__":
    main()

