import json
import math
import os
import argparse

# Need to be installed on a fresh system
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt


class ExperimentSettings:
    pass


def getProcessingSettings(mode, fileContents):
    experimentID = fileContents["experiment"]["index"]
    experimentName = fileContents["configuration"]["experimentsToRun"][experimentID]["name"]
    settings = ExperimentSettings()
    settings.minSamplesPerBin = 50
    settings.binCount = 150
    if experimentName == "normal-noise-only":
        settings.title = ""
        settings.xAxisTitle = "Normal deviation (degrees)"
        settings.yAxisTitle = "Descriptor Index"
        settings.xAxisMin = 0
        settings.xAxisMax = fileContents['configuration']['filterSettings']['normalVectorNoise']['maxAngleDeviationDegrees']
        settings.readValueX = lambda x : x["filterOutput"]["normal-noise-deviationAngle"]
        return settings
    elif experimentName == "subtractive-noise-only":
        settings.title = ""
        settings.xAxisTitle = "Fraction of surface remaining"
        settings.yAxisTitle = "Descriptor Index"
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        return settings
    elif experimentName == "additive-noise-only":
        settings.title = ""
        settings.xAxisTitle = "Fraction of clutter added"
        settings.yAxisTitle = "Descriptor Index"
        settings.xAxisMin = 0
        settings.xAxisMax = 10
        settings.readValueX = lambda x: x["fractionAddedNoise"]
        return settings
    elif experimentName == "support-radius-deviation-only":
        settings.title = ""
        settings.xAxisTitle = "Relative Support Radius"
        settings.yAxisTitle = "Descriptor Index"
        settings.xAxisMin = 1 - fileContents['configuration']['filterSettings']['supportRadiusDeviation']['maxRadiusDeviation']
        settings.xAxisMax = 1 + fileContents['configuration']['filterSettings']['supportRadiusDeviation']['maxRadiusDeviation']
        settings.readValueX = lambda x: x["filterOutput"]["support-radius-scale-factor"]
        return settings
    else:
        raise Exception("Failed to determine chart settings: Unknown experiment name: " + experimentName)




def processSingleFile(jsonContent, settings):
    chartDataSequence = {}
    chartDataSequence["name"] = jsonContent["method"]["name"]
    chartDataSequence["x"] = []
    chartDataSequence["y"] = []

    rawResults = []

    for result in jsonContent["results"]:
        rawResult = [settings.readValueX(result), result['filteredDescriptorRank']]
        rawResults.append(rawResult)
        chartDataSequence["x"].append(rawResult[0])
        chartDataSequence["y"].append(rawResult[1])

    return chartDataSequence, rawResults


def computeStackedHistogram(rawResults, config, settings):
    histogramMax = settings.xAxisMax
    histogramMin = settings.xAxisMin

    delta = (histogramMax - histogramMin) / settings.binCount

    representativeSetSize = config['commonExperimentSettings']['representativeSetSize']
    stepsPerBin = int(math.log10(representativeSetSize)) + 1

    histogram = []
    labels = []
    for i in range(stepsPerBin):
        histogram.append([0] * (settings.binCount + 1))
        if i == 0:
            labels.append('0')
        elif i == 1:
            labels.append('1 - 10')
        elif i == stepsPerBin - 1:
            labels.append(str(int(representativeSetSize)))
        else:
            labels.append(str(int(10 ** (i-1)) + 1) + " - " + str(int(10 ** i)))

    xValues = [((float(x+1) * delta) + histogramMin) for x in range(settings.binCount + 1)]

    removedCount = 0
    for rawResult in rawResults:
        if rawResult[0] is None:
            rawResult[0] = 0
        if rawResult[0] < histogramMin or rawResult[0] > histogramMax:
            removedCount += 1
            continue
        binIndexX = int((rawResult[0] - histogramMin) / delta)
        binIndexY = int(0 if rawResult[1] == 0 else math.log10(rawResult[1]))
        histogram[binIndexY][binIndexX] += 1

    # Time to normalise

    for i in range(settings.binCount + 1):
        stepSum = 0
        for j in range(stepsPerBin):
            stepSum += histogram[j][i]
        if stepSum > settings.minSamplesPerBin:
            for j in range(stepsPerBin):
                histogram[j][i] = float(histogram[j][i]) / float(stepSum)
        else:
            for j in range(stepsPerBin):
                histogram[j][i] = 0

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
            settings = getProcessingSettings(mode, jsonContents)
            dataSequence, rawResults = processSingleFile(jsonContents, settings)
            #figure = go.Figure()
            #figure.add_trace(go.Scatter(x=dataSequence["x"], y=dataSequence['y'], mode='markers', name=dataSequence['name']))
            xAxisLabel = settings.xAxisTitle
            #figure.update_layout(title='Title missing', xaxis_title=xAxisLabel, yaxis_title='Image rank')
            #figure.show()

            stackedXValues, stackedYValues, stackedLabels = computeStackedHistogram(rawResults, jsonContents["configuration"], settings)
            stackFigure = go.Figure()
            for index, yValueStack in enumerate(stackedYValues):
                stackFigure.add_trace(go.Scatter(x=stackedXValues, y=yValueStack, name=stackedLabels[index], stackgroup="main"))
            stackFigure.update_layout(title=jsonFilePath, xaxis_title=xAxisLabel, yaxis_title='Image rank')
            stackFigure.show()

def main():
    parser = argparse.ArgumentParser(description="Generates charts for the experiment results")
    parser.add_argument("--results-directory", help="Directory containing JSON output files produced by ShapeBench", required=True)
    parser.add_argument("--output-dir", help="Where to write the chart images", required=True)
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

    createChart(args.results_directory, args.output_dir, args.mode)



if __name__ == "__main__":
    main()

