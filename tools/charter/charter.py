import json
import math
import os
import argparse

# Need to be installed on a fresh system
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
pio.kaleido.scope.mathjax = None
pio.kaleido.scope.default_width = 350
pio.kaleido.scope.default_height = 350

class ExperimentSettings:
    pass


def getProcessingSettings(mode, fileContents):
    experimentID = fileContents["experiment"]["index"]
    experimentName = fileContents["configuration"]["experimentsToRun"][experimentID]["name"]
    settings = ExperimentSettings()
    settings.minSamplesPerBin = 25
    settings.binCount = 150
    settings.experimentName = experimentName
    settings.methodName = fileContents["method"]['name']
    settings.methodName = "Spin Image" if settings.methodName == "SI" else settings.methodName
    if experimentName == "normal-noise-only":
        settings.title = settings.methodName
        settings.xAxisTitle = "Normal deviation (degrees)"
        settings.yAxisTitle = "Fraction of Descriptor Index"
        settings.xAxisMin = 0
        settings.xAxisMax = fileContents['configuration']['filterSettings']['normalVectorNoise']['maxAngleDeviationDegrees']
        settings.xTick = 5
        settings.readValueX = lambda x : x["filterOutput"]["normal-noise-deviationAngle"]
        return settings
    elif experimentName == "subtractive-noise-only":
        settings.title = settings.methodName
        settings.xAxisTitle = "Fraction of surface remaining"
        settings.yAxisTitle = "Fraction of Descriptor Index"
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.xTick = 0.1
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        return settings
    elif experimentName == "additive-noise-only":
        settings.title = settings.methodName
        settings.xAxisTitle = "Fraction of clutter added"
        settings.yAxisTitle = "Fraction of Descriptor Index"
        settings.xAxisMin = 0
        settings.xAxisMax = 10
        settings.xTick = 1
        settings.readValueX = lambda x: x["fractionAddedNoise"]
        return settings
    elif experimentName == "support-radius-deviation-only":
        settings.title = settings.methodName
        settings.xAxisTitle = "Relative Support Radius"
        settings.yAxisTitle = "Fraction of Descriptor Index"
        settings.xAxisMin = 1 - fileContents['configuration']['filterSettings']['supportRadiusDeviation']['maxRadiusDeviation']
        settings.xAxisMax = 1 + fileContents['configuration']['filterSettings']['supportRadiusDeviation']['maxRadiusDeviation']
        settings.xTick = 0.05
        settings.readValueX = lambda x: x["filterOutput"]["support-radius-scale-factor"]
        return settings
    else:
        raise Exception("Failed to determine chart settings: Unknown experiment name: " + experimentName)




def processSingleFile(jsonContent, settings):
    chartDataSequence = {}
    chartDataSequence["name"] = settings.methodName
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
        if i != stepsPerBin:
            histogram.append([0] * (settings.binCount + 1))
            if i == 0:
                labels.append('0')
            elif i == 1:
                labels.append('1 - 10')
            else:
                labels.append(str(int(10 ** (i-1)) + 1) + " - " + str(int(10 ** i)))

    xValues = [((float(x+1) * delta) + histogramMin) for x in range(settings.binCount)]

    removedCount = 0
    for rawResult in rawResults:
        if rawResult[0] is None:
            rawResult[0] = 0
        if rawResult[0] < histogramMin or rawResult[0] > histogramMax:
            removedCount += 1
            continue

        binIndexX = int((rawResult[0] - histogramMin) / delta)
        binIndexY = int(0 if rawResult[1] == 0 else (math.log10(rawResult[1]) + 1))
        if rawResult[1] == representativeSetSize:
            binIndexY -= 1
        histogram[binIndexY][binIndexX] += 1

    counts = [sum(x) for x in zip(*histogram)]

    # Time to normalise

    for i in range(settings.binCount):
        stepSum = 0
        for j in range(stepsPerBin):
            stepSum += histogram[j][i]

        if stepSum > settings.minSamplesPerBin:

            for j in range(stepsPerBin):
                histogram[j][i] = float(histogram[j][i]) / float(stepSum)
        else:
            for j in range(stepsPerBin):
                histogram[j][i] = None

    return xValues, histogram, labels, counts



def createChart(results_directory, output_directory, mode):
    # Find all JSON files in results directory
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    if len(jsonFilePaths) == 0:
        print("No json files were found in this directory. Aborting.")
        return

    allCounts = []
    countXValues = []
    countLabels = []
    lastSettings = ""

    print("Found {} json files".format(len(jsonFilePaths)))
    for jsonFilePath in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFilePath)) as inFile:
            print('Loading file: {}'.format(jsonFilePath))
            jsonContents = json.load(inFile)
            settings = getProcessingSettings(mode, jsonContents)
            print('Creating chart for method ' + settings.methodName + "..")
            lastSettings = settings
            dataSequence, rawResults = processSingleFile(jsonContents, settings)

            stackedXValues, stackedYValues, stackedLabels, counts = computeStackedHistogram(rawResults, jsonContents["configuration"], settings)
            allCounts.append(counts)
            countLabels.append(settings.methodName)
            countXValues = stackedXValues
            stackFigure = go.Figure()
            for index, yValueStack in enumerate(stackedYValues):
                stackFigure.add_trace(go.Scatter(x=stackedXValues, y=yValueStack, name=stackedLabels[index], stackgroup="main"))


            if jsonFilePath is not jsonFilePaths[-1]:
                stackFigure.update_layout(showlegend=False)
                titleX = 0.5
            else:
                pio.kaleido.scope.default_width = 500
                titleX = (float(350) / float(500)) * 0.5

            stackFigure.update_yaxes(range=[0, 1])
            stackFigure.update_xaxes(range=[settings.xAxisMin, settings.xAxisMax])


            stackFigure.update_layout(title=settings.title, xaxis_title=settings.xAxisTitle, yaxis_title=settings.yAxisTitle, title_x=titleX, margin={'t':30,'l':0,'b':45,'r':0}, xaxis = dict(
        tickmode = 'linear',
        dtick = settings.xTick
    ))
            #stackFigure.show()



            outputFile =os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + ".pdf")
            pio.write_image(stackFigure, outputFile, engine="kaleido", validate=True)

    print('Writing counts chart..')
    countsFigure = go.Figure()
    for index, countSet in enumerate(allCounts):
        countSet = [x if x >= settings.minSamplesPerBin else None for x in countSet]
        countsFigure.add_trace(go.Scatter(x=countXValues, y=countSet, mode='lines', name=countLabels[index]))
    countsFigure.update_yaxes(range=[0, math.log10(max([max(x) for x in allCounts]))], type="log")
    countsFigure.update_xaxes(range=[lastSettings.xAxisMin, lastSettings.xAxisMax])
    countsFigure.update_layout(title='Sample Distribution', xaxis_title=settings.xAxisTitle, yaxis_title='Sample Count',
                              title_x=0.5, margin={'t': 30, 'l': 0, 'b': 45, 'r': 0})
    countsFigure.update_layout(legend=dict(y=0, orientation="h", yanchor="bottom", yref="container"))
    outputFile = os.path.join(output_directory, lastSettings.experimentName + "-counts.pdf")
    pio.write_image(countsFigure, outputFile, engine="kaleido", validate=True)

    print('Done.')


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

