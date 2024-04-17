import json
import math
import os
import argparse
import csv

# Need to be installed on a fresh system
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio

pio.kaleido.scope.mathjax = None


class ExperimentSettings:
    pass


def getProcessingSettings(mode, fileContents):
    experimentID = fileContents["experiment"]["index"]
    experimentName = fileContents["configuration"]["experimentsToRun"][experimentID]["name"]
    settings = ExperimentSettings()
    settings.minSamplesPerBin = 25
    settings.binCount = 75
    settings.xAxisTitleAdjustment = 0
    settings.heatmapRankLimit = 0
    settings.xAxisOutOfRangeMode = 'discard'
    settings.experimentName = experimentName
    settings.methodName = fileContents["method"]['name']
    settings.methodName = "Spin Image" if settings.methodName == "SI" else settings.methodName
    settings.title = settings.methodName
    sharedYAxisTitle = "Proportion of DDI"
    if experimentName == "normal-noise-only":
        settings.xAxisTitle = "Normal vector rotation (°)"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = fileContents['configuration']['filterSettings']['normalVectorNoise'][
            'maxAngleDeviationDegrees']
        settings.xTick = 5
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["normal-noise-deviationAngle"]
        return settings
    elif experimentName == "subtractive-noise-only":
        settings.xAxisTitle = "Fraction of surface removed"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisTitleAdjustment = 2
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisMin = 0
        settings.xAxisMax = 1
        settings.xTick = 0.2
        settings.enable2D = False
        settings.reverse = True
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        return settings
    elif experimentName == "additive-noise-only":
        settings.xAxisTitle = "Fraction of clutter added"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 10
        settings.xTick = 1
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["fractionAddedNoise"]
        return settings
    elif experimentName == "support-radius-deviation-only":
        settings.xAxisTitle = "Support radius scale factor"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 1 - fileContents['configuration']['filterSettings']['supportRadiusDeviation'][
            'maxRadiusDeviation']
        settings.xAxisMax = 1 + fileContents['configuration']['filterSettings']['supportRadiusDeviation'][
            'maxRadiusDeviation']
        settings.xTick = 0.125
        settings.enable2D = False
        settings.reverse = True # scale factor used is stored as-is, but the relative change to the support radius is the inverse
        settings.readValueX = lambda x: x["filterOutput"]["support-radius-scale-factor"]
        return settings
    elif experimentName == "repeated-capture-only":
        settings.xAxisTitle = "Vertex displacement distance"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 0.15
        settings.xAxisTitleAdjustment = 3
        settings.xTick = 0.03
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["triangle-shift-average-edge-length"]
        return settings
    elif experimentName == "gaussian-noise-only":
        settings.xAxisTitle = "Standard Deviation"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisMin = 0
        settings.xAxisMax = 0.01
        settings.xTick = 0.005
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "depth-camera-capture-only":
        settings.xAxisTitle = "Object distance from camera"
        settings.yAxisTitle = sharedYAxisTitle
        settings.xAxisTitleAdjustment = 2
        settings.xAxisMin = 2
        settings.xAxisMax = 10
        settings.xTick = 1
        settings.enable2D = False
        settings.reverse = False
        settings.readValueX = lambda x: x["filterOutput"][
            "depth-camera-capture-distance-from-camera"]  # (float(x["filterOutput"]["depth-camera-capture-initial-vertex-count"])
        # / float(x["filterOutput"]["depth-camera-capture-filtered-vertex-count"]))
        return settings
    elif experimentName == "additive-and-gaussian-noise":
        settings.xAxisTitle = "Fraction added clutter"
        settings.yAxisTitle = "Standard Deviation"
        settings.xAxisBounds = [0, 25]
        settings.yAxisBounds = [0, 0.01]
        settings.xTick = 5
        settings.yTick = 0.002
        settings.binCount = 50
        settings.enable2D = True
        settings.reverseX = False
        settings.readValueX = lambda x: x["fractionAddedNoise"]
        settings.readValueY = lambda x: x["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    elif experimentName == "additive-and-subtractive-noise":
        settings.xAxisTitle = "Fraction of surface removed"
        settings.yAxisTitle = "Fraction of added clutter"
        settings.xAxisTitleAdjustment = 2
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 25]
        settings.xTick = 0.2
        settings.yTick = 5
        settings.binCount = 50
        settings.enable2D = True
        settings.reverseX = True
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        settings.readValueY = lambda x: x["fractionAddedNoise"]
        return settings
    elif experimentName == "subtractive-and-gaussian-noise":
        settings.xAxisTitle = "Fraction of surface removed"
        settings.yAxisTitle = "Standard Deviation"
        settings.xAxisOutOfRangeMode = 'clamp'
        settings.xAxisTitleAdjustment = 6
        settings.xAxisBounds = [0, 1]
        settings.yAxisBounds = [0, 0.01]
        settings.xTick = 0.2
        settings.yTick = 0.002
        settings.binCount = 35
        settings.enable2D = True
        settings.reverseX = True
        settings.readValueX = lambda x: x["fractionSurfacePartiality"]
        settings.readValueY = lambda x: x["filterOutput"]["gaussian-noise-max-deviation"]
        return settings
    else:
        raise Exception("Failed to determine chart settings: Unknown experiment name: " + experimentName)


def processSingleFile(jsonContent, settings):
    chartDataSequence = {}
    chartDataSequence["name"] = settings.methodName
    chartDataSequence["x"] = []
    chartDataSequence["y"] = []
    if settings.enable2D:
        chartDataSequence["ranks"] = []

    rawResults = []

    for result in jsonContent["results"]:
        if not settings.enable2D:
            rawResult = [settings.readValueX(result), result['filteredDescriptorRank']]
        else:
            rawResult = [settings.readValueX(result), settings.readValueY(result), result['filteredDescriptorRank']]
        rawResults.append(rawResult)
        chartDataSequence["x"].append(rawResult[0])
        chartDataSequence["y"].append(rawResult[1])
        if settings.enable2D:
            chartDataSequence["ranks"].append(rawResult[2])

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
                labels.append(str(int(10 ** (i - 1)) + 1) + " - " + str(int(10 ** i)))

    xValues = [((float(x + 1) * delta) + histogramMin) for x in range(settings.binCount)]

    removedCount = 0
    for rawResult in rawResults:
        if rawResult[0] is None:
            rawResult[0] = 0
        if rawResult[0] < histogramMin or rawResult[0] > histogramMax:
            if settings.xAxisOutOfRangeMode == 'discard':
                removedCount += 1
                continue
            elif settings.xAxisOutOfRangeMode == 'clamp':
                rawResult[0] = max(histogramMin, min(histogramMax, rawResult[0]))

        if settings.reverse:
            rawResult[0] = (histogramMax + histogramMin) - rawResult[0]

        binIndexX = int((rawResult[0] - histogramMin) / delta)
        binIndexY = int(0 if rawResult[1] == 0 else (math.log10(rawResult[1]) + 1))
        if rawResult[1] == representativeSetSize:
            binIndexY -= 1
        histogram[binIndexY][binIndexX] += 1

    counts = [sum(x) for x in zip(*histogram)]

    print('Excluded', removedCount, 'entries')

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


def generateSupportRadiusChart(results_directory, output_directory):
    csvFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".txt")]
    csvFilePaths.sort()

    if len(csvFilePaths) == 0:
        print("No json files were found in this directory. Aborting.")
        return

    for index, csvFileName in enumerate(csvFilePaths):
        isLastFile = index + 1 == len(csvFilePaths)
        csvFilePath = os.path.join(results_directory, csvFileName)
        with open(csvFilePath) as csvFile:
            print('Processing file:', csvFileName)
            reader = csv.DictReader(csvFile)
            sequenceDict = {}
            for row in reader:
                for key in row:
                    key_stripped = key.strip()
                    if key_stripped not in sequenceDict:
                        sequenceDict[key_stripped] = []
                    sequenceDict[key_stripped].append(float(row[key]))
            xValues = sequenceDict["radius"]
            yValues_Sequence1 = sequenceDict["Min mean"]
            yValues_Sequence2 = sequenceDict["Mean"]
            yValues_Sequence3 = sequenceDict["Max mean"]

            countsFigure = go.Figure()

            lineLocation = xValues[yValues_Sequence2.index(max(yValues_Sequence2))]
            print(lineLocation)

            methodName = csvFileName.split("_")[3]
            yAxisRange = [0, max(yValues_Sequence3)]
            useLog = False
            if methodName == 'SI':
                # The Spin image method uses Pearson Correlation
                # Because as opposed to all other distance metrics, a higher number is better for Pearson, we computed 1 - pearsonDistance
                # such that any distance would be between 0 and 2, where lower is better like all other methods
                # We therefore need to do an inverse of that here to show correct distances in the chart
                # (that make sense to the reader anyway)
                yValues_Sequence1 = [1 - x for x in yValues_Sequence1]
                yValues_Sequence2 = [1 - x for x in yValues_Sequence2]
                yValues_Sequence3 = [1 - x for x in yValues_Sequence3]
                yAxisRange = [-0.5, 0.5]
            elif methodName == "RICI":
                useLog = True
                # Make line visible
                yValues_Sequence1 = [max(x, 1) for x in yValues_Sequence1]
                yAxisRange = [0, math.log10(max(yValues_Sequence3))]

            yAxisTickMode = 'linear' if not useLog else 'log'

            chartWidth = 285
            if isLastFile:
                #countsFigure.update_layout(legend=dict(y=0, orientation="h", yanchor="bottom", yref="container", xref="paper", xanchor="left"))
                chartWidth = 350
            else:
                countsFigure.update_layout(showlegend=False)

            pio.kaleido.scope.default_width = chartWidth
            pio.kaleido.scope.default_height = 300

            countsFigure.add_trace(go.Scatter(x=xValues, y=yValues_Sequence1, mode='lines', name="Min"))
            countsFigure.add_trace(go.Scatter(x=xValues, y=yValues_Sequence2, mode='lines', name="Mean"))
            countsFigure.add_trace(go.Scatter(x=xValues, y=yValues_Sequence3, mode='lines', name="Max"))
            countsFigure.add_vline(x=lineLocation)
            countsFigure.update_yaxes(range=yAxisRange, type=yAxisTickMode)
            countsFigure.update_layout(xaxis_title="Radius", yaxis_title='Distance',
                                       title_x=0.5, margin={'t': 2, 'l': 0, 'b': 0, 'r': 10}, width=chartWidth, height=270,
                                       font=dict(size=18), xaxis=dict(tickmode='linear', dtick=0.5, range=(0, max(xValues))))

            outputFile = os.path.join(output_directory, "support-radius-" + methodName + ".pdf")

            pio.write_image(countsFigure, outputFile, engine="kaleido", validate=True)
    print('Done.')


def create2DChart(rawResults, configuration, settings, output_directory, jsonFilePath, jsonFilePaths):
    histogramAccepted = [[0] * settings.binCount for i in range(settings.binCount)]
    histogramTotal = [[0] * settings.binCount for i in range(settings.binCount)]

    removedCount = 0

    deltaX = (settings.xAxisBounds[1] - settings.xAxisBounds[0]) / settings.binCount
    deltaY = (settings.yAxisBounds[1] - settings.yAxisBounds[0]) / settings.binCount

    for rawResult in rawResults:
        resultX, resultY, rank = rawResult
        if resultX is None:
            resultX = 0
        if resultY is None:
            resultY = 0
        if settings.reverseX:
            resultX = settings.xAxisBounds[1] - resultX

        if resultX < settings.xAxisBounds[0] or resultX > settings.xAxisBounds[1]:
            if settings.xAxisOutOfRangeMode == 'discard':
                removedCount += 1
                continue
            elif settings.xAxisOutOfRangeMode == 'clamp':
                rawResult[0] = max(settings.xAxisBounds[0], min(settings.xAxisBounds[1], rawResult[0]))
        if resultY < settings.yAxisBounds[0] or resultY > settings.yAxisBounds[1]:
            removedCount += 1
            continue

        binIndexX = max(0, min(settings.binCount - 1, int((resultX - settings.xAxisBounds[0]) / deltaX)))
        binIndexY = max(0, min(settings.binCount - 1, int((resultY - settings.yAxisBounds[0]) / deltaY)))
        if rank <= settings.heatmapRankLimit:
            histogramAccepted[binIndexY][binIndexX] += 1
        histogramTotal[binIndexY][binIndexX] += 1

    dataRangeX = []
    dataRangeY = []
    dataRangeZ = []
    print("Removed", removedCount, "samples")

    for row in range(0, settings.binCount):
        for col in range(0, settings.binCount):
            dataRangeX.append(col * deltaX + 0.5 * deltaX)
            dataRangeY.append(row * deltaY + 0.5 * deltaY)
            if histogramTotal[row][col] < 10:
                dataRangeZ.append(None)
            else:
                dataRangeZ.append(float(histogramAccepted[row][col]) / histogramTotal[row][col])

    stackFigure = go.Figure(go.Heatmap(x=dataRangeX, y=dataRangeY, z=dataRangeZ,
                                       zmin=0, zmax=1, colorscale=
        [
            [0, 'rgb(200, 200, 200)'],
            [0.25, 'rgb(220, 50, 47)'],
            [0.5, 'rgb(203, 75, 22)'],
            [0.75, 'rgb(181, 137, 000)'],
            [1.0, 'rgb(0, 128, 64)']
        ]))

    xAxisTitle = settings.xAxisTitle
    if jsonFilePath is not jsonFilePaths[-1]:
        stackFigure.update_coloraxes(showscale=False)
        stackFigure.update_traces(showscale=False)
        pio.kaleido.scope.default_width = 300
        pio.kaleido.scope.default_height = 300
        if settings.xAxisTitleAdjustment > 0:
            xAxisTitle += ' ' * settings.xAxisTitleAdjustment
            xAxisTitle += 't'
    else:
        pio.kaleido.scope.default_width = 368
        pio.kaleido.scope.default_height = 300

    stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title=settings.yAxisTitle,
                              margin={'t': 0, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18),
                              xaxis=dict(autorange=False, automargin=True, dtick=settings.xTick, range=settings.xAxisBounds),
                              yaxis=dict(autorange=False, automargin=True, dtick=settings.yTick, range=settings.yAxisBounds))
    #stackFigure.show()

    outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + ".pdf")
    pio.write_image(stackFigure, outputFile, engine="kaleido", validate=True)




def createChart(results_directory, output_directory, mode):
    pio.templates[pio.templates.default].layout.colorway = [
        '#%02x%02x%02x' % (0, 128, 64),
        '#%02x%02x%02x' % (181, 137, 000),
        '#%02x%02x%02x' % (203, 75, 22),
        '#%02x%02x%02x' % (220, 50, 47),
        '#%02x%02x%02x' % (211, 54, 130),
        '#%02x%02x%02x' % (108, 113, 196),
        '#%02x%02x%02x' % (38, 139, 210),
        '#%02x%02x%02x' % (42, 161, 152),
        '#%02x%02x%02x' % (133, 153, 000),
    ]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    if mode == "support-radius":
        generateSupportRadiusChart(results_directory, output_directory)
        return

    # Find all JSON files in results directory
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

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

            if settings.enable2D:
                create2DChart(rawResults, jsonContents["configuration"], settings, output_directory, jsonFilePath, jsonFilePaths)
                continue
            else:
                stackedXValues, stackedYValues, stackedLabels, counts = computeStackedHistogram(rawResults, jsonContents["configuration"], settings)
            allCounts.append(counts)
            countLabels.append(settings.methodName)
            countXValues = stackedXValues
            stackFigure = go.Figure()

            for index, yValueStack in enumerate(stackedYValues):
                stackFigure.add_trace(
                    go.Scatter(x=stackedXValues, y=yValueStack, name=stackedLabels[index], stackgroup="main"))

            xAxisTitle = settings.xAxisTitle
            if jsonFilePath is not jsonFilePaths[-1]:
                if settings.xAxisTitleAdjustment > 0:
                    xAxisTitle += ' ' * settings.xAxisTitleAdjustment
                    xAxisTitle += 't'
                stackFigure.update_layout(showlegend=False)
                titleX = 0.5
                pio.kaleido.scope.default_width = 300
                pio.kaleido.scope.default_height = 300
            else:
                pio.kaleido.scope.default_width = 475
                pio.kaleido.scope.default_height = 300
                titleX = (float(200) / float(500)) * 0.5

            stackFigure.update_yaxes(range=[0, 1])
            stackFigure.update_xaxes(range=[settings.xAxisMin, settings.xAxisMax])

            stackFigure.update_layout(xaxis_title=xAxisTitle, yaxis_title=settings.yAxisTitle, title_x=titleX,
                                      margin={'t': 0, 'l': 0, 'b': 45, 'r': 15}, font=dict(size=18), xaxis=dict(tickmode='linear', dtick=settings.xTick))
            # stackFigure.show()

            outputFile = os.path.join(output_directory, settings.experimentName + "-" + settings.methodName + ".pdf")
            pio.write_image(stackFigure, outputFile, engine="kaleido", validate=True)

    if not settings.enable2D:
        print('Writing counts chart..')
        countsFigure = go.Figure()
        for index, countSet in enumerate(allCounts):
            countSet = [x if x >= settings.minSamplesPerBin else None for x in countSet]
            countsFigure.add_trace(go.Scatter(x=countXValues, y=countSet, mode='lines', name=countLabels[index]))
        countsFigure.update_yaxes(range=[0, math.log10(max([max(x) for x in allCounts]))], type="log")
        countsFigure.update_xaxes(range=[lastSettings.xAxisMin, lastSettings.xAxisMax], dtick=settings.xTick)
        countsFigure.update_layout(xaxis_title=settings.xAxisTitle, yaxis_title='Sample Count',
                                   title_x=0.5, margin={'t': 0, 'l': 0, 'b': 0, 'r': 0},
                                   font=dict(size=18), xaxis=dict(
                tickmode='linear',
                dtick=settings.xTick,
                range=(lastSettings.xAxisMin, lastSettings.xAxisMax)

            ))
        #countsFigure.update_layout(
        #    legend=dict(y=0, orientation="h", yanchor="bottom", yref="container", xref="paper", xanchor="left"))
        outputFile = os.path.join(output_directory, lastSettings.experimentName + "-counts.pdf")
        pio.kaleido.scope.default_width = 435
        pio.kaleido.scope.default_height = 300
        pio.write_image(countsFigure, outputFile, engine="kaleido", validate=True)

    print('Done.')


def main():
    parser = argparse.ArgumentParser(description="Generates charts for the experiment results")
    parser.add_argument("--results-directory", help="Directory containing JSON output files produced by ShapeBench",
                        required=True)
    parser.add_argument("--output-dir", help="Where to write the chart images", required=True)
    parser.add_argument("--mode", help="Specifies what the x-axis of the chart should represent",
                        choices=["auto", "support-radius"],
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
