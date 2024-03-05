import json
import math
import os
import argparse
import plotly.graph_objects as go
import numpy as np


def determineProcessingMode(mode, fileContents):
    if mode != "auto":
        return mode

    experimentID = fileContents["experiment"]["index"]
    lastFilterName = fileContents["configuration"]["experimentsToRun"][experimentID]["filters"][-1]['type']
    print(lastFilterName)
    if lastFilterName == "normal-noise":
        return "normalDeviation"
    else:
        return "additiveArea"


def processSingleFile(jsonContent, mode):
    chartDataSequence = {}
    chartDataSequence["name"] = jsonContent["method"]["name"]
    chartDataSequence["x"] = []
    chartDataSequence["y"] = []

    for result in jsonContent["results"]:
        chartDataSequence["y"].append(result['filteredDescriptorRank'])
        if mode == "normalDeviation":
            if jsonContent["version"] == "1.1":
                chartDataSequence["x"].append(result['filterOutput']['normal-noise-deviationAngle'])
            else:
                raise IOError
        elif mode == "additiveArea":
            chartDataSequence['x'].append(result['fractionAddedNoise'])
    return chartDataSequence


def createChart(results_directory, output_file, mode):
    # Find all JSON files in results directory
    jsonFilePaths = [x.name for x in os.scandir(results_directory) if x.name.endswith(".json")]
    jsonFilePaths.sort()

    if len(jsonFilePaths) == 0:
        print("No json files were found in this directory. Aborting.")
        return

    print("Found {} json files".format(len(jsonFilePaths)))
    jsonContents = []
    for jsonFilePath in jsonFilePaths:
        with open(os.path.join(results_directory, jsonFilePath)) as inFile:
            print('    Loading file: {}'.format(jsonFilePath))
            jsonContents.append(json.load(inFile))

    mode = determineProcessingMode(mode, jsonContents[0])

    dataSequences = []

    for jsonContent in jsonContents:
        dataSequences.append(processSingleFile(jsonContent, mode))

    figure = go.Figure()
    for dataSequence in dataSequences:
        figure.add_trace(go.Scatter(x=dataSequence["x"], y=dataSequence['y'], mode='markers', name=dataSequence['name']))

    xAxisLabel = 'Fraction clutter area added'
    if mode == "additive-area":
        xAxisLabel = ""
    elif mode == "normal-noise":
        xAxisLabel = "Normal deviation (degrees)"

    figure.update_layout(
        title='Title missing',
        xaxis_title=xAxisLabel,
        yaxis_title='Image rank',
    )

    figure.show()


def main():
    parser = argparse.ArgumentParser(description="Generates charts for the experiment results")
    parser.add_argument("--results-directory", help="Directory containing JSON output files produced by ShapeBench", required=True)
    parser.add_argument("--output-file", help="Where to write the chart image", required=True)
    parser.add_argument("--mode", help="Specifies what the x-axis of the chart should represent",
                        choices=["auto", "normalDeviation", "additiveArea", "removedArea"],
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

