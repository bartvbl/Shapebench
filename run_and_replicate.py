#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import random
import sys
import multiprocessing
import hashlib
import base64
from shutil import which
from scripts.simple_term_menu import TerminalMenu
from scripts.prettytable import PrettyTable

if not (sys.version_info.major == 3 and sys.version_info.minor >= 8):
    print("This script requires Python 3.8 or higher.")
    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

os.makedirs('input/objaverse-cache', exist_ok=True)
os.makedirs('input/objaverse-uncompressed', exist_ok=True)

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)
    subprocess.run(command, shell=True, check=False, cwd=working_directory)

def ask_for_confirmation(message):
    confirmation_menu = TerminalMenu(["yes", "no"], title=message)
    choice = confirmation_menu.show()
    return choice == 0

def downloadFile(fileURL, tempFile, extractInDirectory, name, unzipCommand = 'p7zip -k -d {}'):
    os.makedirs('input/download', exist_ok=True)
    if not os.path.isfile('input/download/' + tempFile) or ask_for_confirmation('It appears the ' + name + ' archive file has already been downloaded. Would you like to download it again?'):
        print('Downloading the ' + name + ' archive file..')
        run_command_line_command('wget --output-document ' + tempFile + ' ' + fileURL, 'input/download/')
    print()
    os.makedirs(extractInDirectory, exist_ok=True)
    run_command_line_command(unzipCommand.format(os.path.join(os.path.relpath('input/download', extractInDirectory), tempFile)), extractInDirectory)
    #if ask_for_confirmation('Download and extraction complete. Would you like to delete the compressed archive to save disk space?'):
    #    os.remove('input/download/' + tempFile)
    print()

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download all",
        "Download computed results (7.4GB download, 81.7GB uncompressed)",
        "Download cache files (7.4GB download, 8.4GB uncompressed)",
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show() + 1
        os.makedirs('input/download/', exist_ok=True)

        if choice == 1 or choice == 2:
            downloadFile('https://ntnu.box.com/shared/static/rily0qg6tpzb9prym8ois0korr3x4vxa.7z',
                         'precomputed_results.7z', 'precomputed_results/', 'Results computed by the author')
        if choice == 1 or choice == 3:
            downloadFile('https://ntnu.box.com/shared/static/iqerttzman0eua0mrjxslxea8gt30ayu.7z', 'cache.7z',
                         'input/objaverse-cache', 'Precomputed cache files')
        if choice == 4:
            return

def installDependencies():
    run_command_line_command('sudo apt install ninja-build cmake g++ git libwayland-dev libxkbcommon-x11-dev xorg-dev libssl-dev m4 texinfo libboost-dev libeigen3-dev wget xvfb python3-tk python3-pip libstdc++-12-dev libomp-dev')
    run_command_line_command('pip3 install numpy matplotlib plotly wcwidth kaleido')

def compileProject():
    os.makedirs('bin', exist_ok=True)
    run_command_line_command('rm -rf bin/*')

    cudaCompiler = ''
    if which('nvcc') is None:
        print()
        print('It appears that the CUDA NVCC compiler is not on your path.')
        print('This usually means that CMake doesn\'t manage to find it.')
        print('The most common path at which NVCC is found is: /usr/local/cuda/bin/nvcc')
        nvccPath = input('Please paste the path to NVCC here, write "default" to use the default path listed above, or leave empty to try and run CMake as-is: ')
        if nvccPath != '':
            if nvccPath == 'default':
                nvccPath = '/usr/local/cuda/bin/nvcc'
            cudaCompiler = ' -DCMAKE_CUDA_COMPILER=' + nvccPath


    run_command_line_command('cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja' + cudaCompiler, 'bin')
    if not os.path.isfile('bin/build.ninja'):
        print()
        print('Failed to compile the project: CMake exited with an error.')
        print()
        return
    run_command_line_command('./configure', 'lib/gmp-6.3.0/')
    run_command_line_command('make -j', 'lib/gmp-6.3.0/')
    run_command_line_command('make check', 'lib/gmp-6.3.0/')
    run_command_line_command('ninja ', 'bin')

    print()
    print('Complete.')
    print()

def fileMD5(filePath):
    with open(filePath, 'rb') as inFile:
        return hashlib.md5(inFile.read()).hexdigest()


def generateReplicationSettingsString(node):
    if node['recomputeEntirely']:
        return 'recompute entirely'
    elif node['recomputeRandomSubset']:
        return 'recompute ' + str(node['randomSubsetSize']) + ' at random'
    else:
        return 'disabled'

def editSettings(node, name):
    download_menu = TerminalMenu([
        "Recompute entirely",
        "Recompute random subset",
        "Disable replication",
        "back"], title='------------------ Replication Settings for ' + name + ' ------------------')

    choice = download_menu.show() + 1

    if choice == 1:
        node['recomputeEntirely'] = True
        node['recomputeRandomSubset'] = False
    if choice == 2:
        node['recomputeEntirely'] = False
        node['recomputeRandomSubset'] = True
        print()
        numberOfSamplesToReplicate = int(input('Number of samples to replicate: '))
        node['randomSubsetSize'] = numberOfSamplesToReplicate
    if choice == 3:
        node['recomputeEntirely'] = False
        node['recomputeRandomSubset'] = False

    return node

def selectReplicationRandomSeed(originalSeed):
    download_menu = TerminalMenu([
        "Pick new seed at random",
        "Enter a specific random seed",
        "Keep previous seed (" + str(originalSeed) + ')̈́'], title='------------------ Select New Random Seed ------------------')

    choice = download_menu.show() + 1

    if choice == 1:
        return random.getrandbits(64)
    if choice == 2:
        selectedRandomSeed = input('Enter new random seed: ')
        return int(selectedRandomSeed)
    return originalSeed

def readConfigFile(path = 'cfg/config_replication.json'):
    with open(path, 'r') as cfgFile:
        config = json.load(cfgFile)
        return config

def writeConfigFile(config, path = 'cfg/config_replication.json'):
    with open(path, 'w') as cfgFile:
        json.dump(config, cfgFile, indent=4)

def generateThreadLimiterString(configEntry):
    if not 'threadLimit' in configEntry:
        return 'No thread limit'
    else:
        return 'limited to ' + str(configEntry['threadLimit']) + ' threads'

def applyThreadLimiter(config):
    print()
    print('Thread limits ensure the number of threads working on a particular filter does not exceed a set number')
    print('If you have problems with running out of memory, you can set a thread limiter, and see if that helps things')
    print('Since memory usage scales linearly with more threads, that can help your situation')
    print()
    while True:
        download_menu = TerminalMenu([
            'Clutter: ' + generateThreadLimiterString(config['experimentsToRun'][0]),
            'Occlusion: ' + generateThreadLimiterString(config['experimentsToRun'][1]),
            'Normal vector deviation: ' + generateThreadLimiterString(config['experimentsToRun'][2]),
            'Alternate triangulation: ' + generateThreadLimiterString(config['experimentsToRun'][3]),
            'Support radius deviation: ' + generateThreadLimiterString(config['experimentsToRun'][4]),
            'Alternate mesh resolution: ' + generateThreadLimiterString(config['experimentsToRun'][5]),
            'Gaussian noise: ' + generateThreadLimiterString(config['experimentsToRun'][6]),
            'Clutter and Occlusion: ' + generateThreadLimiterString(config['experimentsToRun'][7]),
            'Clutter and Gaussian noise: ' + generateThreadLimiterString(config['experimentsToRun'][8]),
            'Occlusion and Gaussian noise: ' + generateThreadLimiterString(config['experimentsToRun'][9]),
            "back"], title='------------------ Apply Thread Limiter ------------------')
        choice = download_menu.show()
        if choice < 10:
            print()
            print('Please enter the new thread limit. Use 0 to disable the limit.')
            limit = int(input('New thread limit: '))
            if limit == 0:
                if 'threadLimit' in config['experimentsToRun'][choice]:
                    tempCopy = config['experimentsToRun'][choice]
                    del tempCopy['threadLimit']
                    config['experimentsToRun'][choice] = tempCopy
            else:
                config['experimentsToRun'][choice]['threadLimit'] = limit
        else:
            return config

def changeReplicationSettings():
    config = readConfigFile()

    while True:
        download_menu = TerminalMenu([
            'Compute or replicate experimental results: ' + generateReplicationSettingsString(config['replicationOverrides']['experiment']),
            'Compute or replicate reference descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['referenceDescriptorSet']),
            'Compute or replicate sample object unfiltered descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['sampleDescriptorSet']),
            'Random seed used when selecting random subsets to replicate: ' + str(config['replicationOverrides']['replicationRandomSeed']),
            'Verify computed minimum bounding sphere of input objects: ' + ('enabled' if config['datasetSettings']['verifyFileIntegrity'] else 'disabled'),
            'Size of dataset file cache in GB: ' + str(config['datasetSettings']['cacheSizeLimitGB']),
            'Change location of dataset file cache: ' + config['datasetSettings']['compressedRootDir'],
            'Limit the number of threads per experiment (use this if you run out of RAM)',
            'Print individual experiment results as they are being generated: ' + ('enabled' if config['verboseOutput'] else 'disabled'),
            'Enable visualisations of generated occluded scenes and clutter simulations: ' + ('enabled' if config['filterSettings']['additiveNoise']['enableDebugCamera'] else 'disabled'),
            "back"], title='------------------ Configure Replication ------------------')

        choice = download_menu.show() + 1

        if choice == 1:
            config['replicationOverrides']['experiment'] = editSettings(config['replicationOverrides']['experiment'],'Benchmark Results')
        if choice == 2:
            config['replicationOverrides']['referenceDescriptorSet'] = editSettings(config['replicationOverrides']['referenceDescriptorSet'], 'Reference Descriptor Set')
        if choice == 3:
            config['replicationOverrides']['sampleDescriptorSet'] = editSettings(config['replicationOverrides']['sampleDescriptorSet'], 'Sample Object Unfiltered Descriptor Set')
        if choice == 4:
            config['replicationOverrides']['replicationRandomSeed'] = selectReplicationRandomSeed(config['replicationOverrides']['replicationRandomSeed'])
        if choice == 5:
            config['datasetSettings']['verifyFileIntegrity'] = not config['datasetSettings']['verifyFileIntegrity']
        if choice == 6:
            print()
            newSize = int(input('Size of dataset file cache in GB: '))
            config['datasetSettings']['cacheSizeLimitGB'] = newSize
            print()
        if choice == 7:
            print()
            chosenDirectory = input('Enter a directory path here. Write "choose" for a graphical file chooser: ')
            if chosenDirectory == "choose":
                from tkinter import filedialog
                from tkinter import Tk
                root = Tk()
                root.withdraw()
                chosenDirectory = filedialog.askdirectory()
            config['datasetSettings']['compressedRootDir'] = chosenDirectory
        if choice == 8:
            config = applyThreadLimiter(config)
        if choice == 9:
            config['verboseOutput'] = not config['verboseOutput']
        if choice == 10:
            config['filterSettings']['additiveNoise']['enableDebugCamera'] = not config['filterSettings']['additiveNoise']['enableDebugCamera']
            if config['filterSettings']['additiveNoise']['enableDebugCamera']:
                warningBox = TerminalMenu([
                    "Ok"], title='Note: enabling these visualisations will likely cause filters that rely on OpenGL rendering to not replicate properly.')

                warningBox.show()

        if choice == 11:
            with open('cfg/config_replication.json', 'w') as cfgFile:
                json.dump(config, cfgFile, indent=4)
            return

def replicateSimilarityVisualisationFigure():
    #downloadFile('http://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz', 'armadillo.ply.gz', os.path.abspath('input/figure1'), 'Figure 1 armadillo model', 'gunzip -c {} > ./armadillo.ply')
    os.makedirs('output/figure1', exist_ok=True)
    run_command_line_command('../../bin/armadillo ../../input/figure1/Armadillo_vres2_small_scaled_0.ply', 'output/figure1')
    gradientImageBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAgAAAAABCAYAAACouxZ2AAABbmlDQ1BpY2MAACiRdZHPKwRhGMc/u1bEag8cJIc5IIdVQnJkHVw2aVEWl5kxu6tm1zQzm+SqXByUg7j4dfAfcFWulFKkJEdnvy7SeF67tZvWO73zfPq+7/fpfb8vhJO2mfci/ZAv+G5qIqHNpee1hhci1BNliCbd9Jyxqakk/47PO0Kq3vapXv/vqzmalyzPhFCj8LDpuL7wqHBy1XcUbwm3mTl9SfhQOO7KAYWvlG6U+FlxtsTvit2Z1DiEVU8tW8VGFZs5Ny/cK9yVt4tm+TzqJlGrMDsttUNmJx4pJkigYVBkGRufPqkFyay2r//XN8mKeEz5O6zhiiNLTrxxUYvS1ZKaEd2Sz2ZN5f43Ty8zOFDqHk1A/VMQvHVDww58bwfB11EQfB9D3SNcFCr+Fclp5EP07YrWdQCxDTi7rGjGLpxvQvuDo7v6r1QnM5zJwOsptKSh9QaaFkpZldc5uYeZdXmia9jbhx7ZH1v8AcVWZ+8Oq3sSAAAACXBIWXMAAA9hAAAPYQGoP6dpAAAAKElEQVRIx2P8/+/ffwYY+P8fgpH5o/QoTSk9Ggaj9GjaHKVH6UGXRgGGtvwRQRE4UwAAAABJRU5ErkJggg=='
    decodedGradientImage = base64.b64decode(gradientImageBase64)
    with open('output/figure1/gradient.png', 'wb') as output_file:
        output_file.write(decodedGradientImage)
    print('Done. The output file has been written to: output/figure1/armadillo.obj')
    print()

def generateRadiusReplicationSettingsString(config):
    if config['replicationOverrides']['supportRadius']['recomputeEntirely']:
        return 'recompute entirely'
    elif config['replicationOverrides']['supportRadius']['recomputeSingleRadius']:
        selectedRadiusIndex = config['replicationOverrides']['supportRadius']['radiusIndexToRecompute']
        radiusMinValue = config['parameterSelection']['supportRadius']['radiusSearchStart']
        radiusStepValue = config['parameterSelection']['supportRadius']['radiusSearchStep']
        selectedRadius = str(radiusMinValue + float(selectedRadiusIndex) * radiusStepValue)
        return 'recompute statistics for radius ' + selectedRadius + ' only'
    else:
        return 'nothing is replicated'

allMethods = ['QUICCI', 'RICI', 'SI', 'RoPS', 'SHOT', 'USC']
originalExperiments = [
    ('additive-noise-only', 'Clutter'),
    ('subtractive-noise-only', 'Occlusion'),
    ('repeated-capture-only', 'Alternate triangulation'),
    ('normal-noise-only', 'Deviated normal vector'),
    ('support-radius-deviation-only', 'Deviated support radius'),
    ('gaussian-noise-only', 'Gaussian noise'),
    ('depth-camera-capture-only', 'Alternate mesh resolution'),
    ('additive-and-subtractive-noise', 'Clutter and Occlusion'),
    ('additive-and-gaussian-noise', 'Clutter and Gaussian noise'),
    ('subtractive-and-gaussian-noise', 'Occlusion and Gaussian noise')
]

def editSupportRadiusExtent(config):
    download_menu = TerminalMenu([
        "Recompute the support radius from scratch",
        "Replicate the statistics computed for one specific support radius",
        'back'],
        title='------------------ Support Radius Replication ------------------')

    choice = download_menu.show() + 1

    if choice == 1:
        config['replicationOverrides']['supportRadius']['recomputeEntirely'] = True
        config['replicationOverrides']['supportRadius']['recomputeSingleRadius'] = False
    if choice == 2:
        config['replicationOverrides']['supportRadius']['recomputeEntirely'] = False
        config['replicationOverrides']['supportRadius']['recomputeSingleRadius'] = True
        radiusSteps = config['parameterSelection']['supportRadius']['numberOfSupportRadiiToTry']
        radiusMinValue = config['parameterSelection']['supportRadius']['radiusSearchStart']
        radiusStepValue = config['parameterSelection']['supportRadius']['radiusSearchStep']
        print('The minimum, maximum, and average descriptor distances will be computed for a total of ' + str(radiusSteps) + ' radii.')
        print('These vary between {} and {}, in steps of {}.'.format(radiusMinValue, radiusMinValue + float(radiusSteps) * radiusStepValue, radiusStepValue))
        selectedRadius = input('Enter the index of the radius that should be replicated (integer between 0 and {}): '.format(radiusSteps))
        config['replicationOverrides']['supportRadius']['radiusIndexToRecompute'] = int(selectedRadius)
    print()
    return config

def replicateSupportRadiusFigures():
    config = readConfigFile()
    radiusConfigFile = 'cfg/config_support_radius_replication.json'
    if not os.path.isfile(radiusConfigFile):
        with open(radiusConfigFile, 'w') as outfile:
            json.dump(config, outfile, indent=4)

    while True:
        download_menu = TerminalMenu([
            'Select replication extent. Currently selected: ' + generateRadiusReplicationSettingsString(config)]
            + ['Run replication for method ' + x for x in allMethods] + [
             'Generate charts from precomputed support radius CSV files',
             'back'],
            title='------------------ Replicate Support Radius Figures ------------------')

        choice = download_menu.show() + 1

        if choice == 1:
            config = editSupportRadiusExtent(config)
            with open(radiusConfigFile, 'w') as outfile:
                json.dump(config, outfile, indent=4)
        if choice > 1 and choice <= len(allMethods) + 1:
            methodIndex = choice - 2
            methodName = allMethods[methodIndex]

            # Edit config file to only select the selected method
            with open(radiusConfigFile, 'r') as infile:
                config = json.load(infile)
            for method in allMethods:
                config['methodSettings'][method]['enabled'] = method == methodName
            for index, experiment in enumerate(originalExperiments):
                config['experimentsToRun'][index]['enabled'] = False
            with open(radiusConfigFile, 'w') as outfile:
                json.dump(config, outfile, indent=4)

            run_command_line_command('./shapebench --configuration-file=../cfg/config_support_radius_replication.json', 'bin')

            supportRadiusResultFiles = \
                ['support_radii_meanvariance_QUICCI_20240521-041001.txt',
                 'support_radii_meanvariance_RICI_20240520-235415.txt',
                 'support_radii_meanvariance_RoPS_20240521-173103.txt',
                 'support_radii_meanvariance_SHOT_20240531-200850.txt',
                 'support_radii_meanvariance_SI_20240522-033710.txt',
                 'support_radii_meanvariance_USC_20240529-135954.txt']

            print()
            print('Contents of the support radius file computed by the author:')
            print()
            run_command_line_command('cat precomputed_results/support_radius_estimation/' + supportRadiusResultFiles[methodIndex])
            print()
            print('You should compare the line(s) printed out by the replication run to the corresponding line in the file here.')
            print()

        if choice == len(allMethods) + 2:
            os.makedirs('output/charts', exist_ok=True)
            run_command_line_command('python3 tools/charter/charter.py '
                                     '--results-directory=precomputed_results/support_radius_estimation '
                                     '--output-dir=output/charts '
                                     '--mode=support-radius', '.')
            print()
            print('Charts created. You can find them in the output/charts directory.')
            print()
        if choice == len(allMethods) + 3:
            return

def runCharter():
    os.makedirs('output/charts', exist_ok=True)
    run_command_line_command('python3 tools/charter/charter.py --output-dir=output/charts --results-directory=precomputed_results')
    print()
    print('Charts created. You can find them in the output/charts directory.')
    print()

def replicateExperimentResults(figureIndex):
    config = readConfigFile()
    while True:
        print()
        print('Current replication settings:')
        print('- Experimental results:', generateReplicationSettingsString(config['replicationOverrides']['experiment']))
        print('- Replication of reference descriptor set:', generateReplicationSettingsString(config['replicationOverrides']['referenceDescriptorSet']))
        print('- Replication of sample object unfiltered descriptor set:', generateReplicationSettingsString(config['replicationOverrides']['sampleDescriptorSet']))
        print()
        replication_menu = TerminalMenu([
            'Edit replication settings (shortcut to same option in main menu)']
            + ['Subfigure ({}): {}'.format(list('abcdef')[index], method) for index, method in enumerate(allMethods)] + [
            "back"],
            title='------------------ Replicate Figure {}: {} ------------------'.format(7 + figureIndex, originalExperiments[figureIndex][1]))

        choice = replication_menu.show() + 1

        if choice == 1:
            changeReplicationSettings()

        if choice > 1 and choice < len(allMethods) + 2:
            methodIndex = choice - 2
            methodName = allMethods[methodIndex]
            precomputedResultsDir = os.path.join('precomputed_results', originalExperiments[figureIndex][0])
            resultFiles = [x for x in os.listdir(precomputedResultsDir) if methodName in x]
            if len(resultFiles) != 1:
                raise Exception('There should be exactly one result file for each method in the precomputed results directory. Found {}.'.format(len(resultFiles)))
            fileToReplicate = os.path.join(precomputedResultsDir, resultFiles[0])

            print()
            enableVisualisations = config['filterSettings']['additiveNoise']['enableDebugCamera']
            commandPreamble = 'xvfb-run ' if not enableVisualisations else ''
            run_command_line_command(commandPreamble + './shapebench --replicate-results-file=../{} --configuration-file=../cfg/config_replication.json'.format(fileToReplicate), 'bin')
            print('./shapebench --replicate-results-file=../{} --configuration-file=../cfg/config_replication.json'.format(fileToReplicate))
            print()
            print('Complete.')
            print('If you enabled any replication options in the settings, these have been successfully replicated if you did not receive a message about it, or the program has exited with an exception.')
            print('A comparison of the replicated benchmark results should be in a table that is visible a little bit above this message (you may need to scroll a bit)')
            print()
        if choice == 1 + len(allMethods) + 1:
            return

def replicateExperimentsFigures():
    experiments_menu = TerminalMenu([
        "Edit replication settings (shortcut to same option in main menu)"] +
        ['Replicate Figure {}: {}'.format(index + 7, x[1]) for index, x in enumerate(originalExperiments)]
        + ['Generate charts from precomputed results',
           'back'],
        title='------------------ Replicate Benchmark Results ------------------')
    while True:

        choice = experiments_menu.show() + 1
        if choice == 1:  #
            changeReplicationSettings()
        if choice > 1 and choice <= len(originalExperiments) + 1:
            replicateExperimentResults(choice - 2)
        if choice == len(originalExperiments) + 2:  #
            runCharter()
        if choice == len(originalExperiments) + 3:  #
            return

def runReplication():
    while True:
        menu = TerminalMenu([
            "1. Change replication settings",
            "2. Replicate Figure 1 - Similarity visualisation",
            "3. Replicate Figure 4 - Support radius estimation",
            "4. Replicate Figure 7 to 16 - Benchmark results for various filter configurations",
            "5. back"
        ], title='---------------------- Replication Menu ----------------------')

        choice = menu.show() + 1
        
        match choice:
            case 1:
                changeReplicationSettings()
            case 2:
                replicateSimilarityVisualisationFigure()
            case 3:
                replicateSupportRadiusFigures()
            case 4:
                replicateExperimentsFigures()
            case 5:
                return

trackExperiments = [
    ('', 'Occlusion'),
    ('', 'Clutter'),
    ('', 'Gaussian noise'),
    ('', 'Occlusion+Gaussian noise'),
    ('', 'Occlusion and Occlusion'),
    ('', 'Occlusion+fixed Gaussian noise and Occlusion+fixed Gaussian noise'),
    ('', 'Occlusion and Occlusion+Clutter'),
    ('', 'Occlusion+fixed Gaussian noise and Occlusion+Clutter+fixed Gaussian noise'),
    ('', 'Occlusion+less Clutter+fixed Gaussian+Alternate triangulation'),
]

# Run the experiment

def selectMethodsToRun():
    config = readConfigFile() # As a default it reads the config_replication.json
    methodList = list(config['methodSettings'].keys())
    
    while True:
        method_menu = TerminalMenu([f'{index}. {method}: {"enabled" if config["methodSettings"][method]["enabled"] else "disabled"}' for index, method in zip(list('12345678'), methodList)] + 
                                   ['back'],
                                   title='-' * 7 + f' Chose the method ' + '-' * 7)
        
        choice = method_menu.show() + 1
        
        match choice:
            case 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8:
                config['methodSettings'][methodList[choice - 1]]['enabled'] = False if config['methodSettings'][methodList[choice - 1]]['enabled'] else True
            case 9:
                with open('cfg/config_replication.json', 'w') as cfgFile:
                    json.dump(config, cfgFile, indent=4)
                return

def selectFilters(exp, fKey):
    filters = [
        ('additive-noise', 'Additive Noise'),
        ('subtractive-noise', 'Subtractive Noise'),
        ('normal-noise', 'Alternate Triangulation'),
        ('repeated-capture', 'Normal Vector Noise'),
        ('support-radius-deviation', 'Support Radius Deviation'),
        ('depth-camera-capture', 'Depth Camera Capture'),
        ('gaussian-noise', 'Gaussian Noise')
    ]
    addedFilters = []
    
    while True:
        filter_menu = TerminalMenu([f'{index + 1}. {fName[1]}' for index, fName in enumerate(filters)] + 
                                   ['continue'], title='-' * 7 + ' Select one or more filters ' + '-' * 7)
        
        choice = filter_menu.show() + 1
        
        if choice > 0 and choice <= len(filters):
            if fKey not in exp.keys():
                exp[fKey] = [{'type': filters[choice - 1][0]}]
                addedFilters = [filters[choice - 1][1]]
            else:
                if filters[choice - 1][1] not in addedFilters:
                    exp[fKey].append({'type': filters[choice - 1][0]})
                    addedFilters.append(filters[choice - 1][1])
                else:
                    continue            
        else:
            return exp

def createNewExperiment(config_file_to_edit):
    newExperiment = {
                    "enabled": True,
                }
    
    while True:
        create_menu = TerminalMenu([
            'Step 1: Enter a name for the experiment (without spaces)',
            'Step 2: Chose the filters for the scene',
            'Step 3: Chose the filters for the model',
            'save experiment',
            'cancel'
        ], title='-' * 5 + ' Create a custom experiment ' + '-' * 5)
        
        choice = create_menu.show() + 1
        
        match choice:
            case 1:
                experimentName = ''
                checkName = False
                while not checkName:
                    experimentName = input('Name: ')
                    checkName = True if ' ' not in experimentName else print('/' * 10 + ' INVALID NAME ' + '\\' * 10)

                newExperiment['name'] = experimentName
            case 2:
                newExperiment = selectFilters(newExperiment, 'filters')
            case 3:
                newExperiment = selectFilters(newExperiment, 'modelFilters')
            case 4:
                if 'name' not in newExperiment.keys():
                    print('Please insert a valid name fot the experiment')
                elif 'filters' not in newExperiment.keys() and 'modelFilters' not in newExperiment.keys():
                    print('Specify at least one filter')
                else:
                    config = readConfigFile(config_file_to_edit)
                    config['experimentsToRun'].append(newExperiment)
                    
                    writeConfigFile(config, config_file_to_edit)

                    #TODO pass this down as parameter
                    base_config_file = config_file_to_edit.replace('_run', '_base')
                    
                    configBase = readConfigFile(base_config_file)
                    newExperiment['enabled'] = False
                    configBase['experimentsToRun'].append(newExperiment)
                    writeConfigFile(configBase, base_config_file)
                    
                    return
            case 5:
                return

def listExperimets(config_file_to_edit, expList):
    config = readConfigFile(config_file_to_edit)
    allExperimentsName = [experiment['name'] for experiment in config['experimentsToRun']]
    
    while True:
        subMenu = []
        for index, exp in enumerate(expList):
            posExp = allExperimentsName.index(exp)
            subMenu.append(f"{index + 1}. {exp}: {'enabled' if config['experimentsToRun'][posExp]['enabled'] else 'disabled'}")
        
        menu = TerminalMenu(subMenu + 
                            ['back'], title='-' * 10 + ' Enable experiments ' + '-' * 10)
        
        choice = menu.show() + 1
        
        if choice > 0 and choice <= len(expList):
            expIndex = allExperimentsName.index(expList[choice - 1])
            config['experimentsToRun'][expIndex]['enabled'] = False if config['experimentsToRun'][expIndex]['enabled'] else True
        else:
            writeConfigFile(config, config_file_to_edit)
            return

def enableExperimentsToRun(config_file_to_edit):
    # check if there are custom experiments
    
    #originalExperimentsName = [experiment[0] for experiment in originalExperiments]
    #trackExperimentsName = [experiment[0] for experiment in trackExperiments]
    #defaultsExperiments = originalExperimentsName + trackExperimentsName

    while True:
        config = readConfigFile(config_file_to_edit)
        customExperiments = [experiment['name'] for experiment in config['experimentsToRun']]
        
        experiment_menu = TerminalMenu([
            'Experiments List',
            'Define a new experiment',
            'back'
        ], title='-' * 10 + ' Experiment selection ' + '-' * 10)
    
        choice = experiment_menu.show() + 1

        match choice:
            case 1:
                listExperimets(config_file_to_edit, customExperiments)
            case 2:
                createNewExperiment(config_file_to_edit) #DONE
            case 3:
                return

def saveToFile(config):
    saveFiles = [fileName for fileName in os.listdir('cfg') if 'run' in fileName]
    root = 'cfg'
    
    while True:
        menu = TerminalMenu([f"{index + 1}. {fileName}" for index, fileName in enumerate(saveFiles)] + ['Create new file'] + ['back'],
                            title='-' * 5 + ' Select a file to save the configuration' +  '-' * 5)
        
        choice = menu.show() + 1
        
        if choice > 0 and choice <= len(saveFiles):
            savePath = os.path.join(root, saveFiles[choice - 1])
            
            with open(savePath, 'w') as f:
                json.dump(config, f, indent=4)
            
            return
        elif choice == len(saveFiles) + 1:
            fileName = input('Insert the file name:\n')
            
            if '.json' not in fileName:
                fileName += '.json'
            
            if 'run' not in fileName:
                tmpFileName = fileName.split('.')
                fileName = tmpFileName[0] + '_run.' + tmpFileName[1]
            
            savePath = os.path.join(root, fileName)
            
            with open(savePath, 'w') as f:
                json.dump(config, f, indent=4)
            
            return
        
        else:
            return
            
def runExperiments(config_file_to_edit):

    while True:
        menu = TerminalMenu([
            'Run configuration',
            'Edit benchmark settings',
            'Enable or disable experiments',
            'Enable or disable methods to test',
            'Back'
        ], title='-' * 10 + ' Run the benchmark ' + '-' * 10)

        choice = menu.show() + 1

        match choice:
            case 1:
                config = readConfigFile(config_file_to_edit)
                enableVisualisations = config['filterSettings']['additiveNoise']['enableDebugCamera']
                commandPreamble = 'xvfb-run ' if not enableVisualisations else ''

                # I think this should be a separate feature, maybe have it be its own option in the menu
                # The configuration file needs to be saved anyway because otherwise the benchmark cannot run it
                #saveToFile(config)

                print('Now running...')
                run_command_line_command(
                    commandPreamble + './shapebench --configuration-file=../{}'.format(config_file_to_edit), 'bin')

            case 2:
                changeReplicationSettings()  # DONE
            case 3:
                enableExperimentsToRun(config_file_to_edit)
            case 4:
                selectMethodsToRun(config_file_to_edit)  # DONE
            case 5:
                return
    
def runMainMenu(config_file_to_edit):
    config = readConfigFile(config_file_to_edit)
    intendedForReplication = config['intendedForReplication'] if 'intendedForReplication' in config else False
    runOption = '4. Replicate results and experiments' if intendedForReplication else '4. Run experiments'

    while True:
        main_menu = TerminalMenu([
            "1. Install dependencies",
            "2. Download Author computed results and cache files",
            "3. Compile project",
            runOption,
            "5. Exit"], title='---------------------- Main Menu ----------------------')

        choice = main_menu.show() + 1
        
        match choice:
            case 1:
                installDependencies()
            case 2:
                downloadDatasetsMenu()
            case 3:
                compileProject()
            case 4:
                if intendedForReplication:
                    runReplication()
                else:
                    runExperiments(config_file_to_edit)
            case 5:
                return


def computeConfigFileDisplayName(config_directory, config_file_name):
    with open(os.path.join(config_directory, config_file_name), 'r') as f:
        fileContents = json.loads(f.read())
    return tuple((os.path.join(config_directory, config_file_name), config_file_name + (": " + fileContents['description'] if 'description' in fileContents else "")))

def runIntroSequence():
    print()
    print('Greetings!')
    print()
    print('This script is intended to assist with replicating figures from previous papers,')
    print('as well as running the benchmark to produce new ones.')
    print()
    print('If you have not run this script before, you should run step 1 to 3 in order.')
    print('More details can be found in the replication manual PDF file that accompanies this script.')
    print()
    print('The script automatically edits benchmark configuration files.')
    print('These are configuration files in the \'cfg\' directory whose name ends with _base.json')
    print('Please select the configuration file which you would like to use for benchmark runs.')
    print()

    # Patching in absolute paths
    config = None
    
    loadConfig = True
    while loadConfig:
        configDirectory = 'cfg'
        allConfigs = [computeConfigFileDisplayName(configDirectory, fileName) for fileName in os.listdir(configDirectory) if '_base' in fileName]
        loadingMenu = TerminalMenu([f"{index + 1}. {fileName[1]}" for index, fileName in enumerate(allConfigs)],
                                   title='-' * 7 + ' Select the config file to use for this session ' +  '-' * 7)
        choice = loadingMenu.show() + 1

        config_file_to_edit = allConfigs[choice - 1][0]
        print('Selected config file:', config_file_to_edit)
        config = readConfigFile(config_file_to_edit)
        run_configuration_file = config_file_to_edit.replace('_base', '_run')
        print('Saving configuration to', run_configuration_file)
        print()
        if (os.path.isfile(run_configuration_file)):
            print('A configuration file for a previous run with this base configuration was found:')
            print('   ', run_configuration_file)
            print('Would you like to continue where you left off with this file?')
            print('It will otherwise be overwritten with the base configuration.')
            print()
            choice = ask_for_confirmation('Continue from previous run?')
            if choice == True:
                config = readConfigFile(run_configuration_file)
            
        writeConfigFile(config, run_configuration_file)

        loadConfig = False
    
    config['cacheDirectory'] = os.path.abspath(config['cacheDirectory'])
    config['resultsDirectory'] = os.path.abspath(config['resultsDirectory'])
    config['datasetSettings']['compressedRootDir'] = os.path.abspath(config['datasetSettings']['compressedRootDir'])
    config['datasetSettings']['objaverseRootDir'] = os.path.abspath(config['datasetSettings']['objaverseRootDir'])
    writeConfigFile(config)

    runMainMenu(run_configuration_file)

if __name__ == "__main__":
    runIntroSequence()
