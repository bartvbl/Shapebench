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
            downloadFile('https://ntnu.box.com/shared/static/1oo864m02zj9itdptzbwvvj04epigyio.7z', 'cache.7z',
                         'cache', 'Precomputed cache files')
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

def writeConfigFile(config):
    with open('cfg/config_replication.json', 'w') as cfgFile:
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
            'Replication of experimental results: ' + generateReplicationSettingsString(config['replicationOverrides']['experiment']),
            'Replication of reference descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['referenceDescriptorSet']),
            'Replication of sample object unfiltered descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['sampleDescriptorSet']),
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
allExperiments = [
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
            for index, experiment in enumerate(allExperiments):
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
            title='------------------ Replicate Figure {}: {} ------------------'.format(7 + figureIndex, allExperiments[figureIndex][1]))

        choice = replication_menu.show() + 1

        if choice == 1:
            changeReplicationSettings()

        if choice > 1 and choice < len(allMethods) + 2:
            methodIndex = choice - 2
            methodName = allMethods[methodIndex]
            precomputedResultsDir = os.path.join('precomputed_results', allExperiments[figureIndex][0])
            resultFiles = [x for x in os.listdir(precomputedResultsDir) if methodName in x]
            if len(resultFiles) != 1:
                raise Exception('There should be exactly one result file for each method in the precomputed results directory. Found {}.'.format(len(resultFiles)))
            fileToReplicate = os.path.join(precomputedResultsDir, resultFiles[0])

            print()
            enableVisualisations = config['filterSettings']['additiveNoise']['enableDebugCamera']
            commandPreamble = 'xvfb-run ' if not enableVisualisations else ''
            run_command_line_command(commandPreamble + './shapebench --replicate-results-file=../{} --configuration-file=../cfg/config_replication.json'.format(fileToReplicate), 'bin')
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
        ['Replicate Figure {}: {}'.format(index + 7, x[1]) for index, x in enumerate(allExperiments)]
        + ['Generate charts from precomputed results',
           'back'],
        title='------------------ Replicate Benchmark Results ------------------')
    while True:


        choice = experiments_menu.show() + 1
        if choice == 1:  #
            changeReplicationSettings()
        if choice > 1 and choice <= len(allExperiments) + 1:
            replicateExperimentResults(choice - 2)
        if choice == len(allExperiments) + 2:  #
            runCharter()
        if choice == len(allExperiments) + 3:  #
            return


def runMainMenu():
    while True:
        main_menu = TerminalMenu([
            "1. Install dependencies",
            "2. Download Author computed results and cache files",
            "3. Compile project",
            "4. Change replication settings",
            "5. Replicate Figure 1 - Similarity visualisation",
            "6. Replicate Figure 4 - Support radius estimation",
            "7. Replicate Figure 7 to 16 - Benchmark results for various filter configurations",
            "8. exit"], title='---------------------- Main Menu ----------------------')

        choice = main_menu.show() + 1

        if choice == 1:  # Done
            installDependencies()
        if choice == 2:  # Done
            downloadDatasetsMenu()
        if choice == 3:  # Done
            compileProject()
        if choice == 4:  # Done
            changeReplicationSettings()
        if choice == 5:  # Done
            replicateSimilarityVisualisationFigure()
        if choice == 6:  # Done
            replicateSupportRadiusFigures()
        if choice == 7:  #
            replicateExperimentsFigures()
        if choice == 8:  #
            return

def runIntroSequence():
    print()
    print('Greetings!')
    print()
    print('This script is intended to reproduce various figures in an interactive')
    print('(and hopefully convenient) manner.')
    print()
    print('If you have not run this script before, you should run step 1 to 3 in order.')
    print('More details can be found in the replication manual PDF file that accompanies this script.')
    print()

    # Patching in absolute paths
    config = readConfigFile('cfg/config_replication_base.json')
    config['cacheDirectory'] = os.path.abspath(config['cacheDirectory'])
    config['resultsDirectory'] = os.path.abspath(config['resultsDirectory'])
    config['datasetSettings']['compressedRootDir'] = os.path.abspath(config['datasetSettings']['compressedRootDir'])
    config['datasetSettings']['objaverseRootDir'] = os.path.abspath(config['datasetSettings']['objaverseRootDir'])
    writeConfigFile(config)

    runMainMenu()


if __name__ == "__main__":
    runIntroSequence()
