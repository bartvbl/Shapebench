import json
import os
import shutil
import subprocess
import random
import sys
import multiprocessing
import hashlib
from scripts.simple_term_menu import TerminalMenu
from scripts.prettytable import PrettyTable


if not (sys.version_info.major == 3 and sys.version_info.minor >= 8):
    print("This script requires Python 3.8 or higher.")
    print("You are using Python {}.{}.".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

def run_command_line_command(command, working_directory='.'):
    print('>> Executing command:', command)
    subprocess.run(command, shell=True, check=False, cwd=working_directory)

def ask_for_confirmation(message):
    confirmation_menu = TerminalMenu(["yes", "no"], title=message)
    choice = confirmation_menu.show()
    return choice == 0

def downloadFile(fileURL, tempFile, extractInDirectory, name):
    if not os.path.isfile('input/download/' + tempFile) or ask_for_confirmation('It appears the ' + name + ' archive file has already been downloaded. Would you like to download it again?'):
        print('Downloading the ' + name + ' archive file..')
        run_command_line_command('wget --output-document ' + tempFile + ' ' + fileURL, 'input/download/')
    print()
    os.makedirs(extractInDirectory, exist_ok=True)
    run_command_line_command('p7zip -k -d ' + os.path.join(os.path.relpath('input/download', extractInDirectory), tempFile), extractInDirectory)
    #if ask_for_confirmation('Download and extraction complete. Would you like to delete the compressed archive to save disk space?'):
    #    os.remove('input/download/' + tempFile)
    print()

def downloadDatasetsMenu():
    download_menu = TerminalMenu([
        "Download all",
        "Download computed results (XXX MB download, XXX GB uncompressed)",
        "Download cache files (XXX MB download, XXX GB uncompressed)",
        "back"], title='------------------ Download Datasets ------------------')

    while True:
        choice = download_menu.show() + 1
        os.makedirs('input/download/', exist_ok=True)

        if choice == 1 or choice == 2:
           downloadFile('https://ntnu.box.com/shared/static/zb2co430vdcpao7gwco3vaxsf7ahz09u.7z', 'SHREC2016.7z',
                        'input', 'SHREC 2016 Partial Retrieval Dataset')
        if choice == 1 or choice == 3:
            downloadFile('https://ntnu.box.com/shared/static/y29rpwz5n9dj6ljaghr40uj34njs3s4j.7z',
                         'precomputed_descriptors.7z', 'input/', 'Precomputed QUICCI Descriptors')
        if choice == 4:
            return

def installDependencies():
    run_command_line_command('sudo apt install cmake g++ gcc build-essential wget')
    run_command_line_command('pip3 install simple-term-menu xlwt xlrd numpy matplotlib pillow PyQt5 wcwidth')

def compileProject():
    os.makedirs('bin', exist_ok=True)
    run_command_line_command('rm -rf bin/*')

    run_command_line_command('cmake .. -DCMAKE_BUILD_TYPE=Release -G Ninja', 'bin')
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


def changeReplicationSettings():
    with open('cfg/config_replication.json', 'r') as cfgFile:
        config = json.load(cfgFile)

    while True:
        download_menu = TerminalMenu([
            'Print individual experiment results: ' + ('enabled' if config['verboseOutput'] else 'disabled'),
            'Random seed used when selecting random subsets to replicate: ' + str(config['replicationOverrides']['replicationRandomSeed']),
            'Verify computed minimum bounding sphere of input objects: ' + ('enabled' if config['datasetSettings']['verifyFileIntegrity'] else 'disabled'),
            'Size of dataset file cache in GB: ' + str(config['datasetSettings']['cacheSizeLimitGB']),
            'Replication of reference descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['referenceDescriptorSet']),
            'Replication of sample object unfiltered descriptor set: ' + generateReplicationSettingsString(config['replicationOverrides']['sampleDescriptorSet']),
            'Replication of experiment results: ' + generateReplicationSettingsString(config['replicationOverrides']['experiment']),
            "back"], title='------------------ Configure Replication ------------------')

        choice = download_menu.show() + 1

        if choice == 1:
            config['verboseOutput'] = not config['verboseOutput']
        if choice == 2:
            config['replicationOverrides']['replicationRandomSeed'] = selectReplicationRandomSeed(config['replicationOverrides']['replicationRandomSeed'])
        if choice == 3:
            config['datasetSettings']['verifyFileIntegrity'] = not config['datasetSettings']['verifyFileIntegrity']
        if choice == 4:
            print()
            newSize = int(input('Size of dataset file cache in GB: '))
            config['datasetSettings']['cacheSizeLimitGB'] = newSize
            print()
        if choice == 5:
            config['replicationOverrides']['referenceDescriptorSet'] = editSettings(config['replicationOverrides']['referenceDescriptorSet'], 'Reference Descriptor Set')
        if choice == 6:
            config['replicationOverrides']['sampleDescriptorSet'] = editSettings(config['replicationOverrides']['sampleDescriptorSet'], 'Sample Object Unfiltered Descriptor Set')
        if choice == 7:
            config['replicationOverrides']['experiment'] = editSettings(config['replicationOverrides']['experiment'], 'Experiment Results')
        if choice == 8:
            with open('cfg/config_replication.json', 'w') as cfgFile:
                json.dump(config, cfgFile, indent=4)
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

        if choice == 1:  #
            installDependencies()
        if choice == 2:  #
            downloadDatasetsMenu()
        if choice == 3:  #
            compileProject()
        if choice == 4:  #
            changeReplicationSettings()
        if choice == 5:
            replicateSimilarityVisualisationFigure()
        if choice == 6:  #
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
    print('It is recommended you refer to the included PDF manual for instructions')
    print()
    runMainMenu()


if __name__ == "__main__":
    runIntroSequence()
