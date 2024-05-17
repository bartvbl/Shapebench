#pragma once

#include "ExperimentResult.h"

void writeExperimentResults(const ShapeBench::ExperimentResult& results, std::filesystem::path outputDirectory, bool isFinalResult, bool isPRCEnabled);