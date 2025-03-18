#pragma once

#include "ExperimentResult.h"
#include <unordered_set>
#include "benchmarkCore/BenchmarkConfiguration.h"

void writeExperimentResults(const ShapeBench::ExperimentResult& results, std::filesystem::path outputDirectory, bool isFinalResult, bool isPRCEnabled, const ShapeBench::ReplicationSettings& areReplicatedResults);