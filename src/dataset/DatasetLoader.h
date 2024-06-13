#pragma once

#include "Dataset.h"
#include "benchmarkCore/BenchmarkConfiguration.h"

namespace ShapeBench {
    ShapeBench::Dataset computeOrLoadCache(const ShapeBench::BenchmarkConfiguration& setup);
}