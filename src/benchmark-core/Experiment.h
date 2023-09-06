#pragma once

#include <string>
#include "ExperimentConfiguration.h"

class Experiment {
    virtual std::string name() = 0;
    virtual void run(const ExperimentConfiguration &configuration) = 0;

};