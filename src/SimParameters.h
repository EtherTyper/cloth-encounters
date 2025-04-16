#ifndef SIMPARAMETERS_H
#define SIMPARAMETERS_H

struct SimParameters
{
    SimParameters()
    {
        timeStep = 1e-2;
        constraintIters = 5;

        gravityEnabled = true;
        gravityG = -9.8;

        pinEnabled = true;
        pinWeight = 1.0;

        stretchEnabled = true;
        stretchWeight = 0.5;

        bendingEnabled = true;
        bendingWeight = 0.5;

        pullingEnabled = true;
        pullingWeight = 0.5;
    }

    double timeStep;
    int constraintIters;

    bool gravityEnabled;
    double gravityG;
    bool pinEnabled;
    double pinWeight;

    bool stretchEnabled;
    double stretchWeight;

    bool bendingEnabled;
    double bendingWeight;

    bool pullingEnabled;
    double pullingWeight;
};

#endif
