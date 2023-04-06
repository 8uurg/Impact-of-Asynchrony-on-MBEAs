#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>
#include "base.hpp"

class MockGenerationalApproach : public trompeloeil::mock_interface<GenerationalApproach>
{
  public:
    IMPLEMENT_MOCK0(step);
    IMPLEMENT_MOCK0(getSolutionPopulation);
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};

class MockObjectiveFunction : public trompeloeil::mock_interface<ObjectiveFunction>
{
  public:
    IMPLEMENT_MOCK1(evaluate);
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};

class MockAcceptanceCriterion : public trompeloeil::mock_interface<IPerformanceCriterion>
{
  public:
    IMPLEMENT_MOCK2(compare);
    IMPLEMENT_MOCK1(setPopulation);
    IMPLEMENT_MOCK0(registerData);
    IMPLEMENT_MOCK0(afterRegisterData);
};

class MockGenerationalApproachComparator : public trompeloeil::mock_interface<GenerationalApproachComparator>
{
  public:
    IMPLEMENT_MOCK0(clear);
    IMPLEMENT_MOCK2(compare);
};