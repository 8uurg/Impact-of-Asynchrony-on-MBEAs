#pragma once

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>
#include "gomea.hpp"
#include "trompeloeil.hpp"

class MockSamplingDistribution : public trompeloeil::mock_interface<ISamplingDistribution>
{
  public:
    IMPLEMENT_MOCK2(apply_resample);
};