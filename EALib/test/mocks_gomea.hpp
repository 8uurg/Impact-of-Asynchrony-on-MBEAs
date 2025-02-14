//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

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