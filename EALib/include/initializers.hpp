//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once
#include "base.hpp"

/**
 * Initializes each solution independently and uniformly over the alphabet
 */
class CategoricalUniformInitializer : public ISolutionInitializer
{
  public:
    void initialize(std::vector<Individual> &iis) override;
    void afterRegisterData() override;
};

/**
 * Initializes all solutions such that each gene occurs in (approximately) equal counts
 *
 * A difference of at most +/- 1 is possible, and occurs when the alphabet size does
 * not nicely divide the number of individuals.
 */
class CategoricalProbabilisticallyCompleteInitializer : public ISolutionInitializer
{
  public:
    void initialize(std::vector<Individual> &iis) override;
    void afterRegisterData() override;
};