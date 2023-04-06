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