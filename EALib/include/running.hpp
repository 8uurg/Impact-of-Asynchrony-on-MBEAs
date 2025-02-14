//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include "base.hpp"
#include <memory>

class IRunner : public IDataUser
{
  public:
    virtual void run() = 0;
    virtual void step() = 0;
};

// Simply set ups a single instance of a generational approach and steps it until termination.
class TerminationStepper : public IRunner
{
  public:
    TerminationStepper(std::function<std::shared_ptr<GenerationalApproach>()> approach,
                       std::optional<int> step_limit = std::nullopt,
                       bool verbose=false);

    void run() override;
    void step() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

  private:
    std::shared_ptr<GenerationalApproach> approach;
    std::optional<int> step_limit;
    int steps = 0;
    bool verbose;
    bool terminated = false;
};

class InterleavedMultistartScheme : public IRunner
{
  public:
    InterleavedMultistartScheme(std::function<std::shared_ptr<GenerationalApproach>(size_t)> approach_factory,
                                std::shared_ptr<GenerationalApproachComparator> approach_comparator,
                                size_t steps = 4,
                                size_t base = 4,
                                size_t multiplier = 2);

    void run() override;
    void step() override;
    void runRecursiveFold(size_t end_index);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

  private:
    void updateMinimumIndex();

    size_t minimum_index = 0;
    size_t steps;
    size_t current;
    size_t multiplier;
    bool terminated = false;
    const std::function<std::shared_ptr<GenerationalApproach>(size_t)> approach_factory;
    const std::shared_ptr<GenerationalApproachComparator> approach_comparator;
    std::vector<std::shared_ptr<GenerationalApproach>> approaches;
};

class SimpleConfigurator
{
  public:
    SimpleConfigurator(std::shared_ptr<ObjectiveFunction> objective,
                       std::shared_ptr<IRunner> runner,
                       std::optional<size_t> seed);

    void run();
    void step();

    Population &getPopulation()
    {
        return *population;
    };

  private:
    std::shared_ptr<ObjectiveFunction> objective;
    std::shared_ptr<IRunner> runner;
    std::shared_ptr<Population> population;
    Rng rng;
};
