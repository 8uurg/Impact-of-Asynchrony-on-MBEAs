//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include <filesystem>
#include <fstream>

#include "base.hpp"
#include "utilities.hpp"


// -- Item Loggers --

/**
 * @brief 
 * 
 */
class ItemLogger : public IDataUser
{
  public:
    virtual void header(IMapper &mapper) = 0;
    virtual void log(IMapper &mapper, const Individual &i) = 0;
};

class SequencedItemLogger : public ItemLogger
{
  private:
    std::vector<std::shared_ptr<ItemLogger>> subloggers;

  public:
    SequencedItemLogger(std::vector<std::shared_ptr<ItemLogger>> subloggers);
    static std::shared_ptr<SequencedItemLogger> shared(std::vector<std::shared_ptr<ItemLogger>> subloggers);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void header(IMapper &mapper) override;
    void log(IMapper &mapper, const Individual &i) override;
};

class GenotypeCategoricalLogger : public ItemLogger
{
  private:
    std::optional<TypedGetter<GenotypeCategorical>> tggc;

  public:
    static std::shared_ptr<GenotypeCategoricalLogger> shared();

    void afterRegisterData() override;

    void header(IMapper &mapper) override;
    void log(IMapper &mapper, const Individual &i) override;
};

class ObjectiveLogger : public ItemLogger
{
  private:
    std::optional<TypedGetter<Objective>> tgo;

  public:
    static std::shared_ptr<ObjectiveLogger> shared();

    void afterRegisterData() override;

    void header(IMapper &mapper) override;
    void log(IMapper &mapper, const Individual &i) override;
};

class NumEvaluationsLogger : public ItemLogger
{
  private:
    std::shared_ptr<Limiter> limiter;

  public:
    NumEvaluationsLogger(std::shared_ptr<Limiter> limiter);
    static std::shared_ptr<NumEvaluationsLogger> shared(std::shared_ptr<Limiter> limiter);
    void header(IMapper &mapper) override;
    void log(IMapper &mapper, const Individual &) override;
};

class WallTimeLogger : public ItemLogger
{
  private:
    std::shared_ptr<Limiter> limiter;

  public:
    WallTimeLogger(std::shared_ptr<Limiter> limiter);
    static std::shared_ptr<WallTimeLogger> shared(std::shared_ptr<Limiter> limiter);

    void header(IMapper &mapper) override;
    void log(IMapper &mapper, const Individual &) override;
};

class SolutionIndexLogger : public ItemLogger
{
  public:
    static std::shared_ptr<SolutionIndexLogger> shared();

    void header(IMapper &mapper) override;
    void log(IMapper &mapper, const Individual &i) override;
};

// -- Base Loggers --
// The actual string to file (or somewhere else!) adapters

/**
 * @brief 
 * 
 */
class BaseLogger : public IDataUser
{
  public:
    virtual void log(const Individual &i) = 0;
};

class CSVLogger : public BaseLogger
{
  private:
    std::filesystem::path out_path;
    std::shared_ptr<ItemLogger> item_logger;

  public:
    CSVLogger(std::filesystem::path out_path, std::shared_ptr<ItemLogger> item_logger);
    static std::shared_ptr<CSVLogger> shared(std::filesystem::path out_path, std::shared_ptr<ItemLogger> item_logger);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void header();
    void log(const Individual &i) override;
};

