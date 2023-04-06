#pragma once
// This file contains an interface & corresponding implementations for multi-objective archives.

#include <algorithm>

#include "base.hpp"
#include "logging.hpp"

struct Archived
{
    bool added;
    bool dominated;
    size_t ordinal;
    std::vector<size_t> ordinals_removed;
};

class IArchive : public IDataUser
{
  public:
    virtual Archived try_add(Individual candidate) = 0;

    virtual std::vector<Individual> &get_archived() = 0;
};

class BruteforceArchive : public IArchive
{
  public:
    BruteforceArchive(std::vector<size_t> objective_indices);

    Archived try_add(Individual candidate) override;

    std::vector<Individual> &get_archived() override;

    void do_cache();

    void setPopulation(std::shared_ptr<Population> population) override;

  private:
    struct Cache
    {
        TypedGetter<Objective> og;
    };
    std::optional<Cache> cache;

    std::vector<Individual> archive;
    const std::vector<size_t> objective_indices;

    // Insertion & removal bookkeeping for logs.
    std::vector<size_t> archive_ord;
    size_t ordinal = 0;
};

class ArchivedLogger : public ItemLogger
{
  private:
    std::optional<std::reference_wrapper<Archived>> archived;

  public:
    static std::shared_ptr<ArchivedLogger> shared();

    void header(IMapper &mapper);
    void log(IMapper &mapper, const Individual &i);

    void setArchived(Archived &archived);
};

/**
 * @brief A wrapper that performs logging calls upon successful insertion around any archive .
 */
class LoggingArchive : public IArchive
{
  private:
    std::shared_ptr<IArchive> archive;
    std::shared_ptr<BaseLogger> logger;
    std::optional<std::shared_ptr<ArchivedLogger>> archive_log;

  public:
    LoggingArchive(std::shared_ptr<IArchive> archive,
                   std::shared_ptr<BaseLogger> logger,
                   std::optional<std::shared_ptr<ArchivedLogger>> archive_log = std::nullopt);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    Archived try_add(Individual candidate) override;

    std::vector<Individual> &get_archived() override;
};

/**
 * @brief A wrapper that tracks the point at which the last successful insertion occurred.
 */
class ImprovementTrackingArchive : public IArchive
{
  private:
    std::shared_ptr<IArchive> archive;
    std::optional<double> t;
    std::function<double()> get_current_t_w;
    double base;
    double factor;

  public:
    ImprovementTrackingArchive(std::shared_ptr<IArchive> archive, std::function<double()> get_current_t, double base, double factor);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    Archived try_add(Individual candidate) override;

    std::optional<double> get_t();
    double get_current_t();

    std::vector<Individual> &get_archived() override;
};
