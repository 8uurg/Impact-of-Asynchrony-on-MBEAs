//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "archive.hpp"
#include "base.hpp"

// BruteforceArchive
BruteforceArchive::BruteforceArchive(std::vector<size_t> objective_indices) : objective_indices(objective_indices)
{
}

Archived BruteforceArchive::try_add(Individual candidate)
{
    do_cache();
    Population &population = *this->population;

    bool dominated = false;
    bool added = true;
    std::vector<size_t> ordinals_removed;

    Objective &candidate_o = cache->og.getData(candidate);

    for (size_t archive_idx = 0; archive_idx < archive.size(); ++archive_idx)
    {
        Individual &i = archive[archive_idx];
        bool i_has_better_objective_value = false;
        bool candidate_has_better_objective_value = false;

        Objective &i_o = cache->og.getData(i);
        for (size_t io = 0; io < objective_indices.size(); ++io)
        {
            size_t o = objective_indices[io];
            if (i_o.objectives[o] < candidate_o.objectives[o])
                i_has_better_objective_value = true;
            if (i_o.objectives[o] > candidate_o.objectives[o])
                candidate_has_better_objective_value = true;
        }

        if (!i_has_better_objective_value && !candidate_has_better_objective_value)
        {
            // Solutions are equal
            added = false;
            break;
        }
        else if (i_has_better_objective_value && candidate_has_better_objective_value)
        {
            // Solution is not dominated, but does not dominate the other either.
        }
        else if (i_has_better_objective_value)
        {
            // Candidate is dominated -- we can stop.
            dominated = true;
            added = false;
            break;
        }
        else if (candidate_has_better_objective_value)
        {
            // Candidate dominates i -- i can be removed.
            std::swap(archive[archive_idx], archive.back());
            std::swap(archive_ord[archive_idx], archive_ord.back());
            population.dropIndividual(archive.back());
            ordinals_removed.push_back(archive_ord.back());
            archive.pop_back();
            archive_ord.pop_back();

            // Current index is now a new element, so revert a step :)
            archive_idx--;
        }
    }

    if (added)
    {
        Individual archived_candidate = population.newIndividual();
        population.copyIndividual(candidate, archived_candidate);
        archive_ord.push_back(++ordinal);
        archive.push_back(archived_candidate);
    }

    return Archived{added, dominated, ordinal, std::move(ordinals_removed)};
}

std::vector<Individual> &BruteforceArchive::get_archived()
{
    return archive;
}

void BruteforceArchive::do_cache()
{
    if (!cache.has_value())
    {
        Population &population = *this->population;
        auto og = population.getDataContainer<Objective>();
        cache.emplace(Cache{og});
    }
}

// ArchivedLogger
std::shared_ptr<ArchivedLogger> ArchivedLogger::shared()
{
    return std::make_shared<ArchivedLogger>();
}

void ArchivedLogger::header(IMapper &mapper)
{
    mapper << "archive ordinal"
           << "archive ordinals removed";
}
void ArchivedLogger::log(IMapper &mapper, const Individual &)
{
    if (!archived.has_value())
    {
        mapper << "missing"
               << "missing";
        return;
    }
    auto &archived_v = archived->get();
    mapper << std::to_string(archived_v.ordinal);

    std::stringstream ss;
    for (size_t idx = 0; idx < archived_v.ordinals_removed.size(); ++idx)
    {
        if (idx != 0)
            ss << " ";
        ss << archived_v.ordinals_removed[idx];
    }
    mapper << ss.str();

    archived.reset();
}
void ArchivedLogger::setArchived(Archived &archived)
{
    this->archived.emplace(archived);
}

// LoggingArchive
LoggingArchive::LoggingArchive(std::shared_ptr<IArchive> archive,
                               std::shared_ptr<BaseLogger> logger,
                               std::optional<std::shared_ptr<ArchivedLogger>> archive_log) :
    archive(archive), logger(logger), archive_log(archive_log)
{
}
void LoggingArchive::setPopulation(std::shared_ptr<Population> population)
{
    archive->setPopulation(population);
    logger->setPopulation(population);
}
void LoggingArchive::registerData()
{
    archive->registerData();
    logger->registerData();
}
void LoggingArchive::afterRegisterData()
{
    archive->afterRegisterData();
    logger->afterRegisterData();
}
Archived LoggingArchive::try_add(Individual candidate)
{
    Archived a = archive->try_add(candidate);

    if (a.added)
    {
        if (archive_log.has_value())
        {
            auto &al = *archive_log;
            al->setArchived(a);
        }
        logger->log(candidate);
    }

    return a;
}
std::vector<Individual> &LoggingArchive::get_archived()
{
    return archive->get_archived();
}
void BruteforceArchive::setPopulation(std::shared_ptr<Population> population)
{
    IArchive::setPopulation(population);
    cache.reset();
}

// ImprovementTrackingArchive
ImprovementTrackingArchive::ImprovementTrackingArchive(std::shared_ptr<IArchive> archive,
                                                       std::function<double()> get_current_t,
                                                       double base,
                                                       double factor) :
    archive(archive), get_current_t_w(get_current_t), base(base), factor(factor)
{
}
void ImprovementTrackingArchive::setPopulation(std::shared_ptr<Population> population)
{
    IArchive::setPopulation(population);
    archive->setPopulation(population);
}
void ImprovementTrackingArchive::registerData()
{
    archive->registerData();
}
void ImprovementTrackingArchive::afterRegisterData()
{
    archive->afterRegisterData();
}
Archived ImprovementTrackingArchive::try_add(Individual candidate)
{
    Archived a = archive->try_add(candidate);
    if (a.added)
    {
        // Got added -> archive got updated!
        t.emplace(get_current_t());
    }
    else
    {
        double current_t = get_current_t();
        double t_at_last_archive_addition = get_t().value_or(0);
        double t_adj = factor * t_at_last_archive_addition + base;

        if (current_t >= t_adj)
        {
            // std::cerr <<
            //      "aborting. current t: " << current_t << "\n" <<
            //      "t at last archive change: " << t_at_last_archive_addition << "\n" <<
            //      "t threshold: " << t_adj << std::endl; 
            throw limit_reached();
        }
    }
    return a;
}
std::optional<double> ImprovementTrackingArchive::get_t()
{
    return t;
}
std::vector<Individual> &ImprovementTrackingArchive::get_archived()
{
    return archive->get_archived();
}
double ImprovementTrackingArchive::get_current_t()
{
    return get_current_t_w();
}

