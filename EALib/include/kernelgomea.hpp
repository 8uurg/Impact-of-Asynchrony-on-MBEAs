//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include "base.hpp"
#include "gomea.hpp"
#include "utilities.hpp"
#include <random>

template <typename T, // Lambda double(size_t, size_t)
                      // Check lambda signature
          typename = std::enable_if_t<std::is_invocable_r_v<double, T, size_t, size_t>>>
std::vector<size_t> getKNNNeighborhood(T &&distance, Rng &rng, size_t c, std::vector<size_t> &pool, size_t k)
{
    std::uniform_real_distribution<double> r01(0, 1);
    std::vector<std::tuple<double, double, size_t>> v(pool.size());
    for (size_t idx = 0; idx < pool.size(); ++idx)
    {
        v[idx] = std::make_tuple(distance(c, pool[idx]), r01(rng.rng), pool[idx]);
    }
    if (pool.size() > k)
    {
        std::nth_element(v.begin(), v.begin() + static_cast<long long int>(k), v.end());
        v.resize(k);
    }
    std::vector<size_t> neighbors(v.size());
    for (size_t idx = 0; idx < v.size(); ++idx)
    {
        neighbors[idx] = std::get<2>(v[idx]);
    }
    return neighbors;
}

template <typename T, // Lambda double(size_t, size_t)
                      // Check lambda signature
          typename = std::enable_if_t<std::is_invocable_r_v<double, T, size_t, size_t>>>
std::vector<std::vector<size_t>> getNeighborhoods(
    T &&distance, Rng &rng, std::vector<size_t> &pool, size_t k, bool symmetric)
{
    std::vector<std::vector<size_t>> neighborhoods(pool.size());
    std::vector<std::set<size_t>> is_neighbor;
    if (symmetric)
        is_neighbor.resize(pool.size());
    for (size_t idx = 0; idx < pool.size(); ++idx)
    {
        std::vector<size_t> neighborhood = getKNNNeighborhood(distance, rng, pool[idx], pool, k);
        // Append to neighborhood
        neighborhoods[idx].insert(neighborhoods[idx].end(), neighborhood.begin(), neighborhood.end());
        // If symmetric, keep track of which items are neighbors.
        if (symmetric)
            for (auto o_idx : neighborhood)
                is_neighbor[idx].insert(o_idx);
        // If symmetric, add self to neighbors.
        if (symmetric)
        {
            for (size_t o_idx : neighborhood)
            {
                neighborhoods[o_idx].push_back(idx);
                is_neighbor[o_idx].insert(idx);
            }
        }
    }
    return neighborhoods;
}

// -- Neighborhood Related Operations --

/**
 * @brief A neighborhood inference method.
 *
 */
class NeighborhoodLearner : public IDataUser
{
  public:
    virtual std::vector<Individual> get_neighborhood(const Individual &ii,
                                                     std::vector<Individual> &individuals,
                                                     size_t idx) = 0;
};

class BasicHammingKernel : public NeighborhoodLearner
{
    bool symmetric;

    struct Cache
    {
        Rng *rng;
    };
    std::optional<Cache> cache;
    void doCache()
    {
        if (cache.has_value())
            return;
        cache.emplace(Cache{population->getGlobalData<Rng>().get()});
    }

  public:
    BasicHammingKernel(bool symmetric = true) : symmetric(symmetric)
    {
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        NeighborhoodLearner::setPopulation(population);
        cache.reset();
    }

    virtual size_t get_neighborhood_size(const Individual &, size_t population_size)
    {
        return static_cast<size_t>(std::ceil(std::sqrt(population_size)));
    }

    std::vector<Individual> get_neighborhood(const Individual &kernel,
                                             std::vector<Individual> &individuals,
                                             size_t idx) override
    {
        doCache();

        TypedGetter<GenotypeCategorical> gc = population->getDataContainer<GenotypeCategorical>();

        size_t k = get_neighborhood_size(kernel, individuals.size());
        std::vector<size_t> indices(individuals.size());
        std::iota(indices.begin(), indices.end(), 0);
        auto neighborhood_indices = getNeighborhoods(
            [&individuals, &gc](size_t a, size_t b) {
                GenotypeCategorical gca = gc.getData(individuals[a]);
                GenotypeCategorical gcb = gc.getData(individuals[b]);

                return hamming_distance(
                    gca.genotype.data(), gcb.genotype.data(), std::min(gca.genotype.size(), gcb.genotype.size()));
            },
            *cache->rng,
            indices,
            k,
            symmetric);

        auto &nbi = neighborhood_indices[idx];

        std::vector<Individual> selected_neighborhood(nbi.size());
        for (size_t i = 0; i < selected_neighborhood.size(); ++i)
        {
            selected_neighborhood[i] = individuals[nbi[i]];
        }

        return selected_neighborhood;
    }
};

class BasicPopSizeHammingKernel : public BasicHammingKernel
{
  public:
    BasicPopSizeHammingKernel(bool symmetric = true) : BasicHammingKernel(symmetric)
    {
    }
    virtual size_t get_neighborhood_size(const Individual &, size_t population_size)
    {
        return population_size;
    }
};

class NISDoublingHammingKernel : public BasicHammingKernel
{
    struct Cache
    {
        TypedGetter<NIS> tgnis;
    };
    std::optional<Cache> cache;

    void doCache()
    {
        if (cache.has_value())
            return;
        cache.emplace(Cache{population->getDataContainer<NIS>()});
    }

  public:
    NISDoublingHammingKernel(bool symmetric = true) : BasicHammingKernel(symmetric)
    {
    }

    size_t get_neighborhood_size(const Individual &ii, size_t population_size) override
    {
        doCache();
        NIS &nis = cache->tgnis.getData(ii);
        size_t base = BasicHammingKernel::get_neighborhood_size(ii, population_size);
        size_t computed = std::min(base * (1 << nis.nis), population_size);
        std::cout << "Learning kernel with neighborhood size of " << computed << "." << std::endl;
        return computed;
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        BasicHammingKernel::setPopulation(population);
        cache.reset();
    }
};

class RandomPowerHammingKernel : public BasicHammingKernel
{
    struct Cache
    {
        Rng *rng;
    };
    std::optional<Cache> cache;

    void doCache()
    {
        if (cache.has_value())
            return;
        cache.emplace(Cache{population->getGlobalData<Rng>().get()});
    }

  public:
    RandomPowerHammingKernel(bool symmetric = true) : BasicHammingKernel(symmetric)
    {
    }

    size_t get_neighborhood_size(const Individual &, size_t population_size) override
    {
        doCache();

        // For the distribution we want a uniform distribution over the divisor
        // resulting in a distribution for which the maximum value is `population_size`
        // and the minimum value is 16.
        // population_size / max = population_size -> max = 1
        // population_size / min = 16 -> min = population_size / 16
        // > Small modification to ensure that minimum <= maximum: cap min value at 1.0.
        auto d_population_size = static_cast<double>(population_size);
        double m = std::min(d_population_size / 16.0, 1.0);
        std::uniform_real_distribution<double> d01(m, 1.0);
        size_t new_neighborhood_size = static_cast<size_t>(std::round(d_population_size / d01(cache->rng->rng)));
        return new_neighborhood_size;
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        BasicHammingKernel::setPopulation(population);
        cache.reset();
    }
};

struct LastNeighborhoodSize
{
    std::optional<size_t> last_neighborhood_size;
};

class PreservingRandomPowerHammingKernel : public BasicHammingKernel
{
    struct Cache
    {
        Rng *rng;
        TypedGetter<NIS> tgnis;
        TypedGetter<LastNeighborhoodSize> tglns;
    };
    std::optional<Cache> cache;

    void doCache()
    {
        if (cache.has_value())
            return;
        cache.emplace(Cache{population->getGlobalData<Rng>().get(),
                            population->getDataContainer<NIS>(),
                            population->getDataContainer<LastNeighborhoodSize>()});
    }

  public:
    PreservingRandomPowerHammingKernel(bool symmetric = true) : BasicHammingKernel(symmetric)
    {
    }

    size_t get_neighborhood_size(const Individual &ii, size_t population_size) override
    {
        doCache();
        auto &ii_nis = cache->tgnis.getData(ii);
        auto &ii_lns = cache->tglns.getData(ii);
        if (ii_nis.nis == 0 && ii_lns.last_neighborhood_size.has_value())
        {
            return ii_lns.last_neighborhood_size.value();
        }

        // For the distribution we want a uniform distribution over the divisor
        // resulting in a distribution for which the maximum value is `population_size`
        // and the minimum value is 16.
        // population_size / max = population_size -> max = 1
        // population_size / min = 16 -> min = population_size / 16
        // > Small modification to ensure that minimum <= maximum: cap min value at 1.0.
        auto d_population_size = static_cast<double>(population_size);
        double m = std::min(d_population_size / 16.0, 1.0);
        std::uniform_real_distribution<double> d01(m, 1.0);
        size_t new_neighborhood_size = static_cast<size_t>(std::round(d_population_size / d01(cache->rng->rng)));
        ii_lns.last_neighborhood_size = new_neighborhood_size;
        return new_neighborhood_size;
    }

    void registerData() override
    {
        // population->registerData<NIS>();
        population->registerData<LastNeighborhoodSize>();
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        BasicHammingKernel::setPopulation(population);
        cache.reset();
    }
};

class FirstBetterHammingKernel : public NeighborhoodLearner
{
    size_t index = 0;
    size_t num_better_required = 5;

    struct Cache
    {
        Rng *rng;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;
    void doCache()
    {
        if (cache.has_value())
            return;
        cache.emplace(Cache{population->getGlobalData<Rng>().get(), population->getDataContainer<Objective>()});
    }

  public:
    bool is_better(const Individual &a, const Individual &b)
    {
        // TODO: Change this to use acceptance criterion instead?
        auto &oa = cache->go.getData(a);
        auto &ob = cache->go.getData(b);

        if (oa.objectives.size() < index + 1)
            return false;
        if (ob.objectives.size() < index + 1)
            return true;

        return oa.objectives[index] >= ob.objectives[index];
    }

    std::vector<Individual> get_neighborhood(const Individual &kernel,
                                             std::vector<Individual> &individuals,
                                             size_t) override
    {
        doCache();

        TypedGetter<GenotypeCategorical> gc = population->getDataContainer<GenotypeCategorical>();

        std::vector<size_t> indices(individuals.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::vector<int> hamming_distances(individuals.size());
        for (size_t i = 0; i < individuals.size(); ++i)
        {
            auto &gck = gc.getData(kernel);
            auto &gco = gc.getData(individuals[i]);
            auto s = std::min(gck.genotype.size(), gco.genotype.size());
            hamming_distances[i] = hamming_distance(gck.genotype.data(), gco.genotype.data(), s);
        }

        std::sort(indices.begin(), indices.end(), [&hamming_distances](size_t a, size_t b) {
            return hamming_distances[a] < hamming_distances[b];
        });

        std::vector<Individual> selected_neighborhood(indices.size());
        size_t counted_num_better = 0;
        for (size_t i = 0; i < selected_neighborhood.size(); ++i)
        {
            auto &c = indices[i];
            selected_neighborhood[i] = individuals[c];
            if (is_better(individuals[c], kernel))
            {
                counted_num_better++;
            }
            if (counted_num_better >= num_better_required)
            {
                // Current solution is strictly better. Stop neighborhood (including current) here.
                selected_neighborhood.resize(i + 1);
                break;
            }
        }

        return selected_neighborhood;
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        NeighborhoodLearner::setPopulation(population);
        cache.reset();
    }
};