#include "base.hpp"
#include "ga.hpp"
#include "gomea.hpp"
#include "sim.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <random>

class ISolutionSamplingDistribution : public IDataUser
{
  public:
    virtual bool should_update()
    {
        return true;
    };
    virtual void learn(std::vector<Individual> &) = 0;
    virtual void sample(Individual &ii) = 0;

    virtual std::shared_ptr<ISolutionSamplingDistribution> clone() = 0;
};

std::vector<std::vector<size_t>> learnMPM(size_t l, std::vector<std::vector<char>> &&genotypes);
std::vector<std::vector<size_t>> learnMPM(size_t l, std::vector<std::vector<char>> &genotypes);

class ECGAGreedyMarginalProduct : public ISolutionSamplingDistribution
{
  private:
    size_t l;
    size_t update_pop_every_learn_call;
    size_t update_mpm_every_pop_update;
    // First time: always update.
    size_t learn_calls_since_last_update = std::numeric_limits<size_t>::max() - 1;
    size_t updates_since_last_mpm_update = std::numeric_limits<size_t>::max() - 1;

    std::optional<std::filesystem::path> fos_path;

    bool checked_if_should_update = false;

    std::vector<std::vector<char>> genotypes;
    std::vector<std::vector<size_t>> mpm;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> tggc;
        Rng &rng;
    };
    std::optional<Cache> cache;

    ECGAGreedyMarginalProduct(ECGAGreedyMarginalProduct &o) = default;

  public:
    ECGAGreedyMarginalProduct(size_t update_mpm_every_pop_update = 1, size_t update_pop_every_learn_call = 1, std::optional<std::filesystem::path> fos_path = std::nullopt);

    bool should_update() override;
    void learn(std::vector<Individual> &) override;
    void sample(Individual &ii) override;

    std::shared_ptr<ISolutionSamplingDistribution> clone() override;

    void setPopulation(std::shared_ptr<Population> population) override;

    void afterRegisterData() override;
};

/**
 * @brief Implementation of the extended compact GA
 */
class SynchronousSimulatedECGA : public GenerationalApproach
{
  protected:
    std::shared_ptr<SimulatorParameters> sim;

    size_t population_size;
    std::vector<Individual> individuals;
    std::vector<Individual> offspring;
    bool initialized = false;

    std::shared_ptr<ISolutionInitializer> initializer;
    std::shared_ptr<ISelection> selection;
    std::shared_ptr<ISolutionSamplingDistribution> distribution;
    std::shared_ptr<IArchive> archive;
    std::shared_ptr<SpanLogger> sl;

    struct Cache
    {
        TypedGetter<Objective> tgo;
        TypedGetter<TimeSpent> tgts;
        Rng &rng;
    };
    std::optional<Cache> cache;
    void doCache();

    // TODO: Make this class-based instead, because this can get quite complicated...
    // This should however do for now.
    int replacement_strategy = 0;
    // Specific replacement strategies!

    // FIFO - Replace in order.
    size_t current_replacement_index = 0;
    void replace_fifo(Individual ii);

    // Replace predetermined index (i.e. offspring index)
    void replace_idx(size_t idx, Individual ii);

    // Replace uniformly
    void replace_uniformly(Individual ii);

    std::optional<std::shared_ptr<IPerformanceCriterion>> perf_criterion;
    bool replace_if_equal = true;
    bool replace_if_incomparable = true;

    void replace_selection_fifo(Individual ii);
    void replace_selection_idx(size_t idx, Individual ii);
    void replace_selection_uniformly(Individual ii);

    void contender_generational_like_selection(Individual ii);
    size_t target_selection_pool_size;
    bool include_population;
    std::vector<Individual> selection_pool;
    const std::shared_ptr<ISelection> generationalish_selection;

  public:
    SynchronousSimulatedECGA(int replacement_strategy,
                             std::shared_ptr<SimulatorParameters> sim,
                             size_t population_size,
                             std::shared_ptr<ISolutionSamplingDistribution> distribution,
                             std::shared_ptr<ISolutionInitializer> initializer,
                             std::shared_ptr<ISelection> selection,
                             std::shared_ptr<IArchive> archive,
                             std::shared_ptr<SpanLogger> sl = NULL);
    SynchronousSimulatedECGA(int replacement_strategy,
                             std::shared_ptr<IPerformanceCriterion> perf_criterion,
                             std::shared_ptr<SimulatorParameters> sim,
                             size_t population_size,
                             std::shared_ptr<ISolutionSamplingDistribution> distribution,
                             std::shared_ptr<ISolutionInitializer> initializer,
                             std::shared_ptr<ISelection> selection,
                             std::shared_ptr<IArchive> archive,
                             std::shared_ptr<SpanLogger> sl = NULL);
    SynchronousSimulatedECGA(std::shared_ptr<ISelection> generational_selection,
                             bool include_population,
                             std::shared_ptr<SimulatorParameters> sim,
                             size_t population_size,
                             std::shared_ptr<ISolutionSamplingDistribution> distribution,
                             std::shared_ptr<ISolutionInitializer> initializer,
                             std::shared_ptr<ISelection> selection,
                             std::shared_ptr<IArchive> archive,
                             std::shared_ptr<SpanLogger> sl = NULL);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    virtual void initialize();
    virtual void generation();

    void place_in_population(size_t idx, const Individual ii, const std::optional<int> override_replacement_strategy);

    virtual void sample_solution(size_t idx, Individual ii);
    virtual void evaluate_initial_solution(size_t idx, Individual ii);
    virtual void evaluate_solution(size_t idx, Individual ii);

    void step() override;

    std::vector<Individual> &getSolutionPopulation() override;
    bool terminated() override;
};

class AsynchronousSimulatedECGA : public SynchronousSimulatedECGA
{
  public:
    AsynchronousSimulatedECGA(int replacement_strategy,
                              std::shared_ptr<SimulatorParameters> sim,
                              size_t population_size,
                              std::shared_ptr<ISolutionSamplingDistribution> distribution,
                              std::shared_ptr<ISolutionInitializer> initializer,
                              std::shared_ptr<ISelection> selection,
                              std::shared_ptr<IArchive> archive,
                              std::shared_ptr<SpanLogger> sl = NULL);
    AsynchronousSimulatedECGA(int replacement_strategy,
                              std::shared_ptr<IPerformanceCriterion> perf_criterion,
                              std::shared_ptr<SimulatorParameters> sim,
                              size_t population_size,
                              std::shared_ptr<ISolutionSamplingDistribution> distribution,
                              std::shared_ptr<ISolutionInitializer> initializer,
                              std::shared_ptr<ISelection> selection,
                              std::shared_ptr<IArchive> archive,
                              std::shared_ptr<SpanLogger> sl = NULL);
    AsynchronousSimulatedECGA(std::shared_ptr<ISelection> generational_selection,
                              bool include_population,
                              std::shared_ptr<SimulatorParameters> sim,
                              size_t population_size,
                              std::shared_ptr<ISolutionSamplingDistribution> distribution,
                              std::shared_ptr<ISolutionInitializer> initializer,
                              std::shared_ptr<ISelection> selection,
                              std::shared_ptr<IArchive> archive,
                              std::shared_ptr<SpanLogger> sl = NULL);

    void initialize() override;
    void generation() override;
    virtual void sample_solution(size_t idx, Individual ii) override;
    void evaluate_initial_solution(size_t idx, Individual ii) override;
    void evaluate_solution(size_t idx, Individual ii) override;
    void sample_and_evaluate_new_solution(size_t idx);

    bool terminated() override;
};

class SynchronousSimulatedKernelECGA : public SynchronousSimulatedECGA
{
  private:
    size_t neighborhood_size;
    std::vector<std::shared_ptr<ISolutionSamplingDistribution>> per_solution_distribution;

    void initModels();

  public:
    SynchronousSimulatedKernelECGA(int replacement_strategy,
                                   std::shared_ptr<SimulatorParameters> sim,
                                   size_t population_size,
                                   size_t neighborhood_size,
                                   std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                   std::shared_ptr<ISolutionInitializer> initializer,
                                   std::shared_ptr<ISelection> selection,
                                   std::shared_ptr<IArchive> archive,
                                   std::shared_ptr<SpanLogger> sl = NULL);
    SynchronousSimulatedKernelECGA(int replacement_strategy,
                                   std::shared_ptr<IPerformanceCriterion> perf_criterion,
                                   std::shared_ptr<SimulatorParameters> sim,
                                   size_t population_size,
                                   size_t neighborhood_size,
                                   std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                   std::shared_ptr<ISolutionInitializer> initializer,
                                   std::shared_ptr<ISelection> selection,
                                   std::shared_ptr<IArchive> archive,
                                   std::shared_ptr<SpanLogger> sl = NULL);

    void sample_solution(size_t idx, Individual ii) override;
};

class AsynchronousSimulatedKernelECGA : public AsynchronousSimulatedECGA
{
  private:
    size_t neighborhood_size;
    std::vector<std::shared_ptr<ISolutionSamplingDistribution>> per_solution_distribution;

    void initModels();

  public:
    AsynchronousSimulatedKernelECGA(int replacement_strategy,
                                    std::shared_ptr<SimulatorParameters> sim,
                                    size_t population_size,
                                    size_t neighborhood_size,
                                    std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                    std::shared_ptr<ISolutionInitializer> initializer,
                                    std::shared_ptr<ISelection> selection,
                                    std::shared_ptr<IArchive> archive,
                                    std::shared_ptr<SpanLogger> sl = NULL);
    AsynchronousSimulatedKernelECGA(int replacement_strategy,
                                    std::shared_ptr<IPerformanceCriterion> perf_criterion,
                                    std::shared_ptr<SimulatorParameters> sim,
                                    size_t population_size,
                                    size_t neighborhood_size,
                                    std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                    std::shared_ptr<ISolutionInitializer> initializer,
                                    std::shared_ptr<ISelection> selection,
                                    std::shared_ptr<IArchive> archive,
                                    std::shared_ptr<SpanLogger> sl = NULL);

    void sample_solution(size_t idx, Individual ii) override;
};