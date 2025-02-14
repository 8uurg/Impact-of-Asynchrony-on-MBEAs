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
 * @brief An acceptation criterion wrapping an arbitrary function.
 *
 * Useful for calling things from python, or, for testing.
 *
 * The function should adhere to the following properties:
 * - If a is better than b, 1 should be returned.
 * - If b is better than a, 2 should be returned.
 * - If a and b are equally good, 3 should be returned (1 | 2)
 * - If a and b are incomparable -- neither is strictly better than the other, 0 should be returned.
 *
 * Note that some components require a full ordering, if the criterion provided does not adhere to
 * this property, the behavior of these operators may be undefined.
 */
class FunctionAcceptanceCriterion : public IPerformanceCriterion
{
    std::function<short(Population &pop, Individual &a, Individual &b)> criterion;

  public:
    FunctionAcceptanceCriterion(std::function<short(Population &pop, Individual &a, Individual &b)> criterion);

    short compare(Individual &a, Individual &b) override;
};

/**
 * @brief A singleobjective acceptation criterion
 *
 * Performs a three-way comparison between a and b:
 * - If a is better than b, i.e. a has a lower objective value at a certain objective index than b, 1 is returned
 * - Equivalently, for if b is better than a, 2 is returned.
 * - If equal, 3 is returned
 * - If either is nan, 0 is returned.
 */
class SingleObjectiveAcceptanceCriterion : public IPerformanceCriterion
{
    struct Cache
    {
        TypedGetter<Objective> tgo;
    };
    std::optional<Cache> cache;

    void updateCache();

  public:
    SingleObjectiveAcceptanceCriterion(size_t index = 0);

    void afterRegisterData() override;
    void setPopulation(std::shared_ptr<Population> population) override;

    short compare(Individual &a, Individual &b) override;

  private:
    size_t index;
};

/**
 * @brief Performance criterion that performs a domination check
 *
 * Comparison generally behaves as follows:
 * - a is better than b if for all indices prodived a >= b, and for at least one a > b.
 * - comparison behaves equivalently for b better than a.
 * - If all equal, equality is returned.
 * - If none of the aforementioned traits hold, non-determinable is returned.
 */
class DominationObjectiveAcceptanceCriterion : public IPerformanceCriterion
{
  public:
    DominationObjectiveAcceptanceCriterion(const std::vector<size_t> indices);

    void afterRegisterData() override;

    short compare(Individual &a, Individual &b) override;

  private:
    const std::vector<size_t> indices;
};

/**
 * @brief Performance criterion that combines various performance criteria by testing sequentially,
 *        returning the first not-equal result.
 *
 * In particular, it stops upon the first test for which a non-equal result is returned.
 * If this holds for no test, it returns the result of the final criterion in the sequence.
 * Non-deterministic can be interpreted as equal using a flag at construction (defaults to false).
 */
class SequentialCombineAcceptanceCriterion : public IPerformanceCriterion
{
  public:
    SequentialCombineAcceptanceCriterion(std::vector<std::shared_ptr<IPerformanceCriterion>> criteria,
                                         bool nondeterminable_is_equal = false);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    short compare(Individual &a, Individual &b) override;

  private:
    const std::vector<std::shared_ptr<IPerformanceCriterion>> criteria;
    bool nondeterminable_is_equal;
};

/**
 * @brief Performance criterion that only looks at the comparison based on the threshold
 *        i.e., is one below, while the other is above?
 *
 * The general behavior of this criterion is similar to death-penalty as applied to
 * contraints. The degree of invalid-ness is not important, just that it is invalid.
 * General comparison results are as follows
 * - If a is below the threshold, while b is above, 1 is returned.
 * - Equivalently if b is below the threshold, and a is above, 2 is returned.
 * - If both are on the same side of the threshold, 3 is returned.
 */
class ThresholdAcceptanceCriterion : public IPerformanceCriterion
{
  public:
    ThresholdAcceptanceCriterion(size_t index, double threshold);

    short compare(Individual &a, Individual &b) override;

  private:
    size_t index;
    double threshold;
};

/**
 * @brief Compute objective minimum, maximum, and range over a pool of solutions for specific objective indices.
 *
 * @param population Population object to access solution data.
 * @param objective_indices Objective indices to compute min, max, range for.
 * @param pool The pool of individuals to compute min, max, range over.
 * @return std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> Tuple of vectors for min, max,
 * ranges (in that order).
 */
std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> compute_objective_min_max_ranges(
    Population &population, std::vector<size_t> objective_indices, std::vector<Individual> &pool);

struct ScalarizationWeights
{
    std::vector<double> weights;
};
const DataType SCALARIZATIONWEIGHTS{typeid(ScalarizationWeights).name()};

class Scalarizer : public IDataUser
{
  public:
    virtual void update_frame_of_reference(std::vector<Individual> individuals) = 0;
    virtual double scalarize(Individual ii) = 0;
};

class TschebysheffObjectiveScalarizer : public Scalarizer
{
  private:
    std::vector<size_t> objective_indices;
    std::vector<double> ranges;
    std::vector<double> min_v;
    std::vector<double> max_v;

    struct Cache
    {
        TypedGetter<ScalarizationWeights> tgsw;
        TypedGetter<Objective> tgo;
    };
    std::optional<Cache> cache;

    void updateCache();

  public:
    TschebysheffObjectiveScalarizer(std::vector<size_t> objective_indices);

    void setPopulation(std::shared_ptr<Population> population) override;

    void update_frame_of_reference(std::vector<Individual> individuals) override;
    double scalarize(Individual ii) override;
};

/**
 * @brief Performance criterion that uses a scalarizer
 *
 * Comparison is performed as follows
 * - Scalarize a and b
 * - Perform single objective check on scalarized values
 */
class ScalarizationAcceptanceCriterion : public IPerformanceCriterion
{
  private:
    const std::shared_ptr<Scalarizer> scalarizer;

  public:
    ScalarizationAcceptanceCriterion(std::shared_ptr<Scalarizer> scalarizer);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    short compare(Individual &a, Individual &b) override;
};
