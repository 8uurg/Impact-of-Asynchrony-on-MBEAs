#pragma once

#include "base.hpp"
#include "logging.hpp"
#include <filesystem>

// Common
class missing_file : public std::exception
{
  public:
    missing_file(){};

    const char *what() const throw()
    {
        return "file is missing";
    }
};

class invalid_instance : public std::exception
{
  public:
    invalid_instance(){};

    const char *what() const throw()
    {
        return "file does not contain a valid instance";
    }
};

// Generic Problems

/**
 * Evaluation function taking discrete input
 */
class DiscreteObjectiveFunction : public ObjectiveFunction
{
  public:
    DiscreteObjectiveFunction(std::function<double(std::vector<char> &)> evaluation_function,
                              size_t l,
                              std::vector<char> alphabet_size,
                              size_t index = 0);
    void evaluate(Individual i) override;

    void registerData() override;

  private:
    std::function<double(std::vector<char> &)> evaluation_function;
    size_t l;
    std::vector<char> alphabet_size;
    size_t index;
};

/**
 * Evaluation function taking continuous input
 */
class ContinuousObjectiveFunction : public ObjectiveFunction
{
  public:
    ContinuousObjectiveFunction(std::function<double(std::vector<double> &)>, size_t l, size_t index = 0);
    void evaluate(Individual i) override;

    void registerData() override;

  private:
    std::function<double(std::vector<double> &)> evaluation_function;
    size_t l;
    size_t index;
};

// Problem: OneMax

// Common functions
double evaluate_onemax(size_t l, std::vector<char> &genotype);

// OneMax
//
// A simple sum over binary variables for a string length of l.
class OneMax : public ObjectiveFunction
{
  public:
    OneMax(size_t l, size_t index = 0);

    void evaluate(Individual i) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

  private:
    size_t l;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> gc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;
};

class ZeroMax : public ObjectiveFunction
{
  public:
    ZeroMax(size_t l, size_t index = 0);

    void evaluate(Individual i) override;

    void registerData() override;
    void afterRegisterData() override;

    void setPopulation(std::shared_ptr<Population> population) override;

  private:
    size_t l;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> gc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;
};

// Problem: MaxCut

// A weighted edge
struct Edge
{
    size_t i;
    size_t j;
    double w;
};

// A MaxCut instance consisting of weighted edges.
struct MaxCutInstance
{
    size_t num_vertices;
    size_t num_edges;
    std::vector<Edge> edges;
};

// Common functions
double evaluate_maxcut(MaxCutInstance &instance, std::vector<char> &genotype);
MaxCutInstance load_maxcut(std::istream &in);
MaxCutInstance load_maxcut(std::filesystem::path instancePath);

class MaxCut : public ObjectiveFunction
{
  public:
    MaxCut(MaxCutInstance edges, size_t index = 0);
    MaxCut(std::filesystem::path &path, size_t index = 0);

    void evaluate(Individual i) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;

  private:
    MaxCutInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;
};

// Problem: Best-of-Traps

//
struct ConcatenatedPermutedTrap
{
    size_t number_of_parameters;
    size_t block_size;
    std::vector<size_t> permutation;
    std::vector<char> optimum;
};

struct BestOfTrapsInstance
{
    size_t l;
    std::vector<ConcatenatedPermutedTrap> concatenatedPermutedTraps;
};

// Common functions
int evaluate_BestOfTraps(BestOfTrapsInstance &bestOfTraps, char *solution, size_t &best_fn);
BestOfTrapsInstance load_BestOfTraps(std::filesystem::path inpath);
BestOfTrapsInstance load_BestOfTraps(std::istream &in);

class BestOfTraps : public ObjectiveFunction
{
  public:
    BestOfTraps(BestOfTrapsInstance instance, size_t index = 0);
    BestOfTraps(std::filesystem::path &path, size_t index = 0);

    void evaluate(Individual i) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;

  private:
    BestOfTrapsInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;
};

class WorstOfTraps : public ObjectiveFunction
{
  public:
    WorstOfTraps(BestOfTrapsInstance instance, size_t index = 0);
    WorstOfTraps(std::filesystem::path &path, size_t index = 0);

    void evaluate(Individual i) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;

  private:
    BestOfTrapsInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;
};

// Compose multiple functions together by calling them sequentially.
//
// Tip: Force the different functions to target different objectives to construct a multi-objective function
//      from single-objective functions.
class Compose : public ObjectiveFunction
{
  public:
    Compose(std::vector<std::shared_ptr<ObjectiveFunction>> problems) : problems(problems)
    {
    }
    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void evaluate(Individual i) override;

  private:
    std::vector<std::shared_ptr<ObjectiveFunction>> problems;
};

// Peter's Benchmark Function
double GOMEA_HierarchicalDeceptiveTrapProblemEvaluation(int l, int k, char *genes);

class HierarchicalDeceptiveTrap : public ObjectiveFunction
{
  private:
    size_t l;
    size_t k;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

  public:
    HierarchicalDeceptiveTrap(size_t l, size_t k = 3, size_t index = 0);
    void evaluate(Individual ii) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
};

//
struct NKSubfunction
{
    std::vector<size_t> variables;
    std::vector<double> lut;
};

struct NKLandscapeInstance
{
    std::vector<NKSubfunction> subfunctions;
    size_t l;
};

NKLandscapeInstance load_nklandscape(std::istream &in);
NKLandscapeInstance load_nklandscape(std::filesystem::path instancePath);
double evaluate_nklandscape(NKLandscapeInstance &instance, std::vector<char> &genotype);

class NKLandscape : public ObjectiveFunction
{
  private:
    NKLandscapeInstance instance;
    size_t index;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> ggc;
        TypedGetter<Objective> go;
    };
    std::optional<Cache> cache;

    void doCache();

  public:
    NKLandscape(std::filesystem::path path, size_t index = 0);
    NKLandscape(NKLandscapeInstance instance, size_t index = 0);
    void evaluate(Individual ii) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
};

class EvaluationLogger : public ObjectiveFunction
{
    std::shared_ptr<ObjectiveFunction> wrapping;
    std::shared_ptr<BaseLogger> logger;

  public:
    EvaluationLogger(std::shared_ptr<ObjectiveFunction> wrapping, std::shared_ptr<BaseLogger> logger);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void evaluate(Individual ii) override;
};