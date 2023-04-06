#include "initializers.hpp"

void CategoricalUniformInitializer::initialize(std::vector<Individual> &iis)
{
    auto &pop = (*population);
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    Rng &rng = *pop.getGlobalData<Rng>();
    for (auto ii : iis)
    {
        GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(data.l);
        for (unsigned long long int i = 0; i < data.l; ++i)
        {
            std::uniform_int_distribution<unsigned long long int> gene(0, data.alphabet_size[i] - 1);
            genotype.genotype[i] = static_cast<char>(gene(rng.rng));
        }
    }
}
void CategoricalUniformInitializer::afterRegisterData()
{
    ISolutionInitializer::afterRegisterData();
    Population &pop = (*population);
    t_assert(pop.isRegistered<GenotypeCategorical>(),
             "This initializer requires a categorical genotype to be present.");
}
void CategoricalProbabilisticallyCompleteInitializer::initialize(std::vector<Individual> &iis)
{
    auto &pop = (*population);
    GenotypeCategoricalData &data = *pop.getGlobalData<GenotypeCategoricalData>();
    Rng &rng = *pop.getGlobalData<Rng>();
    for (auto ii : iis)
    {
        GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(ii);
        genotype.genotype.resize(data.l);
    }
    std::vector<char> genes(iis.size());
    for (unsigned long long int i = 0; i < data.l; ++i)
    {
        // Fill vector of genes such that each count is expressed approximately equally
        for (unsigned long long int j = 0; j < iis.size(); j++)
        {
            genes[j] = static_cast<char>(j % static_cast<unsigned long long int>(data.alphabet_size[i]));
        }
        // And shuffle it!
        std::shuffle(genes.begin(), genes.end(), rng.rng);
        // Assign the genes to each member
        for (unsigned long long int j = 0; j < iis.size(); j++)
        {
            GenotypeCategorical &genotype = pop.getData<GenotypeCategorical>(iis[j]);
            genotype.genotype[i] = genes[j];
        }
    }
}
void CategoricalProbabilisticallyCompleteInitializer::afterRegisterData()
{
    ISolutionInitializer::afterRegisterData();
    Population &pop = (*population);
    t_assert(pop.isRegistered<GenotypeCategorical>(),
             "This initializer requires a categorical genotype to be present.");
}