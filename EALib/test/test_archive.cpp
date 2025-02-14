//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "archive.hpp"
#include <catch2/catch.hpp>
#include <random>

TEST_CASE("BruteforceArchive")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<Objective>();
    BruteforceArchive archive({0, 1});
    archive.setPopulation(pop);

    Individual ia = pop->newIndividual();
    Objective &ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-1, 0};

    auto r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);

    r = archive.try_add(ia);
    REQUIRE(r.added == false);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {0, 0};

    r = archive.try_add(ia);
    REQUIRE(r.added == false);
    REQUIRE(r.dominated == true);
    REQUIRE(archive.get_archived().size() == 1);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-2, 0};

    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {0, -1};

    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 2);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-1, -0.5};

    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 3);

    ia_o = pop->getData<Objective>(ia);
    ia_o.objectives = {-3, -3};
    r = archive.try_add(ia);
    REQUIRE(r.added == true);
    REQUIRE(r.dominated == false);
    REQUIRE(archive.get_archived().size() == 1);
}
