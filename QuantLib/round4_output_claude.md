#test

**File 1: `pricingengines/vanilla/fdblackscholesvanillaengine.hpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008 Ralph Schreyer
 Copyright (C) 2008, 2009 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file fdblackscholesvanillaengine.hpp
    \brief Finite-Differences Black Scholes vanilla option engine
*/

#ifndef quantlib_fd_black_scholes_vanilla_engine_hpp
#define quantlib_fd_black_scholes_vanilla_engine_hpp

#include <ql/pricingengine.hpp>
#include <ql/instruments/dividendvanillaoption.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>

namespace QuantLib {

    //! Finite-Differences Black Scholes vanilla option engine

    /*! \ingroup vanillaengines

        \test the correctness of the returned value is tested by
              reproducing results available in web/literature
              and comparison with Black pricing.

        \par Exponential fitting (Duffy 2004)
        When \a useExponentialFitting is \c true the engine constructs
        an FdmFittedBlackScholesOp spatial operator that replaces the
        standard centered second-derivative coefficient \f$\sigma\f$
        with a fitted coefficient
        \f$ \rho = (\mu h/2)\coth(\mu h/(2\sigma)) \f$, guaranteeing
        an M-matrix (monotone) discretization for all mesh Peclet
        numbers.  This eliminates spurious oscillations in convection-
        dominated regimes (small volatility, large drift) and near
        non-smooth payoff data (strike kinks).

        \par Usage examples
        \code
        // Standard CN (existing behavior, unchanged):
        auto engine1 = ext::make_shared<FdBlackScholesVanillaEngine>(
            process, 100, 100, 0, FdmSchemeDesc::CrankNicolson());

        // Duffy's fitted implicit Euler (new):
        auto engine2 = ext::make_shared<FdBlackScholesVanillaEngine>(
            process, 100, 100, 0,
            FdmSchemeDesc::FittedImplicitEuler(),
            false, -Null<Real>(),
            FdBlackScholesVanillaEngine::Spot,
            true);  // useExponentialFitting

        // Rannacher CN startup with exponential fitting (new):
        auto engine3 = ext::make_shared<FdBlackScholesVanillaEngine>(
            process, 100, 100, 2,
            FdmSchemeDesc::CrankNicolson(),
            false, -Null<Real>(),
            FdBlackScholesVanillaEngine::Spot,
            true);  // useExponentialFitting

        // Via builder:
        auto engine4 = MakeFdBlackScholesVanillaEngine(process)
            .withTGrid(100).withXGrid(100)
            .withFdmSchemeDesc(FdmSchemeDesc::FittedImplicitEuler())
            .withExponentialFitting()
            ;
        \endcode
    */
    class FdmQuantoHelper;
    class GeneralizedBlackScholesProcess;

    class FdBlackScholesVanillaEngine : public DividendVanillaOption::engine {
      public:
        enum CashDividendModel { Spot, Escrowed };

        // Constructor
        explicit FdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess>,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            CashDividendModel cashDividendModel = Spot,
            bool useExponentialFitting = false);

        FdBlackScholesVanillaEngine(ext::shared_ptr<GeneralizedBlackScholesProcess>,
                                    ext::shared_ptr<FdmQuantoHelper> quantoHelper,
                                    Size tGrid = 100,
                                    Size xGrid = 100,
                                    Size dampingSteps = 0,
                                    const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
                                    bool localVol = false,
                                    Real illegalLocalVolOverwrite = -Null<Real>(),
                                    CashDividendModel cashDividendModel = Spot,
                                    bool useExponentialFitting = false);

        void calculate() const override;

      private:
        const ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        const Size tGrid_, xGrid_, dampingSteps_;
        const FdmSchemeDesc schemeDesc_;
        const bool localVol_;
        const Real illegalLocalVolOverwrite_;
        const ext::shared_ptr<FdmQuantoHelper> quantoHelper_;
        const CashDividendModel cashDividendModel_;
        const bool useExponentialFitting_;
    };


    class MakeFdBlackScholesVanillaEngine {
      public:
        explicit MakeFdBlackScholesVanillaEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process);

        MakeFdBlackScholesVanillaEngine& withQuantoHelper(
            const ext::shared_ptr<FdmQuantoHelper>& quantoHelper);

        MakeFdBlackScholesVanillaEngine& withTGrid(Size tGrid);
        MakeFdBlackScholesVanillaEngine& withXGrid(Size xGrid);
        MakeFdBlackScholesVanillaEngine& withDampingSteps(
            Size dampingSteps);

        MakeFdBlackScholesVanillaEngine& withFdmSchemeDesc(
            const FdmSchemeDesc& schemeDesc);

        MakeFdBlackScholesVanillaEngine& withLocalVol(bool localVol);
        MakeFdBlackScholesVanillaEngine& withIllegalLocalVolOverwrite(
            Real illegalLocalVolOverwrite);

        MakeFdBlackScholesVanillaEngine& withCashDividendModel(
            FdBlackScholesVanillaEngine::CashDividendModel cashDividendModel);

        /*! Enable Duffy (2004) exponential fitting in the spatial
            operator.  Guarantees M-matrix monotonicity for all mesh
            Peclet numbers.  Recommended for small-volatility /
            convection-dominated regimes and near payoff
            discontinuities.                                          */
        MakeFdBlackScholesVanillaEngine& withExponentialFitting(
            bool b = true);

        operator ext::shared_ptr<PricingEngine>() const;
      private:
        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        Size tGrid_, xGrid_, dampingSteps_;
        ext::shared_ptr<FdmSchemeDesc> schemeDesc_;
        bool localVol_;
        Real illegalLocalVolOverwrite_;
        ext::shared_ptr<FdmQuantoHelper> quantoHelper_;
        FdBlackScholesVanillaEngine::CashDividendModel cashDividendModel_;
        bool useExponentialFitting_;
    };
}

#endif
```

---

**File 2: `pricingengines/vanilla/fdblackscholesvanillaengine.cpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008, 2009 Ralph Schreyer
 Copyright (C) 2008, 2009 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <ql/exercise.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/utilities/escroweddividendadjustment.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholessolver.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmescrowedloginnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>

namespace QuantLib {

    FdBlackScholesVanillaEngine::FdBlackScholesVanillaEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process,
        Size tGrid,
        Size xGrid,
        Size dampingSteps,
        const FdmSchemeDesc& schemeDesc,
        bool localVol,
        Real illegalLocalVolOverwrite,
        CashDividendModel cashDividendModel,
        bool useExponentialFitting)
    : process_(std::move(process)), tGrid_(tGrid), xGrid_(xGrid), dampingSteps_(dampingSteps),
      schemeDesc_(schemeDesc), localVol_(localVol),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite),
      quantoHelper_(ext::shared_ptr<FdmQuantoHelper>()), cashDividendModel_(cashDividendModel),
      useExponentialFitting_(useExponentialFitting) {
        registerWith(process_);
    }

    FdBlackScholesVanillaEngine::FdBlackScholesVanillaEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process,
        ext::shared_ptr<FdmQuantoHelper> quantoHelper,
        Size tGrid,
        Size xGrid,
        Size dampingSteps,
        const FdmSchemeDesc& schemeDesc,
        bool localVol,
        Real illegalLocalVolOverwrite,
        CashDividendModel cashDividendModel,
        bool useExponentialFitting)
    : process_(std::move(process)), tGrid_(tGrid), xGrid_(xGrid), dampingSteps_(dampingSteps),
      schemeDesc_(schemeDesc), localVol_(localVol),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite), quantoHelper_(std::move(quantoHelper)),
      cashDividendModel_(cashDividendModel),
      useExponentialFitting_(useExponentialFitting) {
        registerWith(process_);
        registerWith(quantoHelper_);
    }


    void FdBlackScholesVanillaEngine::calculate() const {
        // 0. Cash dividend model
        const Date exerciseDate = arguments_.exercise->lastDate();
        const Time maturity = process_->time(exerciseDate);
        const Date settlementDate = process_->riskFreeRate()->referenceDate();

        Real spotAdjustment = 0.0;
        DividendSchedule dividendSchedule = DividendSchedule();

        ext::shared_ptr<EscrowedDividendAdjustment> escrowedDivAdj;

        switch (cashDividendModel_) {
          case Spot:
            dividendSchedule = arguments_.cashFlow;
            break;
          case Escrowed:
            if  (arguments_.exercise->type() != Exercise::European)
                // add dividend dates as stopping times
                for (const auto& cf: arguments_.cashFlow)
                    dividendSchedule.push_back(
                        ext::make_shared<FixedDividend>(0.0, cf->date()));

            QL_REQUIRE(quantoHelper_ == nullptr,
                "Escrowed dividend model is not supported for Quanto-Options");

            escrowedDivAdj = ext::make_shared<EscrowedDividendAdjustment>(
                arguments_.cashFlow,
                process_->riskFreeRate(),
                process_->dividendYield(),
                [&](Date d){ return process_->time(d); },
                maturity
            );

            spotAdjustment =
                escrowedDivAdj->dividendAdjustment(process_->time(settlementDate));

            QL_REQUIRE(process_->x0() + spotAdjustment > 0.0,
                    "spot minus dividends becomes negative");

            break;
          default:
              QL_FAIL("unknwon cash dividend model");
        }

        // 1. Mesher
        const ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::dynamic_pointer_cast<StrikedTypePayoff>(arguments_.payoff);

        const ext::shared_ptr<Fdm1dMesher> equityMesher =
            ext::make_shared<FdmBlackScholesMesher>(
                    xGrid_, process_, maturity, payoff->strike(), 
                    Null<Real>(), Null<Real>(), 0.0001, 1.5, 
                    std::pair<Real, Real>(payoff->strike(), 0.1),
                    dividendSchedule, quantoHelper_,
                    spotAdjustment);
        
        const ext::shared_ptr<FdmMesher> mesher =
            ext::make_shared<FdmMesherComposite>(equityMesher);
        
        // 2. Calculator
        ext::shared_ptr<FdmInnerValueCalculator> calculator;
        switch (cashDividendModel_) {
          case Spot:
              calculator = ext::make_shared<FdmLogInnerValue>(
                  payoff, mesher, 0);
            break;
          case Escrowed:
              calculator = ext::make_shared<FdmEscrowedLogInnerValueCalculator>(
                  escrowedDivAdj, payoff, mesher, 0);
            break;
          default:
              QL_FAIL("unknwon cash dividend model");
        }

        // 3. Step conditions
        const ext::shared_ptr<FdmStepConditionComposite> conditions = 
            FdmStepConditionComposite::vanillaComposite(
                dividendSchedule, arguments_.exercise, mesher, calculator,
                process_->riskFreeRate()->referenceDate(),
                process_->riskFreeRate()->dayCounter());

        // 4. Boundary conditions
        const FdmBoundaryConditionSet boundaries;

        // 5. Solver
        FdmSolverDesc solverDesc = { mesher, boundaries, conditions, calculator,
                                     maturity, tGrid_, dampingSteps_,
                                     useExponentialFitting_ };

        const ext::shared_ptr<FdmBlackScholesSolver> solver(
            ext::make_shared<FdmBlackScholesSolver>(
                Handle<GeneralizedBlackScholesProcess>(process_),
                payoff->strike(), solverDesc, schemeDesc_,
                localVol_, illegalLocalVolOverwrite_,
                Handle<FdmQuantoHelper>(quantoHelper_),
                useExponentialFitting_));

        const Real spot = process_->x0() + spotAdjustment;

        results_.value = solver->valueAt(spot);
        results_.delta = solver->deltaAt(spot);
        results_.gamma = solver->gammaAt(spot);
        results_.theta = solver->thetaAt(spot);
    }

    MakeFdBlackScholesVanillaEngine::MakeFdBlackScholesVanillaEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process)
    : process_(std::move(process)), tGrid_(100), xGrid_(100), dampingSteps_(0),
      schemeDesc_(ext::make_shared<FdmSchemeDesc>(FdmSchemeDesc::Douglas())), localVol_(false),
      illegalLocalVolOverwrite_(-Null<Real>()), quantoHelper_(ext::shared_ptr<FdmQuantoHelper>()),
      cashDividendModel_(FdBlackScholesVanillaEngine::Spot),
      useExponentialFitting_(false) {}

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withQuantoHelper(
        const ext::shared_ptr<FdmQuantoHelper>& quantoHelper) {
        quantoHelper_ = quantoHelper;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withTGrid(Size tGrid) {
        tGrid_ = tGrid;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withXGrid(Size xGrid) {
        xGrid_ = xGrid;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withDampingSteps(Size dampingSteps) {
        dampingSteps_ = dampingSteps;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withFdmSchemeDesc(
        const FdmSchemeDesc& schemeDesc) {
        schemeDesc_ = ext::make_shared<FdmSchemeDesc>(schemeDesc);
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withLocalVol(bool localVol) {
        localVol_ = localVol;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withIllegalLocalVolOverwrite(
        Real illegalLocalVolOverwrite) {
        illegalLocalVolOverwrite_ = illegalLocalVolOverwrite;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withCashDividendModel(
        FdBlackScholesVanillaEngine::CashDividendModel cashDividendModel) {
        cashDividendModel_ = cashDividendModel;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine&
    MakeFdBlackScholesVanillaEngine::withExponentialFitting(bool b) {
        useExponentialFitting_ = b;
        return *this;
    }

    MakeFdBlackScholesVanillaEngine::operator
    ext::shared_ptr<PricingEngine>() const {
        return ext::make_shared<FdBlackScholesVanillaEngine>(
            process_,
            quantoHelper_,
            tGrid_, xGrid_, dampingSteps_,
            *schemeDesc_,
            localVol_,
            illegalLocalVolOverwrite_,
            cashDividendModel_,
            useExponentialFitting_);
    }
}
```

---

**File 3: `pricingengines/barrier/fdblackscholesbarrierengine.hpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008 Ralph Schreyer
 Copyright (C) 2008 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file fdblackscholesbarrierengine.hpp
    \brief Finite-Differences Black Scholes barrier option engine
*/

#ifndef quantlib_fd_black_scholes_barrier_engine_hpp
#define quantlib_fd_black_scholes_barrier_engine_hpp

#include <ql/processes/blackscholesprocess.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/instruments/dividendbarrieroption.hpp>

namespace QuantLib {

    //! Finite-Differences Black Scholes barrier option engine

    /*!
        \ingroup barrierengines

        \test the correctness of the returned value is tested by
              reproducing results available in web/literature
              and comparison with Black pricing.
    */
    class FdBlackScholesBarrierEngine : public DividendBarrierOption::engine {
      public:
        // Constructor
        explicit FdBlackScholesBarrierEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            bool useExponentialFitting = false);

        void calculate() const override;

      private:
        const ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        const Size tGrid_, xGrid_, dampingSteps_;
        const FdmSchemeDesc schemeDesc_;
        const bool localVol_;
        const Real illegalLocalVolOverwrite_;
        const bool useExponentialFitting_;
    };


}

#endif /*quantlib_fd_black_scholes_barrier_engine_hpp*/
```

---

**File 4: `pricingengines/barrier/fdblackscholesbarrierengine.cpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008, 2009 Ralph Schreyer
 Copyright (C) 2008, 2009 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <ql/exercise.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholessolver.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdividendhandler.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/pricingengines/barrier/fdblackscholesbarrierengine.hpp>
#include <ql/pricingengines/barrier/fdblackscholesrebateengine.hpp>
#include <ql/pricingengines/vanilla/fdblackscholesvanillaengine.hpp>
#include <utility>

namespace QuantLib {

    FdBlackScholesBarrierEngine::FdBlackScholesBarrierEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process,
        Size tGrid,
        Size xGrid,
        Size dampingSteps,
        const FdmSchemeDesc& schemeDesc,
        bool localVol,
        Real illegalLocalVolOverwrite,
        bool useExponentialFitting)
    : process_(std::move(process)), tGrid_(tGrid), xGrid_(xGrid), dampingSteps_(dampingSteps),
      schemeDesc_(schemeDesc), localVol_(localVol),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite),
      useExponentialFitting_(useExponentialFitting) {

        registerWith(process_);
    }

    void FdBlackScholesBarrierEngine::calculate() const {

        // 1. Mesher
        const ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::dynamic_pointer_cast<StrikedTypePayoff>(arguments_.payoff);
        const Time maturity = process_->time(arguments_.exercise->lastDate());

        Real xMin=Null<Real>();
        Real xMax=Null<Real>();
        if (   arguments_.barrierType == Barrier::DownIn
            || arguments_.barrierType == Barrier::DownOut) {
            xMin = std::log(arguments_.barrier);
        }
        if (   arguments_.barrierType == Barrier::UpIn
            || arguments_.barrierType == Barrier::UpOut) {
            xMax = std::log(arguments_.barrier);
        }

        const ext::shared_ptr<Fdm1dMesher> equityMesher(
            new FdmBlackScholesMesher(
                xGrid_, process_, maturity, payoff->strike(),
                xMin, xMax, 0.0001, 1.5,
                std::make_pair(Null<Real>(), Null<Real>()),
                arguments_.cashFlow));
        
        const ext::shared_ptr<FdmMesher> mesher (
            ext::make_shared<FdmMesherComposite>(equityMesher));

        // 2. Calculator
        ext::shared_ptr<FdmInnerValueCalculator> calculator(
            ext::make_shared<FdmLogInnerValue>(payoff, mesher, 0));

        // 3. Step conditions
        std::list<ext::shared_ptr<StepCondition<Array> > > stepConditions;
        std::list<std::vector<Time> > stoppingTimes;

        // 3.1 Step condition if discrete dividends
        ext::shared_ptr<FdmDividendHandler> dividendCondition(
            ext::make_shared<FdmDividendHandler>(arguments_.cashFlow, mesher,
                                   process_->riskFreeRate()->referenceDate(),
                                   process_->riskFreeRate()->dayCounter(), 0));

        if(!arguments_.cashFlow.empty()) {
            stepConditions.push_back(dividendCondition);
            stoppingTimes.push_back(dividendCondition->dividendTimes());
        }

        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "only european style option are supported");

        ext::shared_ptr<FdmStepConditionComposite> conditions(
            ext::make_shared<FdmStepConditionComposite>(stoppingTimes, stepConditions));

        // 4. Boundary conditions
        FdmBoundaryConditionSet boundaries;
        if (   arguments_.barrierType == Barrier::DownIn
            || arguments_.barrierType == Barrier::DownOut) {
            boundaries.push_back(
                ext::make_shared<FdmDirichletBoundary>(mesher, arguments_.rebate, 0,
                                         FdmDirichletBoundary::Lower));

        }

        if (   arguments_.barrierType == Barrier::UpIn
            || arguments_.barrierType == Barrier::UpOut) {
            boundaries.push_back(
                ext::make_shared<FdmDirichletBoundary>(mesher, arguments_.rebate, 0,
                                         FdmDirichletBoundary::Upper));
        }

        // 5. Solver
        FdmSolverDesc solverDesc = { mesher, boundaries, conditions, calculator,
                                     maturity, tGrid_, dampingSteps_,
                                     useExponentialFitting_ };

        ext::shared_ptr<FdmBlackScholesSolver> solver(
            ext::make_shared<FdmBlackScholesSolver>(
                               Handle<GeneralizedBlackScholesProcess>(process_),
                               payoff->strike(), solverDesc, schemeDesc_,
                               localVol_, illegalLocalVolOverwrite_,
                               Handle<FdmQuantoHelper>(),
                               useExponentialFitting_));

        const Real spot = process_->x0();
        results_.value = solver->valueAt(spot);
        results_.delta = solver->deltaAt(spot);
        results_.gamma = solver->gammaAt(spot);
        results_.theta = solver->thetaAt(spot);

        // 6. Calculate vanilla option and rebate for in-barriers
        if (   arguments_.barrierType == Barrier::DownIn
            || arguments_.barrierType == Barrier::UpIn) {
            // Cast the payoff
            ext::shared_ptr<StrikedTypePayoff> payoff =
                    ext::dynamic_pointer_cast<StrikedTypePayoff>(
                                                            arguments_.payoff);
            // Calculate the vanilla option
            
            ext::shared_ptr<DividendVanillaOption> vanillaOption(
                ext::make_shared<DividendVanillaOption>(payoff,arguments_.exercise,
                                          dividendCondition->dividendDates(), 
                                          dividendCondition->dividends()));
            
            vanillaOption->setPricingEngine(
                ext::make_shared<FdBlackScholesVanillaEngine>(
                        process_, tGrid_, xGrid_,
                        0, // dampingSteps
                        schemeDesc_, localVol_, illegalLocalVolOverwrite_,
                        FdBlackScholesVanillaEngine::Spot,
                        useExponentialFitting_));

            // Calculate the rebate value
            ext::shared_ptr<DividendBarrierOption> rebateOption(
                ext::make_shared<DividendBarrierOption>(arguments_.barrierType,
                                          arguments_.barrier,
                                          arguments_.rebate,
                                          payoff, arguments_.exercise,
                                          dividendCondition->dividendDates(), 
                                          dividendCondition->dividends()));
            
            const Size min_grid_size = 50;
            const Size rebateDampingSteps 
                = (dampingSteps_ > 0) ? std::min(Size(1), dampingSteps_/2) : 0; 

            rebateOption->setPricingEngine(ext::make_shared<FdBlackScholesRebateEngine>(
                            process_, tGrid_, std::max(min_grid_size, xGrid_/5), 
                            rebateDampingSteps, schemeDesc_, localVol_, 
                            illegalLocalVolOverwrite_,
                            useExponentialFitting_));

            results_.value = vanillaOption->NPV()   + rebateOption->NPV()
                                                    - results_.value;
            results_.delta = vanillaOption->delta() + rebateOption->delta()
                                                    - results_.delta;
            results_.gamma = vanillaOption->gamma() + rebateOption->gamma()
                                                    - results_.gamma;
            results_.theta = vanillaOption->theta() + rebateOption->theta()
                                                    - results_.theta;
        }
    }
}
```

---

**File 5: `pricingengines/barrier/fdblackscholesrebateengine.hpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008, 2009 Ralph Schreyer
 Copyright (C) 2008 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file fdblackscholesrebateengine.hpp
    \brief Finite-Differences Black Scholes barrier option rebate helper engine
*/

#ifndef quantlib_fd_black_scholes_rebate_engine_hpp
#define quantlib_fd_black_scholes_rebate_engine_hpp

#include <ql/processes/blackscholesprocess.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/instruments/dividendbarrieroption.hpp>

namespace QuantLib {

    //! Finite-Differences Black Scholes barrier option rebate helper engine

    /*!
        \ingroup barrierengines
    */
    class FdBlackScholesRebateEngine : public DividendBarrierOption::engine {
      public:
        // Constructor
        explicit FdBlackScholesRebateEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process,
            Size tGrid = 100,
            Size xGrid = 100,
            Size dampingSteps = 0,
            const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas(),
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            bool useExponentialFitting = false);

        void calculate() const override;

      private:
        const ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        const Size tGrid_, xGrid_, dampingSteps_;
        const FdmSchemeDesc schemeDesc_;
        const bool localVol_;
        const Real illegalLocalVolOverwrite_;
        const bool useExponentialFitting_;
};


}

#endif /*quantlib_fd_black_scholes_rebate_engine_hpp*/
```

---

**File 6: `pricingengines/barrier/fdblackscholesrebateengine.cpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2008 Andreas Gaida
 Copyright (C) 2008, 2009 Ralph Schreyer
 Copyright (C) 2008 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <ql/exercise.hpp>
#include <ql/methods/finitedifferences/meshers/fdmblackscholesmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdmblackscholessolver.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/pricingengines/barrier/fdblackscholesrebateengine.hpp>
#include <utility>

namespace QuantLib {

    FdBlackScholesRebateEngine::FdBlackScholesRebateEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process,
        Size tGrid,
        Size xGrid,
        Size dampingSteps,
        const FdmSchemeDesc& schemeDesc,
        bool localVol,
        Real illegalLocalVolOverwrite,
        bool useExponentialFitting)
    : process_(std::move(process)), tGrid_(tGrid), xGrid_(xGrid), dampingSteps_(dampingSteps),
      schemeDesc_(schemeDesc), localVol_(localVol),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite),
      useExponentialFitting_(useExponentialFitting) {

        registerWith(process_);
    }

    void FdBlackScholesRebateEngine::calculate() const {

        // 1. Mesher
        const ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::dynamic_pointer_cast<StrikedTypePayoff>(arguments_.payoff);
        const Time maturity = process_->time(arguments_.exercise->lastDate());

        Real xMin=Null<Real>();
        Real xMax=Null<Real>();
        if (   arguments_.barrierType == Barrier::DownIn
            || arguments_.barrierType == Barrier::DownOut) {
            xMin = std::log(arguments_.barrier);
        }
        if (   arguments_.barrierType == Barrier::UpIn
            || arguments_.barrierType == Barrier::UpOut) {
            xMax = std::log(arguments_.barrier);
        }

        const ext::shared_ptr<Fdm1dMesher> equityMesher(
            new FdmBlackScholesMesher(
                xGrid_, process_, maturity, payoff->strike(),
                xMin, xMax, 0.0001, 1.5,
                std::make_pair(Null<Real>(), Null<Real>()),
                arguments_.cashFlow));
        
        const ext::shared_ptr<FdmMesher> mesher (
            new FdmMesherComposite(equityMesher));
        
        // 2. Calculator
        const ext::shared_ptr<StrikedTypePayoff> rebatePayoff(
                new CashOrNothingPayoff(Option::Call, 0.0, arguments_.rebate));
        const ext::shared_ptr<FdmInnerValueCalculator> calculator(
                                new FdmLogInnerValue(rebatePayoff, mesher, 0));

        // 3. Step conditions
        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "only european style option are supported");
        
        const ext::shared_ptr<FdmStepConditionComposite> conditions =
            FdmStepConditionComposite::vanillaComposite(
                                arguments_.cashFlow, arguments_.exercise, 
                                mesher, calculator, 
                                process_->riskFreeRate()->referenceDate(),
                                process_->riskFreeRate()->dayCounter());

        // 4. Boundary conditions
        FdmBoundaryConditionSet  boundaries;
        if (   arguments_.barrierType == Barrier::DownIn
            || arguments_.barrierType == Barrier::DownOut) {
            boundaries.push_back(FdmBoundaryConditionSet::value_type(
                new FdmDirichletBoundary(mesher, arguments_.rebate, 0,
                                         FdmDirichletBoundary::Lower)));

        }
        if (   arguments_.barrierType == Barrier::UpIn
            || arguments_.barrierType == Barrier::UpOut) {
            boundaries.push_back(FdmBoundaryConditionSet::value_type(
                new FdmDirichletBoundary(mesher, arguments_.rebate, 0,
                                         FdmDirichletBoundary::Upper)));
        }

        // 5. Solver
        FdmSolverDesc solverDesc = { mesher, boundaries, conditions, calculator,
                                     maturity, tGrid_, dampingSteps_,
                                     useExponentialFitting_ };

        const ext::shared_ptr<FdmBlackScholesSolver> solver(
                new FdmBlackScholesSolver(
                                Handle<GeneralizedBlackScholesProcess>(process_),
                                payoff->strike(), solverDesc, schemeDesc_,
                                localVol_, illegalLocalVolOverwrite_,
                                Handle<FdmQuantoHelper>(),
                                useExponentialFitting_));

        const Real spot = process_->x0();
        results_.value = solver->valueAt(spot);
        results_.delta = solver->deltaAt(spot);
        results_.gamma = solver->gammaAt(spot);
        results_.theta = solver->thetaAt(spot);
    }
}
```

---

**Design notes for Round 5 (test harness):**

All three Black-Scholes FD engines now expose `useExponentialFitting` as a trailing constructor parameter with default `false`, preserving full backward compatibility. The parameter flows through to `FdmSolverDesc` (metadata) and `FdmBlackScholesSolver` (operator selection). The barrier engine propagates the flag to its internal vanilla and rebate sub-engines for in-barrier calculations.


The `MakeFdBlackScholesVanillaEngine` builder provides `withExponentialFitting()` for fluent configuration. The recommended configurations for testing are: `FdmSchemeDesc::FittedImplicitEuler()` with `useExponentialFitting=true` for the pure Duffy approach; `FdmSchemeDesc::CrankNicolson()` with damping steps and `useExponentialFitting=true` for the Rannacher-CN hybrid; and `FdmSchemeDesc::RannacherCrankNicolson()` with `useExponentialFitting=true` for the self-contained Rannacher variant.
