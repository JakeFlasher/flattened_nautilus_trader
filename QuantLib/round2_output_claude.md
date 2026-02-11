
**File 1: `solvers/fdmbackwardsolver.hpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2009 Andreas Gaida
 Copyright (C) 2009 Ralph Schreyer
 Copyright (C) 2009 Klaus Spanderen

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

/*! \file fdmbackwardsolver.hpp
*/

#ifndef quantlib_fdm_backward_solver_hpp
#define quantlib_fdm_backward_solver_hpp

#include <ql/methods/finitedifferences/utilities/fdmboundaryconditionset.hpp>

namespace QuantLib {

    class FdmLinearOpComposite;
    class FdmStepConditionComposite;

    struct FdmSchemeDesc {
        enum FdmSchemeType { HundsdorferType, DouglasType,
                             CraigSneydType, ModifiedCraigSneydType,
                             ImplicitEulerType, ExplicitEulerType,
                             MethodOfLinesType, TrBDF2Type,
                             CrankNicolsonType,
                             FittedImplicitEulerType,
                             RannacherCNType };

        FdmSchemeDesc(FdmSchemeType type, Real theta, Real mu);

        const FdmSchemeType type;
        const Real theta, mu;

        // some default scheme descriptions
        static FdmSchemeDesc Douglas(); //same as Crank-Nicolson in 1 dimension
        static FdmSchemeDesc CrankNicolson();
        static FdmSchemeDesc ImplicitEuler();
        static FdmSchemeDesc ExplicitEuler();
        static FdmSchemeDesc CraigSneyd();
        static FdmSchemeDesc ModifiedCraigSneyd();
        static FdmSchemeDesc Hundsdorfer();
        static FdmSchemeDesc ModifiedHundsdorfer();
        static FdmSchemeDesc MethodOfLines(
            Real eps=0.001, Real relInitStepSize=0.01);
        static FdmSchemeDesc TrBDF2();

        /*! Fully implicit Euler intended for use with
            FdmFittedBlackScholesOp.  Time stepping is identical to
            ImplicitEuler; the distinct enum value signals the solver
            layer to select the exponentially fitted spatial operator.
            Convergence: |u-U| <= M(h+k) with M independent of sigma.
        */
        static FdmSchemeDesc FittedImplicitEuler();

        /*! Rannacher startup: \a rannacherSteps of fully implicit
            Euler followed by Crank-Nicolson for the remaining steps.
            This damps oscillations from non-smooth initial data
            (payoff kinks at the strike) while preserving second-order
            accuracy for the bulk of the time grid.  The implicit step
            count is encoded in the \c mu field of the descriptor.

            \note This is a self-contained scheme; set dampingSteps=0
                  in FdmSolverDesc to avoid redundant external damping.
        */
        static FdmSchemeDesc RannacherCrankNicolson(
            Size rannacherSteps = 2);
    };

    class FdmBackwardSolver {
      public:
        typedef FdmLinearOp::array_type array_type;

        FdmBackwardSolver(ext::shared_ptr<FdmLinearOpComposite> map,
                          FdmBoundaryConditionSet bcSet,
                          const ext::shared_ptr<FdmStepConditionComposite>& condition,
                          const FdmSchemeDesc& schemeDesc);

        void rollback(array_type& a,
                      Time from, Time to,
                      Size steps, Size dampingSteps);

      protected:
        const ext::shared_ptr<FdmLinearOpComposite> map_;
        const FdmBoundaryConditionSet bcSet_;
        const ext::shared_ptr<FdmStepConditionComposite> condition_;
        const FdmSchemeDesc schemeDesc_;
    };
}

#endif
```

---

**File 2: `solvers/fdmbackwardsolver.cpp` (MODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2009 Andreas Gaida
 Copyright (C) 2009 Ralph Schreyer
 Copyright (C) 2009 Klaus Spanderen

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

/*! \file fdmbackwardsolver.cpp
*/

#include <ql/mathconstants.hpp>
#include <ql/methods/finitedifferences/finitedifferencemodel.hpp>
#include <ql/methods/finitedifferences/schemes/craigsneydscheme.hpp>
#include <ql/methods/finitedifferences/schemes/cranknicolsonscheme.hpp>
#include <ql/methods/finitedifferences/schemes/douglasscheme.hpp>
#include <ql/methods/finitedifferences/schemes/expliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/schemes/hundsdorferscheme.hpp>
#include <ql/methods/finitedifferences/schemes/impliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/schemes/methodoflinesscheme.hpp>
#include <ql/methods/finitedifferences/schemes/modifiedcraigsneydscheme.hpp>
#include <ql/methods/finitedifferences/schemes/trbdf2scheme.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <cmath>
#include <utility>


namespace QuantLib {

    FdmSchemeDesc::FdmSchemeDesc(FdmSchemeType aType, Real aTheta, Real aMu)
    : type(aType), theta(aTheta), mu(aMu) { }

    FdmSchemeDesc FdmSchemeDesc::Douglas() { return {FdmSchemeDesc::DouglasType, 0.5, 0.0}; }

    FdmSchemeDesc FdmSchemeDesc::CrankNicolson() {
        return {FdmSchemeDesc::CrankNicolsonType, 0.5, 0.0};
    }

    FdmSchemeDesc FdmSchemeDesc::CraigSneyd() { return {FdmSchemeDesc::CraigSneydType, 0.5, 0.5}; }

    FdmSchemeDesc FdmSchemeDesc::ModifiedCraigSneyd() {
        return {FdmSchemeDesc::ModifiedCraigSneydType, 1.0 / 3.0, 1.0 / 3.0};
    }

    FdmSchemeDesc FdmSchemeDesc::Hundsdorfer() {
        return {FdmSchemeDesc::HundsdorferType, 0.5 + std::sqrt(3.0) / 6, 0.5};
    }

    FdmSchemeDesc FdmSchemeDesc::ModifiedHundsdorfer() {
        return {FdmSchemeDesc::HundsdorferType, 1.0 - std::sqrt(2.0) / 2, 0.5};
    }

    FdmSchemeDesc FdmSchemeDesc::ExplicitEuler() {
        return {FdmSchemeDesc::ExplicitEulerType, 0.0, 0.0};
    }

    FdmSchemeDesc FdmSchemeDesc::ImplicitEuler() {
        return {FdmSchemeDesc::ImplicitEulerType, 0.0, 0.0};
    }

    FdmSchemeDesc FdmSchemeDesc::MethodOfLines(Real eps, Real relInitStepSize) {
        return {FdmSchemeDesc::MethodOfLinesType, eps, relInitStepSize};
    }

    FdmSchemeDesc FdmSchemeDesc::TrBDF2() { return {FdmSchemeDesc::TrBDF2Type, 2 - M_SQRT2, 1e-8}; }

    FdmSchemeDesc FdmSchemeDesc::FittedImplicitEuler() {
        return {FdmSchemeDesc::FittedImplicitEulerType, 0.0, 0.0};
    }

    FdmSchemeDesc FdmSchemeDesc::RannacherCrankNicolson(Size rannacherSteps) {
        return {FdmSchemeDesc::RannacherCNType, 0.5, Real(rannacherSteps)};
    }

    FdmBackwardSolver::FdmBackwardSolver(
        ext::shared_ptr<FdmLinearOpComposite> map,
        FdmBoundaryConditionSet bcSet,
        const ext::shared_ptr<FdmStepConditionComposite>& condition,
        const FdmSchemeDesc& schemeDesc)
    : map_(std::move(map)), bcSet_(std::move(bcSet)),
      condition_((condition) != nullptr ?
                     condition :
                     ext::make_shared<FdmStepConditionComposite>(
                         std::list<std::vector<Time> >(), FdmStepConditionComposite::Conditions())),
      schemeDesc_(schemeDesc) {}

    void FdmBackwardSolver::rollback(FdmBackwardSolver::array_type& rhs,
                                     Time from, Time to,
                                     Size steps, Size dampingSteps) {

        const Time deltaT = from - to;
        const Size allSteps = steps + dampingSteps;
        const Time dampingTo = from - (deltaT*dampingSteps)/allSteps;

        /* External damping: a few fully-implicit startup steps before
           the main scheme.  Skipped for scheme types that are already
           fully implicit or that handle their own startup (Rannacher). */
        if (   (dampingSteps != 0U)
            && schemeDesc_.type != FdmSchemeDesc::ImplicitEulerType
            && schemeDesc_.type != FdmSchemeDesc::FittedImplicitEulerType
            && schemeDesc_.type != FdmSchemeDesc::RannacherCNType) {
            ImplicitEulerScheme implicitEvolver(map_, bcSet_);
            FiniteDifferenceModel<ImplicitEulerScheme>
                    dampingModel(implicitEvolver, condition_->stoppingTimes());
            dampingModel.rollback(rhs, from, dampingTo,
                                  dampingSteps, *condition_);
        }

        switch (schemeDesc_.type) {
          case FdmSchemeDesc::HundsdorferType:
            {
                HundsdorferScheme hsEvolver(schemeDesc_.theta, schemeDesc_.mu,
                                            map_, bcSet_);
                FiniteDifferenceModel<HundsdorferScheme>
                               hsModel(hsEvolver, condition_->stoppingTimes());
                hsModel.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::DouglasType:
            {
                DouglasScheme dsEvolver(schemeDesc_.theta, map_, bcSet_);
                FiniteDifferenceModel<DouglasScheme>
                               dsModel(dsEvolver, condition_->stoppingTimes());
                dsModel.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::CrankNicolsonType:
            {
              CrankNicolsonScheme cnEvolver(schemeDesc_.theta, map_, bcSet_);
              FiniteDifferenceModel<CrankNicolsonScheme>
                             cnModel(cnEvolver, condition_->stoppingTimes());
              cnModel.rollback(rhs, dampingTo, to, steps, *condition_);

            }
            break;
          case FdmSchemeDesc::CraigSneydType:
            {
                CraigSneydScheme csEvolver(schemeDesc_.theta, schemeDesc_.mu,
                                           map_, bcSet_);
                FiniteDifferenceModel<CraigSneydScheme>
                               csModel(csEvolver, condition_->stoppingTimes());
                csModel.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::ModifiedCraigSneydType:
            {
                ModifiedCraigSneydScheme csEvolver(schemeDesc_.theta,
                                                   schemeDesc_.mu,
                                                   map_, bcSet_);
                FiniteDifferenceModel<ModifiedCraigSneydScheme>
                              mcsModel(csEvolver, condition_->stoppingTimes());
                mcsModel.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::ImplicitEulerType:
            {
                ImplicitEulerScheme implicitEvolver(map_, bcSet_);
                FiniteDifferenceModel<ImplicitEulerScheme>
                   implicitModel(implicitEvolver, condition_->stoppingTimes());
                implicitModel.rollback(rhs, from, to, allSteps, *condition_);
            }
            break;
          case FdmSchemeDesc::ExplicitEulerType:
            {
                ExplicitEulerScheme explicitEvolver(map_, bcSet_);
                FiniteDifferenceModel<ExplicitEulerScheme>
                   explicitModel(explicitEvolver, condition_->stoppingTimes());
                explicitModel.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::MethodOfLinesType:
            {
                MethodOfLinesScheme methodOfLines(
                    schemeDesc_.theta, schemeDesc_.mu, map_, bcSet_);
                FiniteDifferenceModel<MethodOfLinesScheme>
                   molModel(methodOfLines, condition_->stoppingTimes());
                molModel.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::TrBDF2Type:
            {
                const FdmSchemeDesc trDesc
                    = FdmSchemeDesc::CraigSneyd();

                const ext::shared_ptr<CraigSneydScheme> hsEvolver(
                    ext::make_shared<CraigSneydScheme>(
                        trDesc.theta, trDesc.mu, map_, bcSet_));

                TrBDF2Scheme<CraigSneydScheme> trBDF2(
                    schemeDesc_.theta, map_, hsEvolver, bcSet_,schemeDesc_.mu);

                FiniteDifferenceModel<TrBDF2Scheme<CraigSneydScheme> >
                   trBDF2Model(trBDF2, condition_->stoppingTimes());
                trBDF2Model.rollback(rhs, dampingTo, to, steps, *condition_);
            }
            break;
          case FdmSchemeDesc::FittedImplicitEulerType:
            {
                /* The exponential fitting lives in the spatial operator
                   (FdmFittedBlackScholesOp); the time stepper is plain
                   implicit Euler.  Using from/allSteps (like the
                   standard ImplicitEulerType) because external damping
                   is skipped for fully-implicit schemes.              */
                ImplicitEulerScheme implicitEvolver(map_, bcSet_);
                FiniteDifferenceModel<ImplicitEulerScheme>
                   implicitModel(implicitEvolver, condition_->stoppingTimes());
                implicitModel.rollback(rhs, from, to, allSteps, *condition_);
            }
            break;
          case FdmSchemeDesc::RannacherCNType:
            {
                /* Rannacher startup: the first rannacherSteps are fully
                   implicit Euler (to damp oscillations from payoff
                   kinks), then the remaining steps use Crank-Nicolson.
                   This is self-contained; external damping is skipped
                   and allSteps covers from → to.                      */
                const auto muRound = std::lround(schemeDesc_.mu);
                const Size rannacherSteps =
                    std::min(allSteps,
                             muRound > 0 ? Size(muRound) : Size(0));
                const Size cnSteps = allSteps - rannacherSteps;

                if (rannacherSteps > 0) {
                    const Time rannacherEnd =
                        from - (deltaT * rannacherSteps) / allSteps;

                    ImplicitEulerScheme implicitEvolver(map_, bcSet_);
                    FiniteDifferenceModel<ImplicitEulerScheme>
                        rannacherModel(implicitEvolver,
                                       condition_->stoppingTimes());
                    rannacherModel.rollback(rhs, from, rannacherEnd,
                                            rannacherSteps, *condition_);

                    if (cnSteps > 0) {
                        CrankNicolsonScheme cnEvolver(
                            schemeDesc_.theta, map_, bcSet_);
                        FiniteDifferenceModel<CrankNicolsonScheme>
                            cnModel(cnEvolver,
                                    condition_->stoppingTimes());
                        cnModel.rollback(rhs, rannacherEnd, to,
                                         cnSteps, *condition_);
                    }
                } else {
                    /* mu <= 0: degenerate — pure Crank-Nicolson */
                    CrankNicolsonScheme cnEvolver(
                        schemeDesc_.theta, map_, bcSet_);
                    FiniteDifferenceModel<CrankNicolsonScheme>
                        cnModel(cnEvolver, condition_->stoppingTimes());
                    cnModel.rollback(rhs, from, to,
                                     allSteps, *condition_);
                }
            }
            break;
          default:
            QL_FAIL("Unknown scheme type");
        }
    }
}
```

---

**File 3: `schemes/cranknicolsonscheme.hpp` (MODIFIED — comment added)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2019 Klaus Spanderen

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

/*! \file cranknicolsonscheme.hpp
    \brief Crank-Nicolson scheme
*/

#ifndef quantlib_crank_nicolson_scheme_hpp
#define quantlib_crank_nicolson_scheme_hpp

#include <ql/methods/finitedifferences/schemes/impliciteulerscheme.hpp>

namespace QuantLib {

    /*! In one dimension the Crank-Nicolson scheme is equivalent to the
        Douglas scheme and in higher dimensions it is usually inferior to
        operator splitting methods like Craig-Sneyd or Hundsdorfer-Verwer.

        \par Monotonicity with exponential fitting
        When the spatial operator passed to this scheme is an instance of
        FdmFittedBlackScholesOp (the exponentially fitted discretization
        from Duffy, Wilmott 2004), the composite scheme gains a monotone
        (M-matrix) spatial discretization for all mesh Peclet numbers.
        The time-averaging inherent in Crank-Nicolson is unchanged; for
        non-smooth initial data (payoff kinks at the strike) a
        Rannacher-style startup — a few fully-implicit steps before
        switching to CN — is recommended.  See
        FdmSchemeDesc::RannacherCrankNicolson().
    */
    class ExplicitEulerScheme;

    class CrankNicolsonScheme  {
      public:
        // typedefs
        typedef OperatorTraits<FdmLinearOp> traits;
        typedef traits::operator_type operator_type;
        typedef traits::array_type array_type;
        typedef traits::bc_set bc_set;
        typedef traits::condition_type condition_type;

        // constructors
        CrankNicolsonScheme(
            Real theta,
            const ext::shared_ptr<FdmLinearOpComposite>& map,
            const bc_set& bcSet = bc_set(),
            Real relTol = 1e-8,
            ImplicitEulerScheme::SolverType solverType
                = ImplicitEulerScheme::BiCGstab);

        void step(array_type& a, Time t);
        void setStep(Time dt);

        Size numberOfIterations() const;
      protected:
        Real dt_;
        const Real theta_;
        const ext::shared_ptr<ExplicitEulerScheme> explicit_;
        const ext::shared_ptr<ImplicitEulerScheme> implicit_;
    };
}

#endif
```

---

**File 4: `schemes/cranknicolsonscheme.cpp` (UNMODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2019 Klaus Spanderen

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

#include <ql/methods/finitedifferences/schemes/expliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/schemes/cranknicolsonscheme.hpp>

namespace QuantLib {
    CrankNicolsonScheme::CrankNicolsonScheme(
        Real theta,
        const ext::shared_ptr<FdmLinearOpComposite> & map,
        const bc_set& bcSet,
        Real relTol,
        ImplicitEulerScheme::SolverType solverType)
    : dt_(Null<Real>()),
      theta_(theta),
      explicit_(ext::make_shared<ExplicitEulerScheme>(map, bcSet)),
      implicit_(ext::make_shared<ImplicitEulerScheme>(
          map, bcSet, relTol, solverType)) {
    }

    void CrankNicolsonScheme::step(array_type& a, Time t) {
        QL_REQUIRE(t-dt_ > -1e-8, "a step towards negative time given");

        if (theta_ != 1.0)
            explicit_->step(a, t, 1.0-theta_);

        if (theta_ != 0.0)
            implicit_->step(a, t, theta_);
    }

    void CrankNicolsonScheme::setStep(Time dt) {
        dt_ = dt;
        explicit_->setStep(dt_);
        implicit_->setStep(dt_);
    }

    Size CrankNicolsonScheme::numberOfIterations() const {
        return implicit_->numberOfIterations();
    }
}
```

---

**File 5: `cranknicolson.hpp` (MODIFIED — comment block added)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl

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

/*! \file cranknicolson.hpp
    \brief Crank-Nicolson scheme for finite difference methods
*/

#ifndef quantlib_crank_nicolson_hpp
#define quantlib_crank_nicolson_hpp

#include <ql/methods/finitedifferences/mixedscheme.hpp>

namespace QuantLib {

    //! Crank-Nicolson scheme for finite difference methods
    /*! In this implementation, the passed operator must be derived
        from either TimeConstantOperator or TimeDependentOperator.
        Also, it must implement at least the following interface:

        \code
        typedef ... array_type;

        // copy constructor/assignment
        // (these will be provided by the compiler if none is defined)
        Operator(const Operator&);
        Operator& operator=(const Operator&);

        // inspectors
        Size size();

        // modifiers
        void setTime(Time t);

        // operator interface
        array_type applyTo(const array_type&);
        array_type solveFor(const array_type&);
        static Operator identity(Size size);

        // operator algebra
        Operator operator*(Real, const Operator&);
        Operator operator+(const Operator&, const Operator&);
        Operator operator+(const Operator&, const Operator&);
        \endcode

        \warning The differential operator must be linear for
                 this evolver to work.

        \note This is a legacy evolver using centered spatial
              differences, which can produce spurious oscillations
              when the mesh Peclet number is large (convection-
              dominated regime / small volatility).  The new
              finite-differences framework provides
              FdmFittedBlackScholesOp as an exponentially fitted
              spatial operator (Duffy, Wilmott 2004) that guarantees
              a monotone spatial discretization for all Peclet
              numbers.  It can be paired with CrankNicolsonScheme,
              ImplicitEulerScheme, or the Rannacher startup variant
              (FdmSchemeDesc::RannacherCrankNicolson).  See also
              RichardsonExtrapolationScheme for recovering second-
              order time accuracy from a first-order base scheme.

        \ingroup findiff
    */
    template <class Operator>
    class CrankNicolson : public MixedScheme<Operator> {
      public:
        // typedefs
        typedef OperatorTraits<Operator> traits;
        typedef typename traits::operator_type operator_type;
        typedef typename traits::array_type array_type;
        typedef typename traits::bc_set bc_set;
        typedef typename traits::condition_type condition_type;
        // constructors
        CrankNicolson(const operator_type& L,
                      const bc_set& bcs)
        : MixedScheme<Operator>(L, 0.5, bcs) {}
    };

}


#endif
```

---

**File 6: `mixedscheme.hpp` (UNMODIFIED)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2002, 2003 Ferdinando Ametrano
 Copyright (C) 2000, 2001, 2002, 2003 RiskMap srl

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

/*! \file mixedscheme.hpp
    \brief Mixed (explicit/implicit) scheme for finite difference methods
*/

#ifndef quantlib_mixed_scheme_hpp
#define quantlib_mixed_scheme_hpp

#include <ql/methods/finitedifferences/finitedifferencemodel.hpp>
#include <utility>

namespace QuantLib {

    //! Mixed (explicit/implicit) scheme for finite difference methods
    /*! In this implementation, the passed operator must be derived
        from either TimeConstantOperator or TimeDependentOperator.
        Also, it must implement at least the following interface:

        \code
        typedef ... array_type;

        // copy constructor/assignment
        // (these will be provided by the compiler if none is defined)
        Operator(const Operator&);
        Operator& operator=(const Operator&);

        // inspectors
        Size size();

        // modifiers
        void setTime(Time t);

        // operator interface
        array_type applyTo(const array_type&);
        array_type solveFor(const array_type&);
        static Operator identity(Size size);

        // operator algebra
        Operator operator*(Real, const Operator&);
        Operator operator+(const Operator&, const Operator&);
        Operator operator+(const Operator&, const Operator&);
        \endcode

        \warning The differential operator must be linear for
                 this evolver to work.

        \todo
        - derive variable theta schemes
        - introduce multi time-level schemes.

        \ingroup findiff
    */
    template <class Operator>
    class MixedScheme  {
      public:
        // typedefs
        typedef OperatorTraits<Operator> traits;
        typedef typename traits::operator_type operator_type;
        typedef typename traits::array_type array_type;
        typedef typename traits::bc_set bc_set;
        typedef typename traits::condition_type condition_type;
        // constructors
        MixedScheme(const operator_type& L, Real theta, bc_set bcs)
        : L_(L), I_(operator_type::identity(L.size())), dt_(0.0), theta_(theta),
          bcs_(std::move(bcs)) {}
        void step(array_type& a,
                  Time t);
        void setStep(Time dt) {
            dt_ = dt;
            if (theta_!=1.0) // there is an explicit part
                explicitPart_ = I_-((1.0-theta_) * dt_)*L_;
            if (theta_!=0.0) // there is an implicit part
                implicitPart_ = I_+(theta_ * dt_)*L_;
        }
      protected:
        operator_type L_, I_, explicitPart_, implicitPart_;
        Time dt_;
        Real theta_;
        bc_set bcs_;
    };


    // inline definitions

    template <class Operator>
    inline void MixedScheme<Operator>::step(array_type& a, Time t) {
        Size i;
        for (i=0; i<bcs_.size(); i++)
            bcs_[i]->setTime(t);
        if (theta_!=1.0) { // there is an explicit part
            if (L_.isTimeDependent()) {
                L_.setTime(t);
                explicitPart_ = I_-((1.0-theta_) * dt_)*L_;
            }
            for (i=0; i<bcs_.size(); i++)
                bcs_[i]->applyBeforeApplying(explicitPart_);
            a = explicitPart_.applyTo(a);
            for (i=0; i<bcs_.size(); i++)
                bcs_[i]->applyAfterApplying(a);
        }
        if (theta_!=0.0) { // there is an implicit part
            if (L_.isTimeDependent()) {
                L_.setTime(t-dt_);
                implicitPart_ = I_+(theta_ * dt_)*L_;
            }
            for (i=0; i<bcs_.size(); i++)
                bcs_[i]->applyBeforeSolving(implicitPart_,a);
            implicitPart_.solveFor(a, a);
            for (i=0; i<bcs_.size(); i++)
                bcs_[i]->applyAfterSolving(a);
        }
    }

}


#endif
```

---

**File 7: `schemes/richardsonextrapolationscheme.hpp` (NEW)**

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2024 QuantLib contributors

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

/*! \file richardsonextrapolationscheme.hpp
    \brief Richardson extrapolation wrapper for finite-difference schemes

    Given a base time-stepping scheme of global order \a p, Richardson
    extrapolation eliminates the leading error term and produces a
    scheme of order \a p+1.  The technique is particularly useful with
    the fully-implicit Euler scheme paired with the exponentially
    fitted spatial operator (FdmFittedBlackScholesOp): the fitted
    implicit scheme converges as O(h+k) uniformly in sigma (Duffy
    2004), and one level of Richardson extrapolation recovers
    O(h+k^2) — matching Crank-Nicolson's temporal accuracy without
    CN's oscillation problems.

    \par Computational cost
    Each outer time step performs three base-scheme evaluations: one
    full step of size dt and two half-steps of size dt/2, plus two
    Array copies for the saved state.

    \par Usage example
    \code
    auto base = ext::make_shared<ImplicitEulerScheme>(map, bcSet);
    RichardsonExtrapolationScheme<ImplicitEulerScheme> rich(base);
    FiniteDifferenceModel<
        RichardsonExtrapolationScheme<ImplicitEulerScheme>>
            model(rich, stoppingTimes);
    model.rollback(rhs, maturity, 0.0, timeSteps, condition);
    \endcode

    \ingroup findiff
*/

#ifndef quantlib_richardson_extrapolation_scheme_hpp
#define quantlib_richardson_extrapolation_scheme_hpp

#include <ql/errors.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/utilities/null.hpp>
#include <ql/methods/finitedifferences/operatortraits.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearop.hpp>
#include <utility>

namespace QuantLib {

    //! Richardson extrapolation wrapper for finite-difference schemes
    /*!
        \tparam BaseScheme  The underlying time-stepping scheme (e.g.
                            ImplicitEulerScheme, CrankNicolsonScheme).
                            Must provide \c step(array_type&,Time) and
                            \c setStep(Time).

        \tparam BaseOrder   Global convergence order of the base scheme
                            (default 1 for implicit Euler, use 2 for CN).
                            Determines the extrapolation weights:
                            U* = (2^p U_{k/2} - U_k) / (2^p - 1).
    */
    template <class BaseScheme, Size BaseOrder = 1>
    class RichardsonExtrapolationScheme {
      public:
        // typedefs required by FiniteDifferenceModel
        typedef OperatorTraits<FdmLinearOp> traits;
        typedef typename traits::operator_type operator_type;
        typedef typename traits::array_type    array_type;
        typedef typename traits::bc_set        bc_set;
        typedef typename traits::condition_type condition_type;

        //! Construct from an externally built base scheme.
        explicit RichardsonExtrapolationScheme(
                        ext::shared_ptr<BaseScheme> baseScheme)
        : dt_(Null<Real>()),
          baseScheme_(std::move(baseScheme)) {}

        void step(array_type& a, Time t);
        void setStep(Time dt) { dt_ = dt; }

      private:
        Time dt_;
        ext::shared_ptr<BaseScheme> baseScheme_;
    };


    // ---------------------------------------------------------------
    //  template definitions
    // ---------------------------------------------------------------

    template <class BaseScheme, Size BaseOrder>
    void RichardsonExtrapolationScheme<BaseScheme, BaseOrder>::step(
                                                array_type& a, Time t) {

        QL_REQUIRE(t - dt_ > -1e-8,
                   "a step towards negative time given");

        /*  Richardson extrapolation for a scheme of global order p:
            -------------------------------------------------------
            U_k       = u + c * k^p + O(k^{p+1})      (one full step)
            U_{k/2,2} = u + c * (k/2)^p * 2 + ...     (two half-steps)
                      = u + c * k^p / 2^{p-1} + O(k^{p+1})

            Eliminating the leading error term:
            U* = (2^p * U_{k/2,2} - U_k) / (2^p - 1)
               = u + O(k^{p+1})

            For p = 1 (implicit Euler):  U* = 2 U_{k/2} - U_k
            For p = 2 (Crank-Nicolson):  U* = (4 U_{k/2} - U_k) / 3
        */

        // Save the initial state
        const array_type a0(a);

        // 1) One full step of size dt
        baseScheme_->setStep(dt_);
        array_type aFull(a0);
        baseScheme_->step(aFull, t);

        // 2) Two half-steps of size dt/2
        const Time halfDt = 0.5 * dt_;
        baseScheme_->setStep(halfDt);
        a = a0;
        baseScheme_->step(a, t);               //  t  →  t - dt/2
        baseScheme_->step(a, t - halfDt);      //  t - dt/2  →  t - dt

        // 3) Extrapolate:  U* = (2^p a - aFull) / (2^p - 1)
        const Real twoP = Real(Size(1) << BaseOrder);
        const Real alpha =  twoP / (twoP - 1.0);
        const Real beta  =  1.0  / (twoP - 1.0);
        //  a = alpha * a_{half} - beta * a_{full}
        a *= alpha;
        a -= beta * aFull;

        // Restore the original step size for the caller
        baseScheme_->setStep(dt_);
    }

}


#endif
```

---

**File 8: `schemes/all.hpp` (MODIFIED — one include added)**

```cpp
/* This file is automatically generated; do not edit.     */
/* Add the files to be included into Makefile.am instead. */

#include <ql/methods/finitedifferences/schemes/boundaryconditionschemehelper.hpp>
#include <ql/methods/finitedifferences/schemes/craigsneydscheme.hpp>
#include <ql/methods/finitedifferences/schemes/cranknicolsonscheme.hpp>
#include <ql/methods/finitedifferences/schemes/douglasscheme.hpp>
#include <ql/methods/finitedifferences/schemes/expliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/schemes/hundsdorferscheme.hpp>
#include <ql/methods/finitedifferences/schemes/impliciteulerscheme.hpp>
#include <ql/methods/finitedifferences/schemes/methodoflinesscheme.hpp>
#include <ql/methods/finitedifferences/schemes/modifiedcraigsneydscheme.hpp>
#include <ql/methods/finitedifferences/schemes/richardsonextrapolationscheme.hpp>
#include <ql/methods/finitedifferences/schemes/trbdf2scheme.hpp>
```

---

**Design notes for Round 3 (solver layer):**

The two new `FdmSchemeType` values are now available for the solver layer to select between spatial operators. The `FittedImplicitEulerType` signals that `FdmFittedBlackScholesOp` should be constructed instead of `FdmBlackScholesOp`. The `RannacherCNType` is usable with either spatial operator. `RichardsonExtrapolationScheme` is a composable building block that can be integrated into the backward solver enum in a later round, or used directly by advanced callers via `FiniteDifferenceModel<RichardsonExtrapolationScheme<ImplicitEulerScheme>>`.