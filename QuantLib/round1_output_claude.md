
**File 1: `operators/fdmfittedblackscholesop.hpp` (NEW)**

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

/*! \file fdmfittedblackscholesop.hpp
    \brief Exponentially fitted Black-Scholes linear operator

    Implements the Duffy (2004) exponential fitting technique that replaces
    the standard centered second-derivative coefficient sigma with a fitted
    coefficient rho = (mu*h/2) * coth(mu*h/(2*sigma)), guaranteeing an
    M-matrix (monotone) spatial discretization for all mesh Peclet numbers.
*/

#ifndef quantlib_fdm_fitted_black_scholes_op_hpp
#define quantlib_fdm_fitted_black_scholes_op_hpp

#include <ql/payoff.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>

namespace QuantLib {

    class FdmFittedBlackScholesOp : public FdmLinearOpComposite {
      public:
        FdmFittedBlackScholesOp(
            const ext::shared_ptr<FdmMesher>& mesher,
            const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
            Real strike,
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            Size direction = 0,
            ext::shared_ptr<FdmQuantoHelper> quantoHelper
                = ext::shared_ptr<FdmQuantoHelper>(),
            bool useFitting = true);

        Size size() const override;
        void setTime(Time t1, Time t2) override;

        Disposable<Array> apply(const Array& r) const override;
        Disposable<Array> apply_mixed(const Array& r) const override;
        Disposable<Array> apply_direction(Size direction,
                                          const Array& r) const override;
        Disposable<Array> solve_splitting(Size direction,
                                          const Array& r,
                                          Real s) const override;
        Disposable<Array> preconditioner(const Array& r,
                                          Real s) const override;

        Disposable<std::vector<SparseMatrix> > toMatrixDecomp() const override;

      private:
        const ext::shared_ptr<FdmMesher> mesher_;
        const ext::shared_ptr<YieldTermStructure> rTS_, qTS_;
        const ext::shared_ptr<BlackVolTermStructure> volTS_;
        const ext::shared_ptr<LocalVolTermStructure> localVol_;
        const Array x_;
        const Array localH_;
        const FirstDerivativeOp  dxMap_;
        const TripleBandLinearOp dxxMap_;
        TripleBandLinearOp mapT_;
        const Real strike_;
        const Real illegalLocalVolOverwrite_;
        const Size direction_;
        const ext::shared_ptr<FdmQuantoHelper> quantoHelper_;
        const bool useFitting_;
    };
}


#endif
```

---

**File 2: `operators/fdmfittedblackscholesop.cpp` (NEW)**

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

#include <ql/math/functional.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmfittedblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <cmath>
#include <utility>

namespace QuantLib {

    namespace {

        /*! Precompute the local representative grid spacing at every node
            in the given direction.

            At interior nodes the conservative choice max(h+, h-) is used.
            This guarantees that the subsequent fitting factor produces an
            M-matrix on arbitrarily non-uniform grids.  On a uniform grid
            the result coincides with the constant mesh width h.

            At boundary nodes (first and last in the direction) the second-
            derivative stencil is zero, so the fitting factor is irrelevant;
            a safe default of 1.0 is stored.
        */
        Array buildLocalH(const ext::shared_ptr<FdmMesher>& mesher,
                          Size direction) {

            const ext::shared_ptr<FdmLinearOpLayout> layout = mesher->layout();
            Array h(layout->size(), 1.0);

            const FdmLinearOpIterator endIter = layout->end();
            for (FdmLinearOpIterator iter = layout->begin();
                 iter != endIter; ++iter) {

                const Size i  = iter.index();
                const Size co = iter.coordinates()[direction];

                if (co == 0 || co == layout->dim()[direction] - 1) {
                    h[i] = 1.0;
                } else {
                    const Real hm = mesher->dminus(iter, direction);
                    const Real hp = mesher->dplus(iter, direction);
                    h[i] = std::max(hp, hm);
                }
            }
            return h;
        }

        /*! Compute the exponential fitting factor

                rho = (mu * h / 2) * coth( mu * h / (2 * sigma) )
                    = sigma * z * coth(z),      z = mu*h / (2*sigma)

            with five numerically safe branches:

            1.  |mu|  < 1e-12          =>  rho = sigma        (pure diffusion)
            2.  sigma < 1e-12          =>  rho = |mu|*h/2     (pure convection / upwind)
            3.  |z|   > 20             =>  rho = sigma*|z|     (large Peclet, coth -> sgn)
            4.  |z|   < 1e-4           =>  Taylor:  z*coth(z) ~ 1 + z^2/3 - z^4/45
            5.  otherwise              =>  direct formula via exp(2z)

            The result is always non-negative, which (together with the
            M-matrix sign structure of the centered stencils) guarantees
            a monotone discrete operator.
        */
        Real fittingFactor(Real mu, Real sigma, Real h) {

            /* Case 1 — vanishing drift (pure diffusion limit).
               lim_{mu->0} (mu*h/2)*coth(mu*h/(2*sigma)) = sigma
               via lim_{x->0} x*coth(x) = 1                         */
            if (std::fabs(mu) < 1e-12) {
                return std::max(sigma, 0.0);
            }

            /* Case 2 — vanishing diffusion (pure convection limit).
               lim_{sigma->0+} rho = |mu|*h/2  (upwind)              */
            if (sigma < 1e-12) {
                return std::fabs(mu) * h * 0.5;
            }

            const Real z = mu * h / (2.0 * sigma);

            /* Case 3 — large |z|: avoid overflow in exp(2z).
               For |z| > 20 we have coth(z) = sign(z) to double
               precision, so z*coth(z) = |z|.                        */
            if (std::fabs(z) > 20.0) {
                return sigma * std::fabs(z);
            }

            /* Case 4 — small |z|: Taylor expansion.
               z*coth(z) = 1 + z^2/3 - z^4/45 + 2*z^6/945 - ...
               Using three terms gives a relative error < 1e-20
               for |z| < 1e-4.                                       */
            if (std::fabs(z) < 1e-4) {
                const Real z2 = z * z;
                const Real z4 = z2 * z2;
                return sigma * (1.0 + z2 / 3.0 - z4 / 45.0);
            }

            /* Case 5 — general formula.
               coth(z) = (e^{2z} + 1) / (e^{2z} - 1)                */
            const Real e2z   = std::exp(2.0 * z);
            const Real cothz = (e2z + 1.0) / (e2z - 1.0);
            return sigma * z * cothz;
        }

    } // anonymous namespace


    FdmFittedBlackScholesOp::FdmFittedBlackScholesOp(
        const ext::shared_ptr<FdmMesher>& mesher,
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& bsProcess,
        Real strike,
        bool localVol,
        Real illegalLocalVolOverwrite,
        Size direction,
        ext::shared_ptr<FdmQuantoHelper> quantoHelper,
        bool useFitting)
    : mesher_(mesher),
      rTS_(bsProcess->riskFreeRate().currentLink()),
      qTS_(bsProcess->dividendYield().currentLink()),
      volTS_(bsProcess->blackVolatility().currentLink()),
      localVol_((localVol) ? bsProcess->localVolatility().currentLink()
                           : ext::shared_ptr<LocalVolTermStructure>()),
      x_((localVol) ? Array(Exp(mesher->locations(direction))) : Array()),
      localH_(buildLocalH(mesher, direction)),
      dxMap_(FirstDerivativeOp(direction, mesher)),
      dxxMap_(SecondDerivativeOp(direction, mesher)),
      mapT_(direction, mesher),
      strike_(strike),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite),
      direction_(direction),
      quantoHelper_(std::move(quantoHelper)),
      useFitting_(useFitting) {}


    void FdmFittedBlackScholesOp::setTime(Time t1, Time t2) {
        const Rate r = rTS_->forwardRate(t1, t2, Continuous).rate();
        const Rate q = qTS_->forwardRate(t1, t2, Continuous).rate();

        const ext::shared_ptr<FdmLinearOpLayout> layout = mesher_->layout();
        const Size n = layout->size();

        // ------------------------------------------------------------------
        // Step 1.  Compute vol^2 at every grid node.
        // ------------------------------------------------------------------
        Array v(n);

        if (localVol_ != nullptr) {
            const FdmLinearOpIterator endIter = layout->end();
            for (FdmLinearOpIterator iter = layout->begin();
                 iter != endIter; ++iter) {
                const Size i = iter.index();

                if (illegalLocalVolOverwrite_ < 0.0) {
                    v[i] = square<Real>()(
                        localVol_->localVol(0.5*(t1+t2), x_[i], true));
                }
                else {
                    try {
                        v[i] = square<Real>()(
                            localVol_->localVol(0.5*(t1+t2), x_[i], true));
                    } catch (Error&) {
                        v[i] = square<Real>()(illegalLocalVolOverwrite_);
                    }
                }
            }
        } else {
            const Real vConst =
                volTS_->blackForwardVariance(t1, t2, strike_) / (t2 - t1);
            for (Size i = 0; i < n; ++i) {
                v[i] = vConst;
            }
        }

        // ------------------------------------------------------------------
        // Step 2.  Diffusion coefficient  sigma = vol^2 / 2
        //          Drift coefficient      mu    = r - q - vol^2 / 2
        // ------------------------------------------------------------------
        Array sigma(n);
        Array mu(n);
        for (Size i = 0; i < n; ++i) {
            sigma[i] = 0.5 * v[i];
            mu[i]    = r - q - sigma[i];
        }

        // ------------------------------------------------------------------
        // Step 3.  Quanto adjustment to the drift (if applicable).
        // ------------------------------------------------------------------
        if (quantoHelper_ != nullptr) {
            mu -= quantoHelper_->quantoAdjustment(Sqrt(v), t1, t2);
        }

        // ------------------------------------------------------------------
        // Step 4.  Assemble the spatial operator  L = mu*D0 + coeff*D+D- - r
        //
        //   Fitted:   coeff = rho  (exponential fitting factor)
        //   Unfitted: coeff = sigma  (standard centered differences)
        // ------------------------------------------------------------------
        if (useFitting_) {
            Array rho(n);
            for (Size i = 0; i < n; ++i) {
                rho[i] = fittingFactor(mu[i], sigma[i], localH_[i]);
            }
            mapT_.axpyb(mu, dxMap_, dxxMap_.mult(rho), Array(1, -r));
        } else {
            mapT_.axpyb(mu, dxMap_, dxxMap_.mult(sigma), Array(1, -r));
        }
    }


    Size FdmFittedBlackScholesOp::size() const { return 1U; }

    Disposable<Array> FdmFittedBlackScholesOp::apply(
                                                const Array& u) const {
        return mapT_.apply(u);
    }

    Disposable<Array> FdmFittedBlackScholesOp::apply_direction(
                                       Size direction,
                                       const Array& r) const {
        if (direction == direction_)
            return mapT_.apply(r);
        else {
            Array retVal(r.size(), 0.0);
            return retVal;
        }
    }

    Disposable<Array> FdmFittedBlackScholesOp::apply_mixed(
                                                const Array& r) const {
        Array retVal(r.size(), 0.0);
        return retVal;
    }

    Disposable<Array> FdmFittedBlackScholesOp::solve_splitting(
                                               Size direction,
                                               const Array& r,
                                               Real dt) const {
        if (direction == direction_)
            return mapT_.solve_splitting(r, dt, 1.0);
        else {
            Array retVal(r);
            return retVal;
        }
    }

    Disposable<Array> FdmFittedBlackScholesOp::preconditioner(
                                                        const Array& r,
                                                        Real dt) const {
        return solve_splitting(direction_, r, dt);
    }

    Disposable<std::vector<SparseMatrix> >
    FdmFittedBlackScholesOp::toMatrixDecomp() const {
        std::vector<SparseMatrix> retVal(1, mapT_.toMatrix());
        return retVal;
    }

}
```

---

**File 3: `operators/fdmblackscholesop.hpp` (MODIFIED — one include added)**

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

/*! \file fdmblackscholesop.hpp
    \brief Black Scholes linear operator
*/

#ifndef quantlib_fdm_black_scholes_op_hpp
#define quantlib_fdm_black_scholes_op_hpp

#include <ql/payoff.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmfittedblackscholesop.hpp>

namespace QuantLib {

    class FdmBlackScholesOp : public FdmLinearOpComposite {
      public:
        FdmBlackScholesOp(
            const ext::shared_ptr<FdmMesher>& mesher,
            const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
            Real strike,
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            Size direction = 0,
            ext::shared_ptr<FdmQuantoHelper> quantoHelper = ext::shared_ptr<FdmQuantoHelper>());

        Size size() const override;
        void setTime(Time t1, Time t2) override;

        Disposable<Array> apply(const Array& r) const override;
        Disposable<Array> apply_mixed(const Array& r) const override;
        Disposable<Array> apply_direction(Size direction, const Array& r) const override;
        Disposable<Array> solve_splitting(Size direction, const Array& r, Real s) const override;
        Disposable<Array> preconditioner(const Array& r, Real s) const override;

        Disposable<std::vector<SparseMatrix> > toMatrixDecomp() const override;

      private:
        const ext::shared_ptr<FdmMesher> mesher_;
        const ext::shared_ptr<YieldTermStructure> rTS_, qTS_;
        const ext::shared_ptr<BlackVolTermStructure> volTS_;
        const ext::shared_ptr<LocalVolTermStructure> localVol_;
        const Array x_;
        const FirstDerivativeOp  dxMap_;
        const TripleBandLinearOp dxxMap_;
        TripleBandLinearOp mapT_;
        const Real strike_;
        const Real illegalLocalVolOverwrite_;
        const Size direction_;
        const ext::shared_ptr<FdmQuantoHelper> quantoHelper_;
    };
}

#endif
```

---

**File 4: `operators/fdmblackscholesop.cpp` (UNMODIFIED — preserved exactly)**

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

#include <ql/instruments/payoffs.hpp>
#include <ql/math/functional.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <utility>

namespace QuantLib {

    FdmBlackScholesOp::FdmBlackScholesOp(
        const ext::shared_ptr<FdmMesher>& mesher,
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& bsProcess,
        Real strike,
        bool localVol,
        Real illegalLocalVolOverwrite,
        Size direction,
        ext::shared_ptr<FdmQuantoHelper> quantoHelper)
    : mesher_(mesher), rTS_(bsProcess->riskFreeRate().currentLink()),
      qTS_(bsProcess->dividendYield().currentLink()),
      volTS_(bsProcess->blackVolatility().currentLink()),
      localVol_((localVol) ? bsProcess->localVolatility().currentLink() :
                             ext::shared_ptr<LocalVolTermStructure>()),
      x_((localVol) ? Array(Exp(mesher->locations(direction))) : Array()),
      dxMap_(FirstDerivativeOp(direction, mesher)), dxxMap_(SecondDerivativeOp(direction, mesher)),
      mapT_(direction, mesher), strike_(strike),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite), direction_(direction),
      quantoHelper_(std::move(quantoHelper)) {}

    void FdmBlackScholesOp::setTime(Time t1, Time t2) {
        const Rate r = rTS_->forwardRate(t1, t2, Continuous).rate();
        const Rate q = qTS_->forwardRate(t1, t2, Continuous).rate();

        if (localVol_ != nullptr) {
            const ext::shared_ptr<FdmLinearOpLayout> layout=mesher_->layout();
            const FdmLinearOpIterator endIter = layout->end();

            Array v(layout->size());
            for (FdmLinearOpIterator iter = layout->begin();
                 iter!=endIter; ++iter) {
                const Size i = iter.index();

                if (illegalLocalVolOverwrite_ < 0.0) {
                    v[i] = square<Real>()(
                        localVol_->localVol(0.5*(t1+t2), x_[i], true));
                }
                else {
                    try {
                        v[i] = square<Real>()(
                            localVol_->localVol(0.5*(t1+t2), x_[i], true));
                    } catch (Error&) {
                        v[i] = square<Real>()(illegalLocalVolOverwrite_);
                    }

                }
            }

            if (quantoHelper_ != nullptr) {
                mapT_.axpyb(r - q - 0.5*v
                    - quantoHelper_->quantoAdjustment(Sqrt(v), t1, t2),
                    dxMap_, dxxMap_.mult(0.5*v), Array(1, -r));
            } else {
                mapT_.axpyb(r - q - 0.5*v, dxMap_,
                            dxxMap_.mult(0.5*v), Array(1, -r));
            }
        } else {
            const Real v
                = volTS_->blackForwardVariance(t1, t2, strike_)/(t2-t1);

            if (quantoHelper_ != nullptr) {
                mapT_.axpyb(
                    Array(1, r - q - 0.5*v)
                        - quantoHelper_->quantoAdjustment(
                            Array(1, std::sqrt(v)), t1, t2),
                    dxMap_,
                    dxxMap_.mult(0.5*Array(mesher_->layout()->size(), v)),
                    Array(1, -r));
            } else {
                mapT_.axpyb(Array(1, r - q - 0.5*v), dxMap_,
                    dxxMap_.mult(0.5*Array(mesher_->layout()->size(), v)),
                    Array(1, -r));
            }
        }
    }

    Size FdmBlackScholesOp::size() const { return 1U; }

    Disposable<Array> FdmBlackScholesOp::apply(const Array& u) const {
        return mapT_.apply(u);
    }

    Disposable<Array> FdmBlackScholesOp::apply_direction(Size direction,
                                                    const Array& r) const {
        if (direction == direction_)
            return mapT_.apply(r);
        else {
            Array retVal(r.size(), 0.0);
            return retVal;
        }
    }

    Disposable<Array> FdmBlackScholesOp::apply_mixed(const Array& r) const {
        Array retVal(r.size(), 0.0);
        return retVal;
    }

    Disposable<Array> FdmBlackScholesOp::solve_splitting(Size direction,
                                                const Array& r, Real dt) const {
        if (direction == direction_)
            return mapT_.solve_splitting(r, dt, 1.0);
        else {
            Array retVal(r);
            return retVal;
        }
    }

    Disposable<Array> FdmBlackScholesOp::preconditioner(const Array& r,
                                                        Real dt) const {
        return solve_splitting(direction_, r, dt);
    }

    Disposable<std::vector<SparseMatrix> >
    FdmBlackScholesOp::toMatrixDecomp() const {
        std::vector<SparseMatrix> retVal(1, mapT_.toMatrix());
        return retVal;
    }

}
```

---

**File 5: `operators/all.hpp` (MODIFIED — one include added)**

```cpp
/* This file is automatically generated; do not edit.     */
/* Add the files to be included into Makefile.am instead. */

#include <ql/methods/finitedifferences/operators/fdm2dblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmbatesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmcevop.hpp>
#include <ql/methods/finitedifferences/operators/fdmfittedblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmg2op.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonhullwhiteop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhestonop.hpp>
#include <ql/methods/finitedifferences/operators/fdmcirop.hpp>
#include <ql/methods/finitedifferences/operators/fdmhullwhiteop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmlocalvolfwdop.hpp>
#include <ql/methods/finitedifferences/operators/fdmornsteinuhlenbeckop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopiterator.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/operators/fdmsabrop.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/ninepointlinearop.hpp>
#include <ql/methods/finitedifferences/operators/nthorderderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/numericaldifferentiation.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondordermixedderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
```

---

**Design notes for subsequent rounds:**

The `FdmFittedBlackScholesOp` is a drop-in replacement for `FdmBlackScholesOp` at the spatial operator level. It implements the same `FdmLinearOpComposite` interface and works with every existing time-stepping scheme (`ImplicitEulerScheme`, `CrankNicolsonScheme`, `DouglasScheme`, `HundsdorferScheme`, etc.) without modification. In Round 2, the solver layer (`fdmblackscholessolver.cpp`) can be modified to accept a flag selecting between the standard and fitted operator. In Round 3, the engine layer can expose this choice to end users.

The `useFitting=false` path reproduces the standard `FdmBlackScholesOp` behavior identically (same drift, same diffusion coefficient, same `axpyb` assembly), enabling controlled A/B comparisons on the same code path.

The conservative `max(h⁺, h⁻)` choice for the local grid spacing guarantees the M-matrix property on non-uniform grids produced by `Concentrating1dMesher` and `FdmBlackScholesMesher`. On uniform grids the fitting factor reduces to the textbook formula exactly.