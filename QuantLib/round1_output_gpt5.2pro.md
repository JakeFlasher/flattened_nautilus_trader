## `operators/fdmfittedblackscholesop.hpp`

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026

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
*/

#ifndef quantlib_fdm_fitted_black_scholes_op_hpp
#define quantlib_fdm_fitted_black_scholes_op_hpp

#include <ql/payoff.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/utilities/null.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/secondderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>

namespace QuantLib {

    //! Exponentially fitted Black-Scholes linear operator (Duffy, 2004)
    /*! Replaces the diffusion coefficient in front of the second derivative by a
        fitted factor (fitting factor method / exponential fitting) in order to
        improve monotonicity/robustness in convection-dominated regimes.

        The operator is built in log-space \(x = \ln(S)\).
    */
    class FdmFittedBlackScholesOp : public FdmLinearOpComposite {
      public:
        FdmFittedBlackScholesOp(
            const ext::shared_ptr<FdmMesher>& mesher,
            const ext::shared_ptr<GeneralizedBlackScholesProcess>& process,
            Real strike,
            bool localVol = false,
            Real illegalLocalVolOverwrite = -Null<Real>(),
            Size direction = 0,
            ext::shared_ptr<FdmQuantoHelper> quantoHelper = ext::shared_ptr<FdmQuantoHelper>(),
            bool useFitting = true);

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
        const SecondDerivativeOp dxxMap_;
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

## `operators/fdmfittedblackscholesop.cpp`

```cpp
/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026

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

/*! \file fdmfittedblackscholesop.cpp
*/

#include <ql/methods/finitedifferences/operators/fdmfittedblackscholesop.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/instruments/payoffs.hpp>
#include <ql/math/functional.hpp>
#include <ql/errors.hpp>

#include <algorithm>
#include <cmath>
#include <utility>

namespace QuantLib {

    namespace {

        Real effectiveH(const ext::shared_ptr<FdmMesher>& mesher,
                        const FdmLinearOpIterator& iter,
                        Size direction) {
            const Real hm = mesher->dminus(iter, direction);
            const Real hp = mesher->dplus(iter, direction);

            if (hm != Null<Real>() && hp != Null<Real>()) {
                // Non-uniform-grid safeguard:
                // choosing the larger local spacing is the conservative choice
                // to preserve the positivity of off-diagonal coefficients when
                // combining mu*D0 + rho*Dxx on strongly non-uniform meshes.
                return std::max(hm, hp);
            }
            else if (hm != Null<Real>()) {
                return hm;
            }
            else if (hp != Null<Real>()) {
                return hp;
            }
            else {
                // Should not happen for a well-defined 1D mesher with size >= 2.
                return 0.0;
            }
        }

        Real fittedRho(Real mu, Real sigma, Real h) {
            // Implements:
            //   rho = (mu*h/2) * coth(mu*h/(2*sigma))
            //
            // with numerical safeguards as specified.

            QL_REQUIRE(h > 0.0, "non-positive grid step encountered");

            // thresholds per specification
            const Real sigmaEps = 1e-12;
            const Real muEps    = 1e-12;
            const Real xSmall   = 1e-4;

            // If sigma is (almost) zero -> pure convection (upwind limit).
            if (sigma < sigmaEps) {
                if (std::fabs(mu) < muEps) {
                    // Degenerate case: both convection and diffusion ~ 0.
                    // Using rho = sigma is consistent with mu -> 0 limit.
                    return sigma;
                }
                return 0.5 * std::fabs(mu) * h;
            }

            // Pure diffusion limit (mu -> 0): rho -> sigma.
            if (std::fabs(mu) < muEps) {
                return sigma;
            }

            const Real x = mu * h / (2.0 * sigma);

            // Taylor expansion for small x:
            // coth(x) ≈ 1/x + x/3 - x^3/45
            // Prefer the stable equivalent for x*coth(x):
            // x*coth(x) ≈ 1 + x^2/3 - x^4/45
            if (std::fabs(x) < xSmall) {
                const Real x2 = x * x;
                const Real x4 = x2 * x2;
                const Real xCoth = 1.0 + x2 / 3.0 - x4 / 45.0;

                const Real rho = sigma * xCoth;
                return (rho > 0.0) ? rho : 0.0;
            }

            // General case: coth(x) = 1/tanh(x)
            const Real tanhX = std::tanh(x);

            // tanh(x) should not be ~0 here, but protect against division by ~0.
            if (std::fabs(tanhX) < QL_EPSILON) {
                const Real x2 = x * x;
                const Real x4 = x2 * x2;
                const Real xCoth = 1.0 + x2 / 3.0 - x4 / 45.0;

                const Real rho = sigma * xCoth;
                return (rho > 0.0) ? rho : 0.0;
            }

            const Real cothX = 1.0 / tanhX;
            const Real rho = 0.5 * mu * h * cothX;

            return (rho > 0.0) ? rho : 0.0;
        }
    }

    FdmFittedBlackScholesOp::FdmFittedBlackScholesOp(
        const ext::shared_ptr<FdmMesher>& mesher,
        const ext::shared_ptr<GeneralizedBlackScholesProcess>& bsProcess,
        Real strike,
        bool localVol,
        Real illegalLocalVolOverwrite,
        Size direction,
        ext::shared_ptr<FdmQuantoHelper> quantoHelper,
        bool useFitting)
    : mesher_(mesher), rTS_(bsProcess->riskFreeRate().currentLink()),
      qTS_(bsProcess->dividendYield().currentLink()),
      volTS_(bsProcess->blackVolatility().currentLink()),
      localVol_((localVol) ? bsProcess->localVolatility().currentLink() :
                             ext::shared_ptr<LocalVolTermStructure>()),
      x_((localVol) ? Array(Exp(mesher->locations(direction))) : Array()),
      dxMap_(direction, mesher), dxxMap_(direction, mesher),
      mapT_(direction, mesher), strike_(strike),
      illegalLocalVolOverwrite_(illegalLocalVolOverwrite), direction_(direction),
      quantoHelper_(std::move(quantoHelper)), useFitting_(useFitting) {}

    void FdmFittedBlackScholesOp::setTime(Time t1, Time t2) {
        const Rate r = rTS_->forwardRate(t1, t2, Continuous).rate();
        const Rate q = qTS_->forwardRate(t1, t2, Continuous).rate();

        const ext::shared_ptr<FdmLinearOpLayout> layout = mesher_->layout();
        const FdmLinearOpIterator endIter = layout->end();

        Array mu(layout->size());
        Array rho(layout->size());

        if (localVol_ != nullptr) {

            Array v(layout->size());
            for (FdmLinearOpIterator iter = layout->begin();
                 iter != endIter; ++iter) {
                const Size i = iter.index();

                if (illegalLocalVolOverwrite_ < 0.0) {
                    v[i] = square<Real>()(
                        localVol_->localVol(0.5*(t1+t2), x_[i], true));
                } else {
                    try {
                        v[i] = square<Real>()(
                            localVol_->localVol(0.5*(t1+t2), x_[i], true));
                    } catch (Error&) {
                        v[i] = square<Real>()(illegalLocalVolOverwrite_);
                    }
                }
            }

            // quanto adjustment (optional)
            Array qa;
            if (quantoHelper_ != nullptr) {
                qa = quantoHelper_->quantoAdjustment(Sqrt(v), t1, t2);
            }

            for (FdmLinearOpIterator iter = layout->begin();
                 iter != endIter; ++iter) {
                const Size i = iter.index();

                const Real sigma = 0.5 * v[i];           // diffusion coefficient
                Real muj = r - q - sigma;                // convection coefficient

                if (quantoHelper_ != nullptr) {
                    muj -= qa[i];
                }

                mu[i] = muj;

                const Real h = effectiveH(mesher_, iter, direction_);

                if (useFitting_) {
                    rho[i] = fittedRho(muj, sigma, h);
                } else {
                    rho[i] = sigma;
                }
            }

        } else {

            const Real v = volTS_->blackForwardVariance(t1, t2, strike_) / (t2 - t1);

            const Real sigma = 0.5 * v;                 // diffusion coefficient
            Real muScalar = r - q - sigma;              // convection coefficient

            if (quantoHelper_ != nullptr) {
                muScalar -= quantoHelper_->quantoAdjustment(std::sqrt(v), t1, t2);
            }

            for (FdmLinearOpIterator iter = layout->begin();
                 iter != endIter; ++iter) {
                const Size i = iter.index();

                mu[i] = muScalar;

                const Real h = effectiveH(mesher_, iter, direction_);

                if (useFitting_) {
                    rho[i] = fittedRho(muScalar, sigma, h);
                } else {
                    rho[i] = sigma;
                }
            }
        }

        // Assemble fitted operator:
        //   L = rho * Dxx + mu * Dx + b * I,   with b = -r
        mapT_.axpyb(mu, dxMap_, dxxMap_.mult(rho), Array(1, -r));
    }

    Size FdmFittedBlackScholesOp::size() const {
        return 1U;
    }

    Disposable<Array> FdmFittedBlackScholesOp::apply(const Array& r) const {
        return mapT_.apply(r);
    }

    Disposable<Array> FdmFittedBlackScholesOp::apply_direction(
        Size direction, const Array& r) const {

        if (direction == direction_) {
            return mapT_.apply(r);
        }
        else {
            Array retVal(r.size(), 0.0);
            return retVal;
        }
    }

    Disposable<Array> FdmFittedBlackScholesOp::apply_mixed(const Array& r) const {
        Array retVal(r.size(), 0.0);
        return retVal;
    }

    Disposable<Array> FdmFittedBlackScholesOp::solve_splitting(
        Size direction, const Array& r, Real s) const {

        if (direction == direction_) {
            return mapT_.solve_splitting(r, s, 1.0);
        }
        else {
            Array retVal(r);
            return retVal;
        }
    }

    Disposable<Array> FdmFittedBlackScholesOp::preconditioner(
        const Array& r, Real s) const {

        return solve_splitting(direction_, r, s);
    }

    Disposable<std::vector<SparseMatrix> >
    FdmFittedBlackScholesOp::toMatrixDecomp() const {
        std::vector<SparseMatrix> retVal(1, mapT_.toMatrix());
        return retVal;
    }

}
```

## `operators/fdmblackscholesop.hpp` (modified — include only)

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

#include "fdmfittedblackscholesop.hpp"
#include <ql/payoff.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/methods/finitedifferences/utilities/fdmquantohelper.hpp>
#include <ql/methods/finitedifferences/operators/firstderivativeop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>

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

## `operators/fdmblackscholesop.cpp` (unchanged)

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

## `operators/all.hpp` (updated)

```cpp
/* This file is automatically generated; do not edit.     */
/* Add the files to be included into Makefile.am instead. */

#include <ql/methods/finitedifferences/operators/fdm2dblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmbatesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmfittedblackscholesop.hpp>
#include <ql/methods/finitedifferences/operators/fdmcevop.hpp>
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