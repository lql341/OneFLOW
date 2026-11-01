/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.

    OneFLOW is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OneFLOW is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OneFLOW.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#pragma once
#include "Ctrl.h"
#include "GridState.h"
#include "Multigrid.h"
#include "NsCom.h"
#include "SolverDef.h"
#include "SolverState.h"
#include "TimeIntegral.h"
#include "HipProductionPolicy.h"

BeginNameSpace( ONEFLOW )

// One decision shared by the two places that must agree about the explicit
// Runge-Kutta bookkeeping:
//   * FieldTaskReg decides whether LOAD_Q / LOAD_RESIDUALS still fill the
//     host FIELD_FLOW / FIELD_RHS slots;
//   * UNsInvFlux decides whether the device residual starts from the uploaded
//     host field or from zero on the device.
// If only one of them changes, the stage silently accumulates on the previous
// stage's residual.  That is a numerical regression, not a performance knob.
//
// The device owns this bookkeeping only when every precondition below holds.
// Otherwise the host slots stay authoritative:
//   * steady explicit multi-stage step (dual time keeps host history levels);
//   * single grid level (multigrid restrict/prolong reads FIELD_FLOW);
//   * finest grid only (coarse grids run the host update);
//   * single local zone, because the validated device chain is one zone per
//     rank;
//   * constant-gamma five-equation NS solver with no chemistry and a single
//     ideal-gas temperature model;
//   * inviscid (the host viscous flux would otherwise add into a residual the
//     device has already zeroed);
//   * no dual-time source (it also accumulates into the host residual).
inline bool SolverSupportsDeviceRkBookkeeping()
{
    return HipProductionBackendSelected()
        && SolverState::solverType == NS_SOLVER
        && nscom.chemModel == 0
        && nscom.nTModel == 1
        && vis_model.vismodel == 0
        && ctrl.idualtime != 1;
}

inline bool DeviceOwnsExplicitRkBookkeeping()
{
#ifdef ONEFLOW_ENABLE_HIP
    const bool stateUpdate =
        HipProductionFeatureEnabled( "ONEFLOW_ENABLE_UNS_HIP_STATE_UPDATE" );
    const bool reconstruction =
        HipProductionFeatureEnabled( "ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION" );
    // The pure seven-argument truth table is unit tested; this call wires the
    // live globals into it so the test and production cannot drift apart.
    return SolverSupportsDeviceRkBookkeeping()
        && HipSkipHostRkSlotCopy(
               HipSkipHostRkSlotCopyRequested(),
               HipDeviceAuthoritativeStateEnabled(),
               stateUpdate && reconstruction,
               GridState::gridLevel == 0,
               MG::nMulti <= 1,
               ctrl.time_integral == MULTI_STAGE,
               ctrl.startStrategy != 2 );
#else
    return false;
#endif
}

EndNameSpace( ONEFLOW )
