/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.
\*---------------------------------------------------------------------------*/

#include "NsResidentMacroStep.h"

#include "NsSolverImp.h"
#include "ULhs.h"
#include "UNsUpdate.h"
#include "UTimeStep.h"
#include "Ctrl.h"
#include "GridState.h"
#include "Multigrid.h"
#include "ResidualBasePolicy.h"
#include "SolverDef.h"
#include "SolverState.h"
#include "ZoneState.h"
#include "CmxTask.h"
#include "CmxTaskNames.h"

#include <cstdlib>

BeginNameSpace( ONEFLOW )

namespace
{

bool ResidentMacroStepRequested()
{
    const char * value =
        std::getenv( "ONEFLOW_HIP_RESIDENT_MACROSTEP" );
    return value != nullptr && value[ 0 ] == '1';
}

}

bool NsResidentMacroStepSupported()
{
    if ( ! ResidentMacroStepRequested() ) return false;
    // One macro-step owns the device state for the whole RK loop.  Coarse grids
    // and multigrid plumbing stay on the task path.
    if ( GridState::gridLevel != 0 || MG::nMulti > 1 ) return false;
    if ( SolverState::solverType != NS_SOLVER ) return false;
    return DeviceOwnsExplicitRkBookkeeping();
}

void NsResidentMacroStep()
{
    StringField unused;

    // CALC_TIME_STEP: local spectral radius and time step for this step.
    UTimeStep timeStep;
    timeStep.CalcTimeStep();

    const int nStages = static_cast< int >( ctrl.rk_coef.size() );
    for ( int stage = 0; stage < nStages; ++ stage )
    {
        ctrl.rkStage = stage;
        ctrl.lhscoef = ctrl.rk_coef[ stage ];

        // UPDATE_RESIDUALS: gradient, reconstruction, fused face flux, RHS.
        NsUpdateResiduals( unused );
        // CALC_LHS: scale the resident residual by dt/vol*rkcoef.
        ULhs lhs;
        lhs.CalcLHS( NS_SOLVER );
        // UPDATE_FLOWFIELD: primitive state update on the device.
        UNsUpdate update;
        update.UpdateFlowField( NS_SOLVER );
        // CALC_BOUNDARY: local boundary models on the resident state.
        NsCalcBoundary( unused );
        // UPDATE_INTERFACE_DATA: the explicit stage must publish updated
        // boundary values before the next stage reads its ghosts.  Still one
        // task dispatch per stage (not four), because the exchange itself is
        // owned by the interface task and its lag must stay exactly as in the
        // validated task path.
        if ( ZoneState::nZones > 1 )
        {
            SingleSolverSingleGridTask( kUpdateInterfaceDataTaskName );
        }
    }
    ctrl.rkStage = -1;
}

EndNameSpace
