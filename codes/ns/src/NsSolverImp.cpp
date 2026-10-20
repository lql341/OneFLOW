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

#include "NsSolverImp.h"
#include "SolverImp.h"
#include "UTimeStep.h"
#include "Zone.h"
#include "ZoneState.h"
#include "UnsGrid.h"
#include "GridState.h"
#include "UNsVisFlux.h"
#include "NsRhs.h"
#include "UCom.h"
#include "CmxTask.h"
#include "CmxTaskNames.h"
#include "Iteration.h"
#include "SolverDef.h"
#include "Ctrl.h"
#include "LaminarPlate.h"
#include "TurbPlate.h"
#include "SolverRegister.h"
#include "DataBase.h"
#include "StageProfiler.h"
#include "AccelRuntime.h"
#include "SolverState.h"
#include "Boundary.h"
#include "BcRecord.h"
#include "UNsCom.h"
#ifdef ONEFLOW_ENABLE_HIP
#include "HipFluxBackend.h"
#endif
#include <cstdlib>
#include <string>
#include <iomanip>
#include <memory>
#include <stdexcept>


BeginNameSpace( ONEFLOW )

namespace
{

bool HipBoundaryExplicitlyRequested()
{
    const char * enabled =
        std::getenv( "ONEFLOW_ENABLE_UNS_HIP_BOUNDARY" );
    return enabled != nullptr && enabled[ 0 ] == '1';
}

bool HipBackendSelected()
{
#ifdef ONEFLOW_ENABLE_HIP
    AccelRuntime & runtime = AccelRuntime::Instance();
    return runtime.IsInitialized()
        && runtime.Backend().Kind() == AccelBackendKind::HIP;
#else
    return false;
#endif
}

bool HipBoundarySupported()
{
#ifndef ONEFLOW_ENABLE_HIP
    return false;
#else
    AccelRuntime & runtime = AccelRuntime::Instance();
    if ( ! runtime.IsInitialized()
         || runtime.Backend().Kind() != AccelBackendKind::HIP )
        return false;
    if ( ZoneState::nZones != 1 || ZoneState::nLocal != 1 )
        return false;
    if ( SolverState::solverType != NS_SOLVER || nscom.nEqu != 5
         || vis_model.vismodel != 0 || ctrl.ilim != 0 )
        return false;
    UnsGrid * grid = Zone::GetUnsGrid();
    if ( grid == nullptr || ug.nBFaces <= 0
         || ug.nTCell - ug.nCells != ug.nBFaces
         || ug.bcRecord == nullptr )
        return false;
    for ( int face = 0; face < ug.nBFaces; ++ face )
    {
        const int type = ug.bcRecord->bcType[ face ];
        if ( type < 0 || type == BC::PERIODIC || type == BC::OVERSET
             || type == BC::EXTRAPOLATION || type == BC::SOLID_SURFACE
             || type == BC::SYMMETRY || type == BC::FARFIELD
             || type == BC::INFLOW || type == BC::OUTFLOW
             || type == BC::POLE || type == BC::POLE1
             || type == BC::POLE2 || type == BC::POLE3 )
            continue;
        return false;
    }
    return true;
#endif
}

void CalcHipBoundary()
{
#ifdef ONEFLOW_ENABLE_HIP
    UnsGrid * grid = Zone::GetUnsGrid();
    ug.Init();
    unsf.Init();
    BoundaryStateView view;
    view.nCells = ug.nCells;
    view.nGhostCells = ug.nTCell - ug.nCells;
    view.nBoundaryFaces = ug.nBFaces;
    view.nEquations = nscom.nEqu;
    view.leftCell = ug.lcf->data();
    view.rightCell = ug.rcf->data();
    view.boundaryType = ug.bcRecord->bcType.data();
    view.xNormal = ug.xfn->data();
    view.yNormal = ug.yfn->data();
    view.zNormal = ug.zfn->data();
    view.meshVelocityNormal = ug.vfn->data();
    view.inflow = nscom.inflow.data();
    view.gamma = nscom.gama_ref;
    view.faceOuterNormal = nscom.faceOuterNormal;
    view.cacheKey = grid;
    view.preferResidentState = true;
    for ( int equation = 0; equation < view.nEquations; ++ equation )
    {
        view.q[ equation ] = ( * unsf.q )[ equation ].data();
        if ( unsf.bc_q != nullptr )
            view.bcQ[ equation ] = ( * unsf.bc_q )[ equation ].data();
    }
    HipFluxBackend::Shared().UpdateBoundaryState( view );
#else
    throw std::runtime_error(
        "HIP boundary update was requested without HIP support" );
#endif
}

}

void RegisterNsFunc()
{
    REGISTER_DATA_CLASS( NsInitFinal );
    REGISTER_DATA_CLASS( NsVisual );
    REGISTER_DATA_CLASS( NsCalcTimeStep );
    REGISTER_DATA_CLASS( NsUpdateResiduals );
    REGISTER_DATA_CLASS( NsImplicitMethod );
    REGISTER_DATA_CLASS( NsPostprocess );
    REGISTER_DATA_CLASS( NsFinalPostprocess );
    REGISTER_DATA_CLASS( NsInitSolver );
    REGISTER_DATA_CLASS( NsCalcBoundary );
    REGISTER_DATA_CLASS( DumpHeatFluxCoeff );
    SetPlateTask();
    SetTurbPlateTask();
}

void NsInitFinal( StringField & data )
{
    //NsCalcGamaT( F_INNER );
    //CalcLaminarViscosity( F_INNER );
    //NsCalcBc();
    //NsCalcGamaT( F_GHOST );
    //CalcLaminarViscosity( F_GHOST );

    NsCalcBoundary( data );

    Grid & grid = Zone::GetGridReference();

    if ( Zone::GetCGrid( &grid ) )
    {
        //RestrictAllQ( NS_SOLVER, FLOW_FIELD_INDEX );

        GridState::gridLevel += 1;

        NsCalcBoundary( data );

        GridState::gridLevel -= 1;
    }
}

void NsVisual( StringField & data )
{
    ;
}

void NsCalcBoundary( StringField & data )
{
    if ( vis_model.vismodel > 0 && nscom.nTModel <= 0 )
    {
        throw std::runtime_error(
            "viscous or turbulent NS boundary conditions require "
            "nTModel > 0" );
    }
    const bool explicitRequest = HipBoundaryExplicitlyRequested();
    // Boundary device execution remains an explicit diagnostic opt-in until
    // its multi-stage ordering is covered by the production trace oracle.
    const bool requested = explicitRequest
        && HipBackendSelected() && HipBoundarySupported();
    if ( explicitRequest && ! requested )
    {
        throw std::runtime_error(
            "ONEFLOW_ENABLE_UNS_HIP_BOUNDARY=1 is unsupported for the "
            "current solver, boundary set, or runtime" );
    }
    {
        ScopedStageTimer timer( "boundary_gamma_inner" );
        NsCalcGamaT( F_INNER );
    }
    if ( vis_model.vismodel != 0 )
    {
        ScopedStageTimer timer( "boundary_viscosity_inner" );
        CalcLaminarViscosity( F_INNER );
    }
    if ( requested )
    {
        ScopedStageTimer timer( "boundary_device" );
        CalcHipBoundary();
    }
    else
    {
        ScopedStageTimer timer( "boundary_bc" );
        NsCalcBc();
    }
    {
        ScopedStageTimer timer( "boundary_gamma_ghost" );
        NsCalcGamaT( F_GHOST );
    }
    if ( vis_model.vismodel != 0 )
    {
        ScopedStageTimer timer( "boundary_viscosity_ghost" );
        CalcLaminarViscosity( F_GHOST );
    }
}

void NsCalcTimeStep( StringField & data )
{
    auto uTimeStep = std::make_unique<UTimeStep>();
    uTimeStep->CalcTimeStep();
}

void NsUpdateResiduals( StringField & data )
{
    auto rhs = std::make_unique<NsRhs>();
    rhs->UpdateResiduals();
}

void NsImplicitMethod( StringField & data )
{
    ;
}

void NsPostprocess( StringField & data )
{
    //After every Iter, the first thing to consider is communication.
    CommInterfaceData();

    //The solution and output of residuals need to be judged logically.
    if ( Iteration::ResOk() )
    {
        ONEFLOW::AddCmdToList( kDumpResidualTaskName );
    }

    //The solution and output of aerodynamic force need to be judged logically.
    if ( Iteration::ForceOk() )
    {
        ONEFLOW::AddCmdToList( kDumpAerodynamicTaskName );
    }

    if ( ! Iteration::InnerOk() ) return;

    ONEFLOW::AddCmdToList( kUpdateUnsteadyFlowTaskName );

    if ( Iteration::outerSteps % Iteration::nVisualSave == 0 )
    {
        ONEFLOW::AddCmdToList( kVisualizationTaskName      );
        ONEFLOW::AddCmdToList( kDumpPressureCoeffTaskName  );
        ONEFLOW::AddCmdToList( kDumpHeatfluxCoeffTaskName  );
        if ( ctrl.idump == 1 )
        {
            ONEFLOW::AddCmdToList( kDumpLaminarPlateTaskName );
        }
        else if ( ctrl.idump == 2 )
        {
            ONEFLOW::AddCmdToList( kDumpTurbPlateTaskName );
        }
    }

    if ( Iteration::outerSteps % Iteration::nFieldSave == 0 )
    {
        ONEFLOW::AddCmdToList( kDumpRestartTaskName );
    }

}

void DumpHeatFluxCoeff( StringField & data )
{
    ;
}

void NsFinalPostprocess( StringField & data )
{
    ONEFLOW::AddCmdToList( kDumpRestartTaskName       );
    ONEFLOW::AddCmdToList( kDumpAerodynamicTaskName   );
    ONEFLOW::AddCmdToList( kDumpPressureCoeffTaskName );
    ONEFLOW::AddCmdToList( kDumpHeatfluxCoeffTaskName );
    ONEFLOW::AddCmdToList( kVisualizationTaskName     );
}

void NsInitSolver( StringField & data )
{
    ONEFLOW::CommInterfaceData();
}

SolverRegData nsReg;
SolverRegData * GetNsReg()
{
    nsReg.solverType = NS_SOLVER;
    nsReg.func = & RegisterNsFunc;
    nsReg.solverName = "ns";
    nsReg.baseKind = SOLVER_BASED;
    nsReg.dataFlag = WITH_DATA;
    return & nsReg;
}

REGISTER_REG_DATA( GetNsReg );

EndNameSpace
