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
#include "FieldSimu.h"
#include "CpuEulerDomainBackend.h"
#include "EulerDomainMrFieldAdapter.h"
#include "SimuContext.h"
#include "Iteration.h"
#include "Ctrl.h"
#include "NsCom.h"
#include "UsdData.h"
#include "MultiBlock.h"
#include "SolverMap.h"
#include "CmxTask.h"
#include "Multigrid.h"
#include "BcData.h"
#include "DataBase.h"
#include "FieldImp.h"
#include "Grid.h"
#include "GridState.h"
#include "SolverState.h"
#include "Zone.h"
#include "ZoneState.h"
#include <cmath>
#include <iostream>
#include <stdexcept>


BeginNameSpace( ONEFLOW )

namespace
{

void SyncCurrentEulerDomainState(
    SimuContext & context,
    CpuEulerDomainBackend & backend,
    bool restart )
{
    Grid * grid = Zone::GetGrid();
    if ( grid == nullptr || grid->nCells <= 0 ) return;

    MRField * q = GetFieldPointer< MRField >( grid, "q" );
    if ( q == nullptr ) return;

    const int nEquations = static_cast< int >( q->GetNEqu() );
    if ( nEquations != 3 && nEquations != 5 ) return;

    Real minDistance = 0.0;
    Real maxDistance = 0.0;
    grid->GetMinMaxDistance( minDistance, maxDistance );
    if ( ! std::isfinite( minDistance ) || minDistance <= 0.0 )
    {
        throw std::runtime_error(
            "cannot derive a positive Euler domain length scale" );
    }
    if ( ! std::isfinite( ctrl.pdt ) || ctrl.pdt <= 0.0 )
    {
        throw std::runtime_error(
            "cannot initialize Euler domain state with invalid dt" );
    }
    if ( ! std::isfinite( nscom.gama_ref ) || nscom.gama_ref <= 1.0 )
    {
        throw std::runtime_error(
            "cannot initialize Euler domain state with invalid gamma" );
    }

    EulerDomainProblem problem;
    problem.nCells = grid->nCells;
    problem.nGhostCells = grid->nBFaces;
    problem.nEquations = nEquations;
    problem.gamma = nscom.gama_ref;
    problem.dt = ctrl.pdt;
    problem.dx = minDistance;

    EulerDomainStateKey key{
        SolverState::solverIndex,
        ZoneState::zid,
        GridState::gridLevel,
        AccelBackendKind::CPU };
    EulerDomainMrFieldSnapshot snapshot( *q, grid->nCells );
    const EulerDomainConstFieldView field = snapshot.View();

    if ( restart )
    {
        context.RestartAccelState( backend, problem, key, field );
    }
    else
    {
        context.InitializeAccelState( backend, problem, key, field );
    }
}

void SyncAllEulerDomainStates( SimuContext & context )
{
    CpuEulerDomainBackend backend;
    const bool restart = ctrl.startStrategy == 1 || ctrl.startStrategy == 3;
    for ( int solverIndex = 0;
        solverIndex < SolverState::nSolver; ++ solverIndex )
    {
        SolverState::SetSolverTypeBySolverIndex( solverIndex );
        for ( int gridLevel = 0;
            gridLevel < GridState::nGrids; ++ gridLevel )
        {
            GridState::SetGridLevel( gridLevel );
            SyncCurrentEulerDomainState( context, backend, restart );
        }
    }
}

}

void FieldSimu( SimuContext & context )
{
    InitFlowSimuGlobal();
    MultiBlock::LoadGridAndBuildLink();
    MultiBlock::ProcessFlowWallDist();
    SolverMap::CreateSolvers();
    InitializeSolver();
    SyncAllEulerDomainStates( context );
    MultigridSolve();
}

void InitFlowSimuGlobal()
{
    vis_model.Init();
    ctrl.Init();
    Iteration::Init();
    usd.InitBasic();
}

void InitializeSolver()
{
    ONEFLOW::MultiSolverMultiGridTask( "INIT_FLOWFIELD" );
}

EndNameSpace
