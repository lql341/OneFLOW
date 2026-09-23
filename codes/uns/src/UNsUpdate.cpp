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

#include "UNsUpdate.h"
#include "UCom.h"
#include "NsCom.h"
#include "UNsCom.h"
#include "NsIdx.h"
#include "Zone.h"
#include "ZoneState.h"
#include "HXMath.h"
#include "Iteration.h"
#include "GridState.h"
#include "SolverDef.h"
#include "AccelViews.h"
#include "AccelRuntime.h"
#ifdef ONEFLOW_ENABLE_HIP
#include "HipFluxBackend.h"
#endif
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdexcept>


BeginNameSpace( ONEFLOW )

namespace
{

void AppendStateStageTrace(
    const char * path,
    std::uint32_t sequence,
    std::uint64_t nItems,
    std::uint32_t nEquations,
    const MRField & field )
{
    std::ofstream output( path, std::ios::binary | std::ios::app );
    if ( ! output )
    {
        throw std::runtime_error( "cannot open UNs stage trace file" );
    }

    const char magic[ 8 ] = "OFSTG01";
    const std::uint32_t kind = 3;
    const std::int32_t outerStep = Iteration::outerSteps;
    const std::int32_t gridLevel = GridState::gridLevel;
    const std::uint32_t nArrays = 1;
    output.write( magic, sizeof( magic ) );
    output.write(
        reinterpret_cast< const char * >( & kind ), sizeof( kind ) );
    output.write(
        reinterpret_cast< const char * >( & sequence ), sizeof( sequence ) );
    output.write(
        reinterpret_cast< const char * >( & outerStep ), sizeof( outerStep ) );
    output.write(
        reinterpret_cast< const char * >( & gridLevel ), sizeof( gridLevel ) );
    output.write(
        reinterpret_cast< const char * >( & nEquations ),
        sizeof( nEquations ) );
    output.write(
        reinterpret_cast< const char * >( & nArrays ), sizeof( nArrays ) );
    output.write(
        reinterpret_cast< const char * >( & nItems ), sizeof( nItems ) );

    if ( field.GetNEqu() < nEquations )
    {
        throw std::runtime_error(
            "UNs state trace field has an invalid equation extent" );
    }
    for ( std::uint32_t equation = 0;
          equation < nEquations; ++ equation )
    {
        const auto & values = field[ equation ];
        if ( values.size() < nItems )
        {
            throw std::runtime_error(
                "UNs state trace field has an invalid cell extent" );
        }
        output.write(
            reinterpret_cast< const char * >( values.data() ),
            static_cast< std::streamsize >( nItems * sizeof( Real ) ) );
    }
    if ( ! output )
    {
        throw std::runtime_error( "failed while writing UNs state trace" );
    }
}

}

UNsUpdate::UNsUpdate()
{
}

UNsUpdate::~UNsUpdate()
{
}

namespace
{

bool UseHipDeviceStateUpdate( int solverType )
{
    const char * enabled =
        std::getenv( "ONEFLOW_ENABLE_UNS_HIP_STATE_UPDATE" );
    if ( enabled == nullptr || enabled[ 0 ] != '1' ) return false;
    if ( solverType != NS_SOLVER )
    {
        throw std::runtime_error(
            "HIP state update is only supported for the NS solver" );
    }
    const char * reconstruction =
        std::getenv( "ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION" );
    if ( reconstruction == nullptr || reconstruction[ 0 ] != '1' )
    {
        throw std::runtime_error(
            "HIP state update requires HIP device reconstruction" );
    }
    if ( nscom.chemModel != 0 || nscom.nTModel != 1 )
    {
        throw std::runtime_error(
            "HIP state update requires a single ideal-gas temperature field" );
    }
#ifndef ONEFLOW_ENABLE_HIP
    throw std::runtime_error(
        "HIP state update was requested without HIP support" );
#else
    return true;
#endif
}

}

void UNsUpdate::UpdateFlowField( int solverType )
{
    if ( UseHipDeviceStateUpdate( solverType ) )
    {
#ifdef ONEFLOW_ENABLE_HIP
        UnsGrid * grid = Zone::GetUnsGrid();
        ug.Init();
        unsf.Init();

        CellStateUpdateView view;
        view.nCells = ug.nCells;
        view.nGhostCells = ug.nTCell - ug.nCells;
        view.nEquations = nscom.nEqu;
        view.timeStep = ( * unsf.timestep )[ 0 ].data();
        view.cellVolume = ug.cvol->data();
        view.gamma = nscom.gama_ref;
        view.rkCoefficient = 1.0;
        view.cacheKey = grid;
        for ( int equation = 0; equation < view.nEquations; ++ equation )
        {
            view.primitive[ equation ] = ( * unsf.q )[ equation ].data();
        }
        HipFluxBackend::Shared().UpdatePrimitiveState( view, true );
        for ( int cId = 0; cId < ug.nCells; ++ cId )
        {
            const Real density = ( * unsf.q )[ IDX::IR ][ cId ];
            const Real pressure = ( * unsf.q )[ IDX::IP ][ cId ];
            ( * unsf.tempr )[ 0 ][ cId ] =
                pressure / ( nscom.statecoef * density );
        }
        this->DumpUpdatedStateTrace();
        return;
#endif
    }

    GetUpdateField( solverType, this->q, this->dq );

    ug.Init();
    unsf.Init();

    for ( int cId = 0; cId < ug.nCells; ++ cId )
    {
        ug.cId = cId;

        this->PrepareData();

        this->CalcFlowField();

        this->UpdateFlowFieldValue();
    }

    this->DumpUpdatedStateTrace();
}

void UNsUpdate::DumpUpdatedStateTrace()
{
    const char * traceFile = std::getenv( "ONEFLOW_UNS_STAGE_TRACE_FILE" );
    if ( traceFile == nullptr || traceFile[ 0 ] == 0 ) return;
    if ( unsf.q == nullptr )
    {
        throw std::runtime_error(
            "UNs state trace requested before flow state is available" );
    }

    static std::uint32_t sequence = 0;
    AppendStateStageTrace(
        traceFile, sequence,
        static_cast< std::uint64_t >( ug.nCells ),
        static_cast< std::uint32_t >( nscom.nTEqu ), * unsf.q );
    ++ sequence;
}

void UNsUpdate::PrepareData()
{
    for ( int iEqu = 0; iEqu < nscom.nTEqu; ++ iEqu )
    {
        nscom.prim [ iEqu ] = ( * unsf.q )[ iEqu ][ ug.cId ];
        nscom.prim0[ iEqu ] = ( * unsf.q )[ iEqu ][ ug.cId ];
    }

    for ( int iEqu = 0; iEqu < nscom.nTModel; ++ iEqu )
    {
        nscom.t [ iEqu ] = ( * unsf.tempr )[ iEqu ][ ug.cId ];
        nscom.t0[ iEqu ] = ( * unsf.tempr )[ iEqu ][ ug.cId ];
    }

    for ( int iEqu = 0; iEqu < nscom.nTEqu; ++ iEqu )
    {
        nscom.dq[ iEqu ] = ( * unsf.dq )[ iEqu ][ ug.cId ];
    }

    nscom.gama = ( * unsf.gama )[ 0 ][ ug.cId ];
}

void UNsUpdate::DumpProbeInfo()
{
    std::cout << std::setprecision( 17 );
    std::cout << "Warning : non-physical state"
              << " step = " << Iteration::outerSteps
              << ", zid = " << ZoneState::zid
              << ", cid = " << ug.cId
              << ", density = " << nscom.prim[ IDX::IR ]
              << ", pressure = " << nscom.prim[ IDX::IP ]
              << ", density0 = " << nscom.prim0[ IDX::IR ]
              << ", pressure0 = " << nscom.prim0[ IDX::IP ]
              << ", dq_density = " << nscom.dq[ IDX::IR ]
              << ", dq_pressure = " << nscom.dq[ IDX::IP ]
              << ", timestep = " << nscom.timestep
              << ", dt = " << nscom.dt
              << std::endl;
}

void UNsUpdate::SolutionFix()
{
    nscom.prim = 0;

    Real sumV = 0.0;

    int fn = ( * ug.c2f )[ ug.cId ].size();

    for ( int iFace = 0; iFace < fn; ++ iFace )
    {
        int fId = ( * ug.c2f )[ ug.cId ][ iFace ];

        ug.lc = ( * ug.lcf )[ fId ];
        ug.rc = ( * ug.rcf )[ fId ];

        int iNei = ug.lc;
        if ( ug.cId == ug.lc  ) iNei = ug.rc;

        Real volN = one;

        sumV += volN;

        for ( int iEqu = 0; iEqu < nscom.nTEqu; ++ iEqu )
        {
            Real f = ( * unsf.q )[ iEqu ][ iNei ];
            if ( iEqu == IDX::IR || iEqu == IDX::IP )
            {
                f = ABS( f );
            }
            nscom.prim[ iEqu ] += f * volN;
        }
    }

    Real rVol = 1.0 / sumV;

    for ( int iEqu = 0; iEqu < nscom.nTEqu; ++ iEqu )
    {
        nscom.prim[ iEqu ] *= rVol;
    }
}

void UNsUpdate::UpdateFlowFieldValue()
{
    for ( int iEqu = 0; iEqu < nscom.nTEqu; ++ iEqu )
    {
        ( * unsf.q )[ iEqu ][ ug.cId ] = nscom.prim[ iEqu ];
    }

    for ( int iEqu = 0; iEqu < nscom.nTModel; ++ iEqu )
    {
        ( * unsf.tempr )[ iEqu ][ ug.cId ] = nscom.t[ iEqu ];
    }
}

EndNameSpace
