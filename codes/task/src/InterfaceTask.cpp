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

#include "InterfaceTask.h"
#include "Parallel.h"
#include "Zone.h"
#include "ZoneState.h"
#include "GridState.h"
#include "PIO.h"
#include "ActionState.h"
#include "DataBook.h"
#include "InterFace.h"
#include "Grid.h"
#include "UnsGrid.h"
#include "FaceTopo.h"
#include "DataBase.h"
#include "SolverDef.h"
#include "FieldWrap.h"
#include "SimuContext.h"
#include "AccelRuntime.h"
#ifdef ONEFLOW_ENABLE_HIP
#include "HipFluxBackend.h"
#endif
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <vector>

BeginNameSpace( ONEFLOW )

namespace
{

bool HipInterfaceRequested()
{
    const char * enabled =
        std::getenv( "ONEFLOW_ENABLE_UNS_HIP_INTERFACE" );
    return enabled != nullptr && enabled[ 0 ] == '1';
}

bool GpuAwareMpiRequested()
{
    const char * enabled =
        std::getenv( "ONEFLOW_ENABLE_UNS_GPU_AWARE_MPI" );
    return enabled != nullptr && enabled[ 0 ] == '1';
}

bool GpuAwareMpiConfirmed()
{
    const char * confirmed =
        std::getenv( "ONEFLOW_GPU_AWARE_MPI_CONFIRMED" );
    return confirmed != nullptr && confirmed[ 0 ] == '1';
}

#ifdef ONEFLOW_ENABLE_HIP
bool BuildHipInterfaceEndpoint(
    UnsGrid * grid,
    int neighborZone,
    int sendRecv,
    BoundaryEndpointMap & endpoint )
{
    if ( grid == nullptr || grid->interFace == nullptr ) return false;
    InterFace * interFace = grid->interFace.get();
    int neighborIndex = -1;
    if ( ! interFace->TryGetNeighborIndex( neighborZone, neighborIndex ) )
        return false;

    const IntField & interfaceIds =
        interFace->GetInterfaceId( neighborIndex, sendRecv );
    endpoint.cellCount = grid->nCells + grid->nBFaces;
    endpoint.cellIds.resize( interfaceIds.size() );
    for ( HXSize_t iFace = 0; iFace < interfaceIds.size(); ++ iFace )
    {
        int cellId = -1;
        if ( sendRecv == GREAT_SEND )
            grid->GetFaceTopo().GetSId( interfaceIds[ iFace ], 1, cellId );
        else
            grid->GetFaceTopo().GetTId( interfaceIds[ iFace ], 1, cellId );
        endpoint.cellIds[ iFace ] = cellId;
    }
    endpoint.Validate();
    return true;
}

void ExchangeHipInterfaceQ(
    SimuContext & context,
    int sourceZone,
    int targetZone,
    int sourceRank,
    int targetRank )
{
    AccelRuntime & runtime = AccelRuntime::Instance();
    if ( ! runtime.IsInitialized()
         || runtime.Backend().Kind() != AccelBackendKind::HIP )
        return;

    const bool isSource = Parallel::pid == sourceRank;
    const bool isTarget = Parallel::pid == targetRank;
    if ( ! isSource && ! isTarget ) return;

    const int previousZone = ZoneState::zid;
    const int tagBase =
        700000 + sourceZone * ZoneState::nZones + targetZone;
    UnsGrid * sourceGrid = nullptr;
    UnsGrid * targetGrid = nullptr;
    BoundaryEndpointMap sourceEndpoint;
    BoundaryEndpointMap targetEndpoint;
    HipFluxBackend & backend = HipFluxBackend::Shared();
    const void * sourceKey = nullptr;
    const void * targetKey = nullptr;
    MRField * sourceQ = nullptr;
    MRField * targetQ = nullptr;
    int nEquations = 5;
    const Real * sourceComponents[ 5 ] = {};
    const Real * targetComponents[ 5 ] = {};
    const bool useDeviceDirect =
        sourceRank != targetRank
        && GpuAwareMpiRequested()
        && GpuAwareMpiConfirmed();

    if ( sourceRank != targetRank
         && GpuAwareMpiRequested()
         && ! GpuAwareMpiConfirmed() )
    {
        throw std::runtime_error(
            "GPU-aware MPI was requested without a confirmed device-buffer "
            "MPI probe; set ONEFLOW_GPU_AWARE_MPI_CONFIRMED=1 only after "
            "the target MPI has passed that probe" );
    }

    // Establish readiness symmetrically before any count or payload
    // communication.  A missing/incompatible local endpoint must not leave
    // the peer blocked in HXSwapData.
    int sourceReady = 0;
    int targetReady = 0;
    if ( isSource )
    {
        try
        {
            Grid * gridBase =
                Zone::GetGrid( sourceZone, GridState::gridLevel );
            sourceGrid = gridBase != nullptr && IsUnsGrid( gridBase->type )
                ? UnsGridCast( gridBase ) : nullptr;
            sourceQ = sourceGrid == nullptr
                ? nullptr : GetFieldPointer< MRField >( sourceGrid, "q" );
            sourceReady =
                sourceGrid != nullptr
                && sourceQ != nullptr
                && sourceQ->GetNEqu() == nEquations
                && BuildHipInterfaceEndpoint(
                    sourceGrid, targetZone, GREAT_SEND, sourceEndpoint );
        }
        catch ( const std::exception & )
        {
            sourceReady = 0;
        }
    }

    if ( isTarget )
    {
        try
        {
            Grid * gridBase =
                Zone::GetGrid( targetZone, GridState::gridLevel );
            targetGrid = gridBase != nullptr && IsUnsGrid( gridBase->type )
                ? UnsGridCast( gridBase ) : nullptr;
            targetQ = targetGrid == nullptr
                ? nullptr : GetFieldPointer< MRField >( targetGrid, "q" );
            targetReady =
                targetGrid != nullptr
                && targetQ != nullptr
                && targetQ->GetNEqu() == nEquations
                && BuildHipInterfaceEndpoint(
                    targetGrid, sourceZone, GREAT_RECV, targetEndpoint );
        }
        catch ( const std::exception & )
        {
            targetReady = 0;
        }
    }

    if ( ! ( isSource && isTarget ) )
    {
        HXSwapData(
            & sourceReady, 1, sourceRank, targetRank, tagBase );
        HXSwapData(
            & targetReady, 1, targetRank, sourceRank, tagBase + 1 );
    }
    if ( sourceReady == 0 || targetReady == 0 )
    {
        ZoneState::zid = previousZone;
        return;
    }

    if ( isSource )
    {
        sourceKey = sourceGrid;
        for ( int equation = 0; equation < nEquations; ++ equation )
            sourceComponents[ equation ] =
                ( *sourceQ )[ equation ].data();
    }
    if ( isTarget )
    {
        targetKey = targetGrid;
        for ( int equation = 0; equation < nEquations; ++ equation )
            targetComponents[ equation ] =
                ( *targetQ )[ equation ].data();
    }

    BoundaryBufferView send;
    BoundaryBufferView receive;
    if ( isSource )
    {
        ZoneState::zid = sourceZone;
        backend.UploadBoundaryState(
            sourceKey, sourceGrid->nCells, sourceGrid->nBFaces,
            nEquations, sourceComponents );
        send = backend.PackBoundary(
            sourceKey, sourceEndpoint, nEquations,
            useDeviceDirect
                ? BoundaryTransferMode::DeviceDirect
                : BoundaryTransferMode::HostStaged );
    }
    if ( isTarget )
    {
        ZoneState::zid = targetZone;
        backend.UploadBoundaryState(
            targetKey, targetGrid->nCells, targetGrid->nBFaces,
            nEquations, targetComponents );
        receive = backend.PrepareBoundaryReceive(
            targetKey, targetEndpoint, nEquations,
            useDeviceDirect
                ? BoundaryTransferMode::DeviceDirect
                : BoundaryTransferMode::HostStaged );
    }

    int sendCount = isSource
        ? static_cast< int >( send.valueCount ) : 0;
    int receiveCount = 0;
    HXSwapData( &sendCount, 1, sourceRank, targetRank, tagBase );
    if ( isTarget )
    {
        receiveCount = static_cast< int >( receive.valueCount );
        if ( sendCount != receiveCount )
            throw std::runtime_error(
                "HIP interface exchange value count mismatch" );
    }

    if ( isSource && isTarget )
    {
        // Same-rank exchanges remain host-staged.  Device pointers are not
        // valid inputs to std::copy, and the host path is deterministic for
        // the local multi-zone case.
        if ( send.valueCount != receive.valueCount )
        {
            throw std::runtime_error(
                "local HIP interface exchange value count mismatch" );
        }
        std::copy(
            send.values,
            send.values + send.valueCount,
            receive.values );
    }
    else
    {
        if ( useDeviceDirect )
        {
            Real * data = isSource
                ? send.values : receive.values;
            HXSwapDeviceData(
                data,
                static_cast< std::size_t >( sendCount ) * sizeof( Real ),
                sourceRank,
                targetRank,
                tagBase + 1 );
        }
        else
        {
            Real * data = isSource
                ? send.values : receive.values;
            HXSwapData(
                data, sendCount, sourceRank, targetRank, tagBase + 1 );
        }
    }

    if ( isTarget )
    {
        backend.UnpackBoundary(
            targetKey, targetEndpoint, nEquations,
            useDeviceDirect
                ? BoundaryTransferMode::DeviceDirect
                : BoundaryTransferMode::HostStaged );
    }
    ZoneState::zid = previousZone;
    (void) context;
}
#endif

}


CUpdateInterface::CUpdateInterface()
{
    ;
}

CUpdateInterface::~CUpdateInterface()
{
    ;
}

void CUpdateInterface::Run()
{
    ActionState::dataBook = this->dataBook.get();

    std::vector< int > ownerRanks(
        ZoneState::pid.begin(),
        ZoneState::pid.end() );
    std::vector< std::vector< int > > zoneNeighbors(
        ZoneState::nZones );
    for ( int zId = 0; zId < ZoneState::nZones; ++ zId )
    {
        int nNei = Zone::GetNumberOfZoneNeighbors( zId );
        for ( int iNei = 0; iNei < nNei; ++ iNei )
        {
            zoneNeighbors[ zId ].push_back(
                Zone::GetNeighborZoneId( zId, iNei ) );
        }
    }

    const DistributedDomainPlan plan = DistributedDomainPlan::Build(
        ownerRanks,
        zoneNeighbors,
        Parallel::GetNProc() );
    for ( const BoundaryExchange & exchange : plan.BoundaryExchanges() )
    {
        this->SwapInterfaceData( plan, exchange );
    }
}

void CUpdateInterface::SwapInterfaceData(
    const DistributedDomainPlan & plan,
    const BoundaryExchange & exchange )
{
    const int iZone = exchange.sourceZone;
    const int jZone = exchange.targetZone;
    const int sPid = plan.OwnerRank( iZone );
    const int rPid = plan.OwnerRank( jZone );

    if ( Parallel::pid == sPid )
    {
        ZoneState::zid  = iZone;
        ZoneState::rzid = jZone;

        this->sendAction();
    }

    HXSwapData( ActionState::dataBook, sPid, rPid );

    if ( Parallel::pid == rPid )
    {
        ZoneState::zid  = jZone;
        ZoneState::szid = iZone;

        this->recvAction();
    }

#ifdef ONEFLOW_ENABLE_HIP
    if ( this->Context() != nullptr && HipInterfaceRequested() )
    {
        ExchangeHipInterfaceQ(
            *this->Context(), iZone, jZone, sPid, rPid );
    }
#endif
}

EndNameSpace
