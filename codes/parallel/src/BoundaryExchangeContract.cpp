/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.
\*---------------------------------------------------------------------------*/

#include "BoundaryExchangeContract.h"

#include <stdexcept>

namespace ONEFLOW
{

void BoundaryEndpointMap::Validate() const
{
    if ( cellCount < 0 )
    {
        throw std::invalid_argument(
            "boundary endpoint cell count must be positive" );
    }
    for ( const int cellId : cellIds )
    {
        if ( cellId < 0 || cellId >= cellCount )
        {
            throw std::out_of_range(
                "boundary endpoint cell is out of range" );
        }
    }
}

void BoundaryFieldExchangePlan::Validate() const
{
    if ( sourceZone < 0 || targetZone < 0
         || sourceRank < 0 || targetRank < 0 )
    {
        throw std::invalid_argument(
            "boundary exchange zone and rank ids must be non-negative" );
    }
    if ( ghostLevel < 0 || nEquations <= 0
         || sourceCellCount <= 0 || targetCellCount <= 0 )
    {
        throw std::invalid_argument(
            "boundary exchange dimensions must be positive" );
    }
    if ( sendCellIds.size() != receiveCellIds.size() )
    {
        throw std::invalid_argument(
            "boundary exchange send and receive maps differ in size" );
    }
    BoundaryEndpointMap{ sourceCellCount, sendCellIds }.Validate();
    BoundaryEndpointMap{ targetCellCount, receiveCellIds }.Validate();
}

std::size_t BoundaryFieldExchangePlan::InterfaceCount() const
{
    return sendCellIds.size();
}

std::size_t BoundaryFieldExchangePlan::ValueCount() const
{
    return InterfaceCount() * static_cast< std::size_t >( nEquations );
}

void PackBoundaryHost(
    const BoundaryEndpointMap & endpoint,
    int nEquations,
    const std::vector< const Real * > & sourceComponents,
    std::vector< Real > & packedValues )
{
    endpoint.Validate();
    if ( nEquations <= 0 )
    {
        throw std::invalid_argument(
            "boundary exchange equation count must be positive" );
    }
    if ( sourceComponents.size()
         != static_cast< std::size_t >( nEquations ) )
    {
        throw std::invalid_argument(
            "boundary exchange source component count is inconsistent" );
    }
    for ( const Real * component : sourceComponents )
    {
        if ( component == nullptr )
        {
            throw std::invalid_argument(
                "boundary exchange source component is null" );
        }
    }

    const std::size_t interfaceCount = endpoint.cellIds.size();
    packedValues.resize(
        interfaceCount * static_cast< std::size_t >( nEquations ) );
    for ( int equation = 0; equation < nEquations; ++ equation )
    {
        for ( std::size_t interfaceId = 0;
            interfaceId < interfaceCount; ++ interfaceId )
        {
            packedValues[
                static_cast< std::size_t >( equation ) * interfaceCount
                + interfaceId ] =
                sourceComponents[ equation ][
                    endpoint.cellIds[ interfaceId ] ];
        }
    }
}

void UnpackBoundaryHost(
    const BoundaryEndpointMap & endpoint,
    int nEquations,
    const std::vector< Real > & packedValues,
    const std::vector< Real * > & targetComponents )
{
    endpoint.Validate();
    if ( nEquations <= 0 )
    {
        throw std::invalid_argument(
            "boundary exchange equation count must be positive" );
    }
    const std::size_t interfaceCount = endpoint.cellIds.size();
    if ( packedValues.size()
         != interfaceCount * static_cast< std::size_t >( nEquations ) )
    {
        throw std::invalid_argument(
            "boundary exchange packed value count is inconsistent" );
    }
    if ( targetComponents.size()
         != static_cast< std::size_t >( nEquations ) )
    {
        throw std::invalid_argument(
            "boundary exchange target component count is inconsistent" );
    }
    for ( Real * component : targetComponents )
    {
        if ( component == nullptr )
        {
            throw std::invalid_argument(
                "boundary exchange target component is null" );
        }
    }

    for ( int equation = 0; equation < nEquations; ++ equation )
    {
        for ( std::size_t interfaceId = 0;
            interfaceId < interfaceCount; ++ interfaceId )
        {
            targetComponents[ equation ][
                endpoint.cellIds[ interfaceId ] ] =
                packedValues[
                    static_cast< std::size_t >( equation ) * interfaceCount
                    + interfaceId ];
        }
    }
}

void PackBoundaryHost(
    const BoundaryFieldExchangePlan & plan,
    const std::vector< const Real * > & sourceComponents,
    std::vector< Real > & packedValues )
{
    plan.Validate();
    PackBoundaryHost(
        { plan.sourceCellCount, plan.sendCellIds },
        plan.nEquations,
        sourceComponents,
        packedValues );
}

void UnpackBoundaryHost(
    const BoundaryFieldExchangePlan & plan,
    const std::vector< Real > & packedValues,
    const std::vector< Real * > & targetComponents )
{
    plan.Validate();
    UnpackBoundaryHost(
        { plan.targetCellCount, plan.receiveCellIds },
        plan.nEquations,
        packedValues,
        targetComponents );
}

}
