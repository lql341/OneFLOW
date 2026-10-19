#include <gtest/gtest.h>

#include "FaceTopo.h"
#include "InterfaceEndpoint.h"
#include "InterFace.h"
#include "SolverDef.h"
#include "UnsGrid.h"

#include <memory>
#include <stdexcept>

namespace
{

void InitializeInterfaceGrid( ONEFLOW::UnsGrid & grid )
{
    grid.Init();
    grid.nCells = 2;
    grid.nFaces = 2;
    grid.nBFaces = 2;

    auto & faceTopo = grid.GetFaceTopo();
    faceTopo.SetNBFaces( 2 );
    faceTopo.GetLeftCells() = { 0, 1 };
    faceTopo.GetRightCells() = { 2, 3 };

    grid.interFace->Set( 2, &grid );
    grid.interFace->i2b = { 1, 0 };
    grid.interFace->interFacePairs.resize( 1 );
    grid.interFace->interFacePairs[ 0 ] =
        std::make_unique< ONEFLOW::InterfacePair >();
    grid.interFace->interFacePairs[ 0 ]->nzid = 7;
    grid.interFace->z2n[ 7 ] = 0;
    grid.interFace->interFacePairs[ 0 ]->idsend = { 0, 1 };
    grid.interFace->interFacePairs[ 0 ]->idrecv = { 1, 0 };
}

TEST( InterfaceEndpointTest, ResolvesOwnedSendAndGhostReceiveCells )
{
    ONEFLOW::UnsGrid grid;
    InitializeInterfaceGrid( grid );

    ONEFLOW::BoundaryEndpointMap send;
    ASSERT_TRUE(
        ONEFLOW::BuildInterfaceEndpoint(
            &grid,
            7,
            ONEFLOW::GREAT_SEND,
            ONEFLOW::BoundaryEndpointRole::OwnedCell,
            send ) );
    EXPECT_EQ( send.cellCount, 4 );
    EXPECT_EQ( send.ownedCellCount, 2 );
    EXPECT_EQ( send.cellIds, ( ONEFLOW::IntField{ 1, 0 } ) );

    ONEFLOW::BoundaryEndpointMap receive;
    ASSERT_TRUE(
        ONEFLOW::BuildInterfaceEndpoint(
            &grid,
            7,
            ONEFLOW::GREAT_RECV,
            ONEFLOW::BoundaryEndpointRole::GhostCell,
            receive ) );
    EXPECT_EQ( receive.cellIds, ( ONEFLOW::IntField{ 2, 3 } ) );
}

TEST( InterfaceEndpointTest, RejectsMissingNeighborWithoutMutatingEndpoint )
{
    ONEFLOW::UnsGrid grid;
    InitializeInterfaceGrid( grid );
    ONEFLOW::BoundaryEndpointMap endpoint;
    endpoint.cellCount = 9;
    endpoint.cellIds = { 8 };

    EXPECT_FALSE(
        ONEFLOW::BuildInterfaceEndpoint(
            &grid,
            99,
            ONEFLOW::GREAT_SEND,
            ONEFLOW::BoundaryEndpointRole::OwnedCell,
            endpoint ) );
    EXPECT_EQ( endpoint.cellCount, 9 );
    EXPECT_EQ( endpoint.cellIds, ( ONEFLOW::IntField{ 8 } ) );
}

} // namespace
