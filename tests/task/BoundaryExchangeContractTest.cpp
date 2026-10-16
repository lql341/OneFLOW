#include <gtest/gtest.h>

#include "BoundaryExchangeContract.h"

#include <stdexcept>
#include <vector>

namespace
{

using ONEFLOW::BoundaryFieldExchangePlan;
using ONEFLOW::PackBoundaryHost;
using ONEFLOW::Real;
using ONEFLOW::UnpackBoundaryHost;

BoundaryFieldExchangePlan MakePlan()
{
    BoundaryFieldExchangePlan plan;
    plan.sourceZone = 2;
    plan.targetZone = 7;
    plan.sourceRank = 0;
    plan.targetRank = 1;
    plan.ghostLevel = 1;
    plan.nEquations = 2;
    plan.sourceCellCount = 5;
    plan.targetCellCount = 6;
    plan.sendCellIds = { 4, 1 };
    plan.receiveCellIds = { 3, 5 };
    return plan;
}

TEST( BoundaryExchangeContractTest, PacksAndUnpacksEquationMajorValues )
{
    const BoundaryFieldExchangePlan plan = MakePlan();
    const std::vector< Real > source0 = { 10, 11, 12, 13, 14 };
    const std::vector< Real > source1 = { 20, 21, 22, 23, 24 };
    std::vector< Real > packed;

    PackBoundaryHost(
        plan,
        { source0.data(), source1.data() },
        packed );

    EXPECT_EQ(
        packed,
        ( std::vector< Real >{ 14, 11, 24, 21 } ) );

    std::vector< Real > target0( 6, -1 );
    std::vector< Real > target1( 6, -1 );
    UnpackBoundaryHost(
        plan,
        packed,
        { target0.data(), target1.data() } );

    EXPECT_EQ( target0[ 3 ], 14 );
    EXPECT_EQ( target0[ 5 ], 11 );
    EXPECT_EQ( target1[ 3 ], 24 );
    EXPECT_EQ( target1[ 5 ], 21 );
    EXPECT_EQ( target0[ 0 ], -1 );
    EXPECT_EQ( target1[ 4 ], -1 );
}

TEST( BoundaryExchangeContractTest, SupportsZeroLengthExchange )
{
    BoundaryFieldExchangePlan plan = MakePlan();
    plan.sendCellIds.clear();
    plan.receiveCellIds.clear();
    std::vector< Real > packed = { 1 };
    const std::vector< Real > source0( 5, 0 );
    const std::vector< Real > source1( 5, 0 );

    PackBoundaryHost(
        plan,
        { source0.data(), source1.data() },
        packed );

    EXPECT_TRUE( packed.empty() );
    std::vector< Real > target0( 6, -1 );
    std::vector< Real > target1( 6, -1 );
    EXPECT_NO_THROW(
        UnpackBoundaryHost(
            plan,
            packed,
            { target0.data(), target1.data() } ) );
}

TEST( BoundaryExchangeContractTest, RejectsInconsistentMappings )
{
    BoundaryFieldExchangePlan plan = MakePlan();
    plan.receiveCellIds.pop_back();
    EXPECT_THROW( plan.Validate(), std::invalid_argument );

    plan = MakePlan();
    plan.sendCellIds[ 0 ] = plan.sourceCellCount;
    EXPECT_THROW( plan.Validate(), std::out_of_range );

    plan = MakePlan();
    plan.receiveCellIds[ 0 ] = -1;
    EXPECT_THROW( plan.Validate(), std::out_of_range );
}

}
