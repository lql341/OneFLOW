#include <gtest/gtest.h>
#include <cmath>

#include "NsBcPolicy.h"
#include "BoundaryArithmetic.h"
#include "HXMath.h"

using namespace ONEFLOW;

TEST( BcPolicy, ReflectVelocityCancelsNormalComponent )
{
    Real ur, vr, wr;
    ReflectVelocityRelativeToMovingFace( 3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, ur, vr, wr );
    EXPECT_DOUBLE_EQ( ur, -3.0 );
    EXPECT_DOUBLE_EQ( vr, 4.0 );
    EXPECT_DOUBLE_EQ( wr, 0.0 );
}

TEST( BcPolicy, ImageVelocityAboutStationaryWallNegates )
{
    Real ui, vi, wi;
    ImageVelocityAboutWall( 2.0, -1.0, 0.5, 0.0, 0.0, 0.0, ui, vi, wi );
    EXPECT_DOUBLE_EQ( ui, -2.0 );
    EXPECT_DOUBLE_EQ( vi, 1.0 );
    EXPECT_DOUBLE_EQ( wi, -0.5 );
}

TEST( BcPolicy, ImageVelocityAboutMovingWall )
{
    Real ui, vi, wi;
    ImageVelocityAboutWall( 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, ui, vi, wi );
    EXPECT_DOUBLE_EQ( ui, 5.0 );
}

TEST( BcPolicy, ArithmeticAverageIsMidpoint )
{
    EXPECT_DOUBLE_EQ( ArithmeticAverage( 1.0, 3.0 ), 2.0 );
}

TEST( BcPolicy, SubsonicAcousticRiemannCombineMatchesInvariants )
{
    Real vnb, cb;
    SubsonicAcousticRiemannCombine( 0.0, 1.0, 0.0, 1.0, 0.4, vnb, cb );
    EXPECT_NEAR( vnb, 0.0, 1.0e-12 );
    EXPECT_NEAR( cb, 1.0, 1.0e-12 );
}

TEST( BcPolicy, TangentialAndReconstructAreInverse )
{
    Real vtx, vty, vtz;
    TangentialVelocityFromNormalProjection( 3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 3.0, vtx, vty, vtz );
    EXPECT_DOUBLE_EQ( vtx, 0.0 );
    EXPECT_DOUBLE_EQ( vty, 4.0 );
    Real u, v, w;
    ReconstructVelocityFromTangentialAndNormal( vtx, vty, vtz, 1.0, 0.0, 0.0, 3.0, u, v, w );
    EXPECT_DOUBLE_EQ( u, 3.0 );
    EXPECT_DOUBLE_EQ( v, 4.0 );
}

TEST( BcPolicy, DensityAndPressureFromIsentropicSound )
{
    const Real gama = 1.4;
    const Real c = std::sqrt( gama );
    const Real rho = DensityFromIsentropicSoundSpeed( c, 1.0, gama, gama - 1.0 );
    EXPECT_NEAR( rho, 1.0, 1.0e-12 );
    EXPECT_NEAR( PressureFromSoundSpeedDensity( c, rho, gama ), 1.0, 1.0e-12 );
}

TEST( BcPolicy, IdealGasDensityFromPressureTemperature )
{
    EXPECT_DOUBLE_EQ(
        IdealGasDensityFromPressureTemperature( 2.0, 4.0, 0.5, 1.0 ), 1.0 );
}

TEST( BcPolicy, MirrorScalarWithFloorClamps )
{
    EXPECT_DOUBLE_EQ( MirrorScalarWithFloor( 10.0, 1.0, 0.5 ), 19.0 );
    EXPECT_DOUBLE_EQ( MirrorScalarWithFloor( 1.0, 10.0, 0.5 ), 0.5 );
}

TEST( BcPolicy, MirrorDensityAboutWallFloorsAtHalfWall )
{
    EXPECT_DOUBLE_EQ( MirrorDensityAboutWall( 2.0, 0.1 ), 3.9 );
    EXPECT_DOUBLE_EQ( MirrorDensityAboutWall( 2.0, 10.0 ), 1.0 );
}

// Symmetry ghost contract (B.4.1).
//
// The host writes the reflected velocity into the symmetry ghost:
// NsBcSolver::SymmetryBc copies prims1 into primt1 and then overwrites the
// three velocity components, and UNsBcSolver::UpdateBc stores primt1 in
// q[rc].  With ilim=0 that ghost feeds 0.5*(q[lc]+q[rc]), so a ghost left equal
// to the owner keeps the normal velocity on a symmetry face and shifts the
// inviscid flux.  The HIP kernel used to do exactly that; these tests pin the
// shared helper the kernel now calls, and the face average it must produce.
TEST( BcPolicy, SymmetryGhostCancelsNormalVelocity )
{
    // Face normal along x, stationary face, owner moving at 3 in x and 4 in y.
    Real ur, vr, wr;
    ReflectVelocityRelativeToMovingFaceHd(
        3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, ur, vr, wr );
    EXPECT_DOUBLE_EQ( ur, -3.0 );
    EXPECT_DOUBLE_EQ( vr, 4.0 );
    EXPECT_DOUBLE_EQ( wr, 0.0 );

    // The shared helper must agree with the host entry point bit for bit: the
    // device kernel and the host oracle now call the same expressions.
    Real hu, hv, hw;
    ReflectVelocityRelativeToMovingFace(
        3.0, 4.0, 0.0, 1.0, 0.0, 0.0, 0.0, hu, hv, hw );
    EXPECT_DOUBLE_EQ( ur, hu );
    EXPECT_DOUBLE_EQ( vr, hv );
    EXPECT_DOUBLE_EQ( wr, hw );
}

TEST( BcPolicy, SymmetryGhostDropsNormalPartOfFaceAverage )
{
    // What the solver actually consumes: 0.5*(owner + reflected ghost).  The
    // normal component must vanish; the tangential one must survive.  This is
    // the invariant the HIP "ghost = owner" branch violated.
    const Real nx = 1.0, ny = 0.0, nz = 0.0, vfn = 0.0;
    const Real u = 3.0, v = 4.0, w = 0.0;
    Real gu, gv, gw;
    ReflectVelocityRelativeToMovingFaceHd(
        u, v, w, nx, ny, nz, vfn, gu, gv, gw );

    const Real au = ArithmeticAverageHd( u, gu );
    const Real av = ArithmeticAverageHd( v, gv );
    const Real aw = ArithmeticAverageHd( w, gw );

    EXPECT_DOUBLE_EQ( au, 0.0 );
    EXPECT_DOUBLE_EQ( av, 4.0 );
    EXPECT_DOUBLE_EQ( aw, 0.0 );

    // A ghost left equal to the owner would keep au = 3.0 here.
    const Real wrongAu = ArithmeticAverageHd( u, u );
    EXPECT_DOUBLE_EQ( wrongAu, 3.0 );
    EXPECT_NE( au, wrongAu );
}

TEST( BcPolicy, SymmetryReflectionHandlesTiltedNormal )
{
    // Normalised diagonal normal: the reflected velocity's normal component
    // must be cancelled while the magnitude is preserved.
    const Real s = 1.0 / std::sqrt( 2.0 );
    const Real nx = s, ny = s, nz = 0.0;
    const Real u = 1.0, v = 0.0, w = 0.5;
    Real gu, gv, gw;
    ReflectVelocityRelativeToMovingFaceHd(
        u, v, w, nx, ny, nz, 0.0, gu, gv, gw );

    const Real vnOwner = nx * u + ny * v + nz * w;
    const Real vnGhost = nx * gu + ny * gv + nz * gw;
    EXPECT_NEAR( vnOwner + vnGhost, 0.0, 1.0e-15 );

    const Real magOwner = u * u + v * v + w * w;
    const Real magGhost = gu * gu + gv * gv + gw * gw;
    EXPECT_NEAR( magOwner, magGhost, 1.0e-15 );

    // w is tangential to this face and must be untouched.
    EXPECT_DOUBLE_EQ( gw, 0.5 );
}
