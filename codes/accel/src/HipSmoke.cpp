/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.
\*---------------------------------------------------------------------------*/

#include "AccelRuntime.h"
#include "CpuFluxBackend.h"
#include "EulerCpuAdapter.h"
#include "HipFluxBackend.h"
#include "HipKernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <vector>

namespace
{

double MaxDiff( const std::vector< ONEFLOW::Real > & a,
                const std::vector< ONEFLOW::Real > & b )
{
    double maxVal = 0.0;
    for ( std::size_t i = 0; i < a.size(); ++ i )
    {
        maxVal = std::max( maxVal,
            static_cast< double >( std::abs( a[ i ] - b[ i ] ) ) );
    }
    return maxVal;
}

// --- Scalar convection test (existing) ---

bool TestScalarConvection()
{
    constexpr int nFaces = 513;
    constexpr int nCells = 1026;
    constexpr int nBoundaryFaces = 17;
    std::vector< ONEFLOW::Real > qLeft( nFaces );
    std::vector< ONEFLOW::Real > qRight( nFaces );
    std::vector< ONEFLOW::Real > xNormal( nFaces );
    std::vector< ONEFLOW::Real > area( nFaces );
    std::vector< int > leftCell( nFaces );
    std::vector< int > rightCell( nFaces );
    for ( int face = 0; face < nFaces; ++ face )
    {
        qLeft[ face ] = 0.25 + 0.01 * face;
        qRight[ face ] = 0.75 - 0.003 * face;
        xNormal[ face ] = ( face % 3 == 0 ) ? -1.0 : 1.0;
        area[ face ] = 0.5 + 0.002 * ( face % 11 );
        leftCell[ face ] = face;
        rightCell[ face ] = nFaces + face;
    }

    ONEFLOW::FaceStateView state;
    state.nFaces = nFaces;
    state.nEquations = 1;
    state.qLeft = qLeft.data();
    state.qRight = qRight.data();
    state.xNormal = xNormal.data();
    state.faceArea = area.data();

    std::vector< ONEFLOW::Real > cpuFlux( nFaces );
    std::vector< ONEFLOW::Real > hipFlux( nFaces );
    ONEFLOW::FaceFluxView cpuFluxView{ nFaces, 1, cpuFlux.data() };
    ONEFLOW::FaceFluxView hipFluxView{ nFaces, 1, hipFlux.data() };

    ONEFLOW::CpuFluxBackend cpuBackend;
    ONEFLOW::HipFluxBackend hipBackend;
    cpuBackend.CalcInvFlux( state, cpuFluxView, 0 );
    hipBackend.CalcInvFlux( state, hipFluxView, 0 );

    double fluxError = MaxDiff( cpuFlux, hipFlux );
    if ( fluxError > 1.0e-15 )
    {
        std::fprintf( stderr, "Scalar flux FAIL: error %.3e\n", fluxError );
        return false;
    }

    std::vector< ONEFLOW::Real > cpuResidual( nCells, 0.0 );
    std::vector< ONEFLOW::Real > hipResidual( nCells, 0.0 );
    ONEFLOW::FaceConnectivityView conn;
    conn.nFaces = nFaces;
    conn.nBoundaryFaces = nBoundaryFaces;
    conn.leftCell = leftCell.data();
    conn.rightCell = rightCell.data();
    ONEFLOW::ResidualView cpuResView{ nCells, 1, cpuResidual.data() };
    ONEFLOW::ResidualView hipResView{ nCells, 1, hipResidual.data() };
    cpuBackend.AddFaceFlux( cpuFluxView, conn, cpuResView );
    hipBackend.AddFaceFlux( hipFluxView, conn, hipResView );

    double resError = MaxDiff( cpuResidual, hipResidual );
    if ( resError > 1.0e-15 )
    {
        std::fprintf( stderr, "Scalar residual FAIL: error %.3e\n", resError );
        return false;
    }

    std::printf( "OneFLOW HIP scalar flux: PASS (flux %.3e, residual %.3e)\n",
        fluxError, resError );
    return true;
}

// --- Euler Rusanov test (new) ---

bool TestEulerRusanov()
{
    constexpr int nFaces = 128;
    constexpr int nCells = 256;
    constexpr int nBoundaryFaces = 8;
    constexpr int nEq = 3;  // 1D Euler

    // Create a smooth 1D Euler initial condition (same as contract test)
    constexpr double kDx = 1.0 / nCells;
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kGamma = 1.4;

    // Cell-centered conserved state
    std::vector< ONEFLOW::Real > cellState( nCells * nEq );
    for ( int cell = 0; cell < nCells; ++ cell )
    {
        const double x = ( cell + 0.5 ) * kDx;
        const double density = 1.0 + 0.08 * std::sin( 2.0 * kPi * x );
        const double velocity = 0.2 + 0.04 * std::cos( 2.0 * kPi * x );
        const double pressure = 1.0 + 0.05 * std::sin( 4.0 * kPi * x );
        cellState[ 0 * nCells + cell ] = density;
        cellState[ 1 * nCells + cell ] = density * velocity;
        cellState[ 2 * nCells + cell ] = pressure / ( kGamma - 1.0 )
            + 0.5 * density * velocity * velocity;
    }

    // Build face data from cell-centered (simple average for left/right)
    std::vector< ONEFLOW::Real > qLeft( nFaces * nEq );
    std::vector< ONEFLOW::Real > qRight( nFaces * nEq );
    std::vector< ONEFLOW::Real > xNormal( nFaces, 1.0 );
    std::vector< ONEFLOW::Real > area( nFaces, 1.0 );
    std::vector< int > leftCell( nFaces );
    std::vector< int > rightCell( nFaces );

    for ( int face = 0; face < nFaces; ++ face )
    {
        const int il = face;
        const int ir = ( face + 1 ) % nCells;
        leftCell[ face ] = il;
        rightCell[ face ] = ir;
        for ( int eq = 0; eq < nEq; ++ eq )
        {
            qLeft[ eq * nFaces + face ] = cellState[ eq * nCells + il ];
            qRight[ eq * nFaces + face ] = cellState[ eq * nCells + ir ];
        }
    }

    ONEFLOW::FaceStateView state;
    state.nFaces = nFaces;
    state.nEquations = nEq;
    state.qLeft = qLeft.data();
    state.qRight = qRight.data();
    state.xNormal = xNormal.data();
    state.faceArea = area.data();
    state.gamma = kGamma;

    std::vector< ONEFLOW::Real > cpuFlux( nFaces * nEq );
    std::vector< ONEFLOW::Real > hipFlux( nFaces * nEq );
    ONEFLOW::FaceFluxView cpuFluxView{ nFaces, nEq, cpuFlux.data() };
    ONEFLOW::FaceFluxView hipFluxView{ nFaces, nEq, hipFlux.data() };

    ONEFLOW::CpuFluxBackend cpuBackend;
    ONEFLOW::HipFluxBackend hipBackend;
    cpuBackend.CalcInvFlux( state, cpuFluxView, 0 );
    hipBackend.CalcInvFlux( state, hipFluxView, 0 );

    double fluxError = MaxDiff( cpuFlux, hipFlux );
    // Euler flux uses more fp ops, tolerance relaxed slightly
    if ( fluxError > 1.0e-14 )
    {
        std::fprintf( stderr, "Euler flux FAIL: error %.3e\n", fluxError );
        return false;
    }

    // Test residual accumulation
    std::vector< ONEFLOW::Real > cpuResidual( nCells * nEq, 0.0 );
    std::vector< ONEFLOW::Real > hipResidual( nCells * nEq, 0.0 );
    ONEFLOW::FaceConnectivityView conn;
    conn.nFaces = nFaces;
    conn.nBoundaryFaces = nBoundaryFaces;
    conn.leftCell = leftCell.data();
    conn.rightCell = rightCell.data();
    ONEFLOW::ResidualView cpuResView{ nCells, nEq, cpuResidual.data() };
    ONEFLOW::ResidualView hipResView{ nCells, nEq, hipResidual.data() };
    cpuBackend.AddFaceFlux( cpuFluxView, conn, cpuResView );
    hipBackend.AddFaceFlux( hipFluxView, conn, hipResView );

    double resError = MaxDiff( cpuResidual, hipResidual );
    if ( resError > 1.0e-14 )
    {
        std::fprintf( stderr, "Euler residual FAIL: error %.3e\n", resError );
        return false;
    }

    // Physicality check: flux should be finite
    for ( int i = 0; i < nFaces * nEq; ++ i )
    {
        if ( !std::isfinite( cpuFlux[ i ] ) )
        {
            std::fprintf( stderr, "Euler flux non-finite at %d\n", i );
            return false;
        }
    }

    std::printf( "OneFLOW HIP Euler flux: PASS (flux %.3e, residual %.3e, %d faces, %d eq)\n",
        fluxError, resError, nFaces, nEq );
    return true;
}

bool TestMainSolverFiveEquationLaxFriedrichs()
{
    constexpr int nFaces = 257;
    constexpr int nCells = 300;
    constexpr int nEq = 5;
    constexpr double gamma = 1.4;
    std::vector< ONEFLOW::Real > primitiveLeft( nFaces * nEq );
    std::vector< ONEFLOW::Real > primitiveRight( nFaces * nEq );
    std::vector< ONEFLOW::Real > xNormal( nFaces );
    std::vector< ONEFLOW::Real > yNormal( nFaces );
    std::vector< ONEFLOW::Real > zNormal( nFaces );
    std::vector< ONEFLOW::Real > meshVelocityNormal( nFaces );
    std::vector< ONEFLOW::Real > area( nFaces );

    for ( int face = 0; face < nFaces; ++ face )
    {
        const double phase = 0.03125 * face;
        double nx = std::cos( phase );
        double ny = std::sin( phase );
        double nz = 0.25 * std::sin( 0.7 * phase );
        const double normalMagnitude = std::sqrt( nx * nx + ny * ny + nz * nz );
        xNormal[ face ] = nx / normalMagnitude;
        yNormal[ face ] = ny / normalMagnitude;
        zNormal[ face ] = nz / normalMagnitude;
        meshVelocityNormal[ face ] = 0.02 * std::cos( 0.3 * phase );
        area[ face ] = 0.5 + 0.01 * ( face % 19 );

        primitiveLeft[ 0 * nFaces + face ] = 1.0 + 0.05 * std::sin( phase );
        primitiveLeft[ 1 * nFaces + face ] = 0.7 + 0.08 * std::cos( phase );
        primitiveLeft[ 2 * nFaces + face ] = -0.2 + 0.04 * std::sin( 0.5 * phase );
        primitiveLeft[ 3 * nFaces + face ] = 0.1 + 0.03 * std::cos( 0.8 * phase );
        primitiveLeft[ 4 * nFaces + face ] = 1.0 + 0.06 * std::cos( 1.1 * phase );
        primitiveRight[ 0 * nFaces + face ] = 0.95 + 0.04 * std::cos( 0.9 * phase );
        primitiveRight[ 1 * nFaces + face ] = 0.6 + 0.07 * std::sin( 1.2 * phase );
        primitiveRight[ 2 * nFaces + face ] = -0.1 + 0.05 * std::cos( 0.6 * phase );
        primitiveRight[ 3 * nFaces + face ] = 0.15 + 0.02 * std::sin( 0.4 * phase );
        primitiveRight[ 4 * nFaces + face ] = 0.9 + 0.05 * std::sin( phase );
    }

    ONEFLOW::PrimitiveFaceStateView state;
    state.nFaces = nFaces;
    state.nEquations = nEq;
    state.primitiveLeft = primitiveLeft.data();
    state.primitiveRight = primitiveRight.data();
    state.xNormal = xNormal.data();
    state.yNormal = yNormal.data();
    state.zNormal = zNormal.data();
    state.meshVelocityNormal = meshVelocityNormal.data();
    state.faceArea = area.data();
    state.gamma = gamma;

    std::vector< ONEFLOW::Real > cpuFlux( nFaces * nEq );
    std::vector< ONEFLOW::Real > hipFlux( nFaces * nEq );
    ONEFLOW::FaceFluxView cpuFluxView{ nFaces, nEq, cpuFlux.data() };
    ONEFLOW::FaceFluxView hipFluxView{ nFaces, nEq, hipFlux.data() };
    ONEFLOW::CpuFluxBackend cpuBackend;
    ONEFLOW::HipFluxBackend hipBackend;
    ONEFLOW::EulerCpuAdapter adapter;
    adapter.CalcInvFlux( state, cpuFluxView, cpuBackend, 1 );
    adapter.CalcInvFlux( state, hipFluxView, hipBackend, 1 );

    const double fluxError = MaxDiff( cpuFlux, hipFlux );
    if ( fluxError > 1.0e-12 )
    {
        std::fprintf( stderr,
            "Main solver 5-equation Lax-Friedrichs flux FAIL: %.3e\n",
            fluxError );
        return false;
    }
    for ( ONEFLOW::Real value : hipFlux )
    {
        if ( ! std::isfinite( value ) )
        {
            std::fprintf( stderr, "Main solver HIP flux is non-finite\n" );
            return false;
        }
    }

    std::vector< int > leftCell( nFaces );
    std::vector< int > rightCell( nFaces );
    std::vector< unsigned char > boundaryMask( nFaces );
    int nBoundaryFaces = 0;
    for ( int face = 0; face < nFaces; ++ face )
    {
        leftCell[ face ] = face % nCells;
        rightCell[ face ] = ( face * 17 + 3 ) % nCells;
        if ( rightCell[ face ] == leftCell[ face ] )
            rightCell[ face ] = ( rightCell[ face ] + 1 ) % nCells;
        boundaryMask[ face ] = face % 13 == 7 ? 1 : 0;
        nBoundaryFaces += boundaryMask[ face ] != 0;
    }

    std::vector< ONEFLOW::Real > cpuResidual( nCells * nEq );
    std::vector< ONEFLOW::Real > hipResidual( nCells * nEq );
    for ( int i = 0; i < nCells * nEq; ++ i )
    {
        cpuResidual[ i ] = 1.0e-6 * ( i % 7 );
        hipResidual[ i ] = cpuResidual[ i ];
    }
    ONEFLOW::FaceConnectivityView connectivity{
        nFaces, nBoundaryFaces, leftCell.data(), rightCell.data(),
        boundaryMask.data() };
    ONEFLOW::ResidualView cpuResidualView{
        nCells, nEq, cpuResidual.data() };
    ONEFLOW::ResidualView hipResidualView{
        nCells, nEq, hipResidual.data() };
    cpuBackend.AddFaceFlux( cpuFluxView, connectivity, cpuResidualView );
    hipBackend.AddCurrentFaceFlux( connectivity, hipResidualView );

    std::vector< ONEFLOW::Real > fusedFlux( nFaces * nEq );
    std::vector< ONEFLOW::Real > fusedResidual =
        std::vector< ONEFLOW::Real >( nCells * nEq );
    for ( int i = 0; i < nCells * nEq; ++ i )
    {
        fusedResidual[ i ] = 1.0e-6 * ( i % 7 );
    }
    ONEFLOW::FaceFluxView fusedFluxView{
        nFaces, nEq, fusedFlux.data() };
    ONEFLOW::ResidualView fusedResidualView{
        nCells, nEq, fusedResidual.data() };
    ONEFLOW::HipFluxBackend fusedBackend;
    fusedBackend.CalcAndAddPrimitiveFaceFlux(
        state, connectivity, fusedResidualView, 1, & fusedFluxView );
    const double fusedFluxError = MaxDiff( cpuFlux, fusedFlux );
    const double fusedResidualError = MaxDiff( cpuResidual, fusedResidual );
    if ( fusedFluxError > 2.0e-12 || fusedResidualError > 2.0e-12 )
    {
        std::fprintf( stderr,
            "Main solver fused primitive path FAIL: flux %.3e residual %.3e\n",
            fusedFluxError, fusedResidualError );
        return false;
    }

    std::vector< ONEFLOW::Real > noTraceResidual =
        std::vector< ONEFLOW::Real >( nCells * nEq );
    for ( int i = 0; i < nCells * nEq; ++ i )
    {
        noTraceResidual[ i ] = 1.0e-6 * ( i % 7 );
    }
    ONEFLOW::ResidualView noTraceResidualView{
        nCells, nEq, noTraceResidual.data() };
    ONEFLOW::HipFluxBackend noTraceBackend;
    noTraceBackend.CalcAndAddPrimitiveFaceFlux(
        state, connectivity, noTraceResidualView, 1 );
    const double noTraceResidualError =
        MaxDiff( cpuResidual, noTraceResidual );
    if ( noTraceResidualError > 2.0e-12 )
    {
        std::fprintf( stderr,
            "Main solver fused NoTrace path FAIL: %.3e\n",
            noTraceResidualError );
        return false;
    }

    const double residualError = MaxDiff( cpuResidual, hipResidual );
    if ( residualError > 2.0e-12 )
    {
        std::fprintf( stderr,
            "Main solver explicit-mask residual FAIL: %.3e\n",
            residualError );
        return false;
    }

    std::vector< ONEFLOW::Real > expectedResidual( nCells * nEq );
    for ( int i = 0; i < nCells * nEq; ++ i )
    {
        expectedResidual[ i ] = 1.0e-6 * ( i % 7 );
    }
    for ( int face = 0; face < nFaces; ++ face )
    {
        for ( int eq = 0; eq < nEq; ++ eq )
        {
            const ONEFLOW::Real value = hipFlux[ eq * nFaces + face ];
            expectedResidual[ eq * nCells + leftCell[ face ] ] -= value;
            if ( boundaryMask[ face ] == 0 )
            {
                expectedResidual[ eq * nCells + rightCell[ face ] ] += value;
            }
        }
    }
    const double boundarySemanticError =
        MaxDiff( expectedResidual, hipResidual );
    if ( boundarySemanticError > 2.0e-12 )
    {
        std::fprintf( stderr,
            "Main solver boundary-mask semantics FAIL: %.3e\n",
            boundarySemanticError );
        return false;
    }

    double conservationError = 0.0;
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        double residualDelta = 0.0;
        double boundaryFlux = 0.0;
        for ( int cell = 0; cell < nCells; ++ cell )
        {
            const int index = eq * nCells + cell;
            residualDelta += hipResidual[ index ]
                - 1.0e-6 * ( index % 7 );
        }
        for ( int face = 0; face < nFaces; ++ face )
        {
            if ( boundaryMask[ face ] != 0 )
            {
                boundaryFlux -= hipFlux[ eq * nFaces + face ];
            }
        }
        conservationError = std::max(
            conservationError, std::abs( residualDelta - boundaryFlux ) );
    }
    if ( conservationError > 2.0e-11 )
    {
        std::fprintf( stderr,
            "Main solver internal-face conservation FAIL: %.3e\n",
            conservationError );
        return false;
    }

    std::printf(
        "OneFLOW HIP main solver one-call: PASS "
        "(flux %.3e, residual %.3e, fused %.3e/%.3e, "
        "boundary %.3e, conservation %.3e, %d faces, %d eq)\n",
        fluxError, residualError, fusedFluxError, fusedResidualError,
        boundarySemanticError, conservationError,
        nFaces, nEq );
    return true;
}

bool TestMainSolverAleAreaContract()
{
    constexpr int nFaces = 1;
    constexpr int nEq = 5;
    constexpr double gamma = 1.4;
    constexpr double density = 1.2;
    constexpr double u = 0.4;
    constexpr double v = -0.2;
    constexpr double w = 0.1;
    constexpr double pressure = 0.9;
    constexpr double nx = 0.6;
    constexpr double ny = 0.8;
    constexpr double nz = 0.0;
    constexpr double meshVelocityNormal = 0.15;
    constexpr double area = 2.5;
    const ONEFLOW::Real primitive[] = {
        density, u, v, w, pressure };
    const ONEFLOW::Real normalX[] = { nx };
    const ONEFLOW::Real normalY[] = { ny };
    const ONEFLOW::Real normalZ[] = { nz };
    const ONEFLOW::Real meshVelocity[] = { meshVelocityNormal };
    const ONEFLOW::Real faceArea[] = { area };
    ONEFLOW::Real fluxValues[ nEq ] = {};

    ONEFLOW::PrimitiveFaceStateView state;
    state.nFaces = nFaces;
    state.nEquations = nEq;
    state.primitiveLeft = primitive;
    state.primitiveRight = primitive;
    state.xNormal = normalX;
    state.yNormal = normalY;
    state.zNormal = normalZ;
    state.meshVelocityNormal = meshVelocity;
    state.faceArea = faceArea;
    state.gamma = gamma;

    ONEFLOW::FaceFluxView flux{ nFaces, nEq, fluxValues };
    ONEFLOW::HipFluxBackend hipBackend;
    ONEFLOW::EulerCpuAdapter adapter;
    adapter.CalcInvFlux( state, flux, hipBackend, 1 );

    const double relativeNormalVelocity =
        nx * u + ny * v + nz * w - meshVelocityNormal;
    const double massFlux = density * relativeNormalVelocity;
    const double totalEnergy = pressure / ( gamma - 1.0 )
        + 0.5 * density * ( u * u + v * v + w * w );
    const double totalEnthalpy = ( totalEnergy + pressure ) / density;
    const double expected[] = {
        area * massFlux,
        area * ( massFlux * u + nx * pressure ),
        area * ( massFlux * v + ny * pressure ),
        area * ( massFlux * w + nz * pressure ),
        area * ( massFlux * totalEnthalpy
            + meshVelocityNormal * pressure )
    };

    double maxError = 0.0;
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        if ( ! std::isfinite( fluxValues[ eq ] ) )
        {
            std::fprintf( stderr,
                "Main solver ALE/area flux is non-finite at %d\n", eq );
            return false;
        }
        maxError = std::max(
            maxError,
            static_cast< double >(
                std::abs( fluxValues[ eq ] - expected[ eq ] ) ) );
    }
    if ( maxError > 1.0e-12 )
    {
        std::fprintf( stderr,
            "Main solver ALE/face-area ownership FAIL: %.3e\n", maxError );
        return false;
    }

    std::printf(
        "OneFLOW HIP main solver ALE/face-area contract: PASS "
        "(max error %.3e)\n",
        maxError );
    return true;
}

} // namespace

int main()
{
    ONEFLOW::InitializeAccelRuntime( 0, 1 );
    try
    {
        ONEFLOW::RunHipBackendSelfTest();

        bool ok = true;
        ok = TestScalarConvection() && ok;
        ok = TestEulerRusanov() && ok;
        ok = TestMainSolverFiveEquationLaxFriedrichs() && ok;
        ok = TestMainSolverAleAreaContract() && ok;

        ONEFLOW::FinalizeAccelRuntime();
        return ok ? 0 : 1;
    }
    catch ( const std::exception & error )
    {
        std::fprintf( stderr, "OneFLOW HIP smoke: FAIL: %s\n", error.what() );
        ONEFLOW::FinalizeAccelRuntime();
        return 1;
    }
}
