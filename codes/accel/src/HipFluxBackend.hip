/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.
\*---------------------------------------------------------------------------*/

#include "HipFluxBackend.h"

#include "AccelRuntime.h"
#include "EulerCpuAdapter.h"
#include "DeviceBuffer.h"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ONEFLOW
{
namespace
{

void CheckHip( hipError_t status, const char * operation )
{
    if ( status == hipSuccess ) return;
    throw std::runtime_error(
        std::string( operation ) + ": " + hipGetErrorString( status ) );
}

std::unique_ptr< HipFluxBackend > sharedHipFluxBackend;

// ---------------------------------------------------------------------------
// Scalar convection kernel (nEquations == 1)
// ---------------------------------------------------------------------------

__global__ void CalcInvFluxScalarKernel(
    const Real * qLeft,
    const Real * qRight,
    const Real * xNormal,
    const Real * area,
    Real * flux,
    int nFaces )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nFaces ) return;

    const Real normalVelocity = xNormal[ face ];
    const Real magnitude = normalVelocity >= 0.0 ? normalVelocity : -normalVelocity;
    const Real positive = 0.5 * ( normalVelocity + magnitude );
    const Real negative = 0.5 * ( normalVelocity - magnitude );
    flux[ face ] =
        ( qLeft[ face ] * positive + qRight[ face ] * negative ) * area[ face ];
}

__global__ void AddFaceFluxScalarKernel(
    const Real * flux,
    const int * leftCell,
    const int * rightCell,
    const unsigned char * boundaryMask,
    int nBoundaryFaces,
    int nFaces,
    int nCells,
    Real * residual )
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cell >= nCells ) return;

    Real value = 0.0;
    for ( int face = 0; face < nFaces; ++ face )
    {
        const bool boundary = boundaryMask != nullptr
            ? boundaryMask[ face ] != 0
            : face < nBoundaryFaces;
        if ( leftCell[ face ] == cell ) value -= flux[ face ];
        if ( ! boundary && rightCell[ face ] == cell )
        {
            value += flux[ face ];
        }
    }
    residual[ cell ] += value;
}

// ---------------------------------------------------------------------------
// Euler Rusanov kernel (nEquations >= 3)
//
// Layout: equation-major: data[eq * nFaces + face]
// Input:  conserved variables Q = [rho, rho*u, rho*v, rho*w, rho*E]
// Output: numerical flux F = 0.5*(F_L+F_R) - 0.5*|lambda|*(Q_R-Q_L) * area
// ---------------------------------------------------------------------------

struct Primitive
{
    Real density;
    Real u, v, w;
    Real pressure;
    Real enthalpy;  // total enthalpy H = (rho*E + p) / rho
    Real soundSpeed;
};

__device__ void ConservedToPrimitive(
    const Real * q, int face, int nFaces, int nEq,
    Real gamma, Primitive & p )
{
    const Real gamm1 = gamma - 1.0;

    p.density = q[ 0 * nFaces + face ];
    const Real rhou = q[ 1 * nFaces + face ];
    p.u = rhou / p.density;
    Real ke = 0.5 * p.u * p.u;

    if ( nEq >= 4 )
    {
        const Real rhov = q[ 2 * nFaces + face ];
        const Real rhow = q[ 3 * nFaces + face ];
        p.v = rhov / p.density;
        p.w = rhow / p.density;
        ke = 0.5 * ( p.u * p.u + p.v * p.v + p.w * p.w );
    }
    else
    {
        p.v = 0.0;
        p.w = 0.0;
    }

    const Real rhoE = q[ ( nEq - 1 ) * nFaces + face ];
    p.pressure = gamm1 * ( rhoE - ke * p.density );
    p.enthalpy = ( rhoE + p.pressure ) / p.density;
    p.soundSpeed = sqrt( fabs( gamma * p.pressure / p.density ) );
}

__device__ void PhysicalFlux(
    const Primitive & p,
    Real nx, Real ny, Real nz, Real vfn,
    Real * f, int nEq )
{
    const Real normalVelocity = p.u * nx + p.v * ny + p.w * nz;
    const Real massFlux = p.density * normalVelocity;

    f[ 0 ] = massFlux;
    f[ 1 ] = massFlux * p.u + p.pressure * nx;
    f[ 2 ] = massFlux * p.v + p.pressure * ny;
    if ( nEq >= 4 )
    {
        f[ 3 ] = massFlux * p.w + p.pressure * nz;
        f[ 4 ] = massFlux * p.enthalpy + p.pressure * vfn;
    }
    else
    {
        f[ 2 ] = massFlux * p.enthalpy + p.pressure * vfn;
    }
}

__device__ Real MaxWaveSpeed(
    const Primitive & p,
    Real nx, Real ny, Real nz, Real vfn )
{
    const Real normalVelocity = p.u * nx + p.v * ny + p.w * nz;
    return fabs( normalVelocity - vfn ) + p.soundSpeed;
}

__global__ void CalcInvFluxEulerKernel(
    const Real * qLeft,
    const Real * qRight,
    const Real * xNormal,
    const Real * yNormal,
    const Real * zNormal,
    const Real * meshVelocityNormal,
    const Real * area,
    Real * flux,
    int nFaces,
    int nEq,
    Real gamma,
    int scheme )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nFaces ) return;

    const Real nx = xNormal ? xNormal[ face ] : 1.0;
    const Real ny = yNormal ? yNormal[ face ] : 0.0;
    const Real nz = zNormal ? zNormal[ face ] : 0.0;
    const Real vfn = meshVelocityNormal ? meshVelocityNormal[ face ] : 0.0;
    const Real faceArea = area ? area[ face ] : 1.0;

    Primitive pL, pR;
    ConservedToPrimitive( qLeft,  face, nFaces, nEq, gamma, pL );
    ConservedToPrimitive( qRight, face, nFaces, nEq, gamma, pR );

    Real fL[ 5 ] = {};
    Real fR[ 5 ] = {};
    Real lambdaMax = 0.0;
    if ( scheme == 1 )
    {
        const Real ratio = sqrt( pR.density / pL.density );
        const Real coef = 1.0 / ( 1.0 + ratio );
        const Real densityMean = sqrt( pL.density * pR.density );
        const Real uMean = ( pL.u + pR.u * ratio ) * coef;
        const Real vMean = ( pL.v + pR.v * ratio ) * coef;
        const Real wMean = ( pL.w + pR.w * ratio ) * coef;
        const Real pressureMean =
            ( pL.pressure + pR.pressure * ratio ) * coef;
        const Real normalVelocityMean =
            nx * uMean + ny * vMean + nz * wMean - vfn;
        const Real soundSpeedMean = sqrt(
            fabs( gamma * pressureMean / densityMean ) );
        lambdaMax = fabs( normalVelocityMean ) + soundSpeedMean;

        const Real normalVelocityLeft =
            nx * pL.u + ny * pL.v + nz * pL.w - vfn;
        const Real normalVelocityRight =
            nx * pR.u + ny * pR.v + nz * pR.w - vfn;
        const Real massFluxLeft = pL.density * normalVelocityLeft;
        const Real massFluxRight = pR.density * normalVelocityRight;
        fL[ 0 ] = massFluxLeft;
        fL[ 1 ] = massFluxLeft * pL.u + nx * pL.pressure;
        fR[ 0 ] = massFluxRight;
        fR[ 1 ] = massFluxRight * pR.u + nx * pR.pressure;
        if ( nEq >= 4 )
        {
            fL[ 2 ] = massFluxLeft * pL.v + ny * pL.pressure;
            fL[ 3 ] = massFluxLeft * pL.w + nz * pL.pressure;
            fL[ 4 ] = massFluxLeft * pL.enthalpy + vfn * pL.pressure;
            fR[ 2 ] = massFluxRight * pR.v + ny * pR.pressure;
            fR[ 3 ] = massFluxRight * pR.w + nz * pR.pressure;
            fR[ 4 ] = massFluxRight * pR.enthalpy + vfn * pR.pressure;
        }
        else
        {
            fL[ 2 ] = massFluxLeft * pL.enthalpy + vfn * pL.pressure;
            fR[ 2 ] = massFluxRight * pR.enthalpy + vfn * pR.pressure;
        }
    }
    else
    {
        PhysicalFlux( pL, nx, ny, nz, vfn, fL, nEq );
        PhysicalFlux( pR, nx, ny, nz, vfn, fR, nEq );
        const Real lambdaL = MaxWaveSpeed( pL, nx, ny, nz, vfn );
        const Real lambdaR = MaxWaveSpeed( pR, nx, ny, nz, vfn );
        lambdaMax = lambdaL > lambdaR ? lambdaL : lambdaR;
    }

    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const int idx = eq * nFaces + face;
        flux[ idx ] = faceArea * (
            0.5 * ( fL[ eq ] + fR[ eq ] )
            - 0.5 * lambdaMax * ( qRight[ idx ] - qLeft[ idx ] ) );
    }
}

__device__ void PrimitiveToConservedFace(
    const Real * primitive, int face, int nFaces, int nEq,
    Real gamma, Real * q )
{
    const Real density = primitive[ 0 * nFaces + face ];
    const Real u = primitive[ 1 * nFaces + face ];
    const Real v = nEq >= 4 ? primitive[ 2 * nFaces + face ] : 0.0;
    const Real w = nEq >= 4 ? primitive[ 3 * nFaces + face ] : 0.0;
    const Real pressure = primitive[ ( nEq == 3 ? 2 : 4 ) * nFaces + face ];
    const Real kineticEnergy = 0.5 * density *
        ( u * u + v * v + w * w );

    q[ 0 ] = density;
    q[ 1 ] = density * u;
    if ( nEq == 3 )
    {
        q[ 2 ] = pressure / ( gamma - 1.0 ) + kineticEnergy;
    }
    else
    {
        q[ 2 ] = density * v;
        q[ 3 ] = density * w;
        q[ 4 ] = pressure / ( gamma - 1.0 ) + kineticEnergy;
    }
}

__device__ void ConservedValuesToPrimitive(
    const Real * q, int nEq, Real gamma, Primitive & p )
{
    const Real gamm1 = gamma - 1.0;
    p.density = q[ 0 ];
    p.u = q[ 1 ] / p.density;
    p.v = 0.0;
    p.w = 0.0;
    Real kinetic = 0.5 * p.u * p.u;
    if ( nEq >= 4 )
    {
        p.v = q[ 2 ] / p.density;
        p.w = q[ 3 ] / p.density;
        kinetic = 0.5 * ( p.u * p.u + p.v * p.v + p.w * p.w );
    }
    const Real rhoE = q[ nEq - 1 ];
    p.pressure = gamm1 * ( rhoE - kinetic * p.density );
    p.enthalpy = ( rhoE + p.pressure ) / p.density;
    p.soundSpeed = sqrt( fabs( gamma * p.pressure / p.density ) );
}

__global__ void CalcAndScatterPrimitiveEulerKernel(
    const Real * primitiveLeft,
    const Real * primitiveRight,
    const Real * xNormal,
    const Real * yNormal,
    const Real * zNormal,
    const Real * meshVelocityNormal,
    const Real * area,
    const int * leftCell,
    const int * rightCell,
    const unsigned char * boundaryMask,
    int nBoundaryFaces,
    int nFaces,
    int nCells,
    int nEq,
    Real gamma,
    int scheme,
    Real * flux,
    Real * residual )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nFaces ) return;

    const Real nx = xNormal ? xNormal[ face ] : 1.0;
    const Real ny = yNormal ? yNormal[ face ] : 0.0;
    const Real nz = zNormal ? zNormal[ face ] : 0.0;
    const Real vfn = meshVelocityNormal ? meshVelocityNormal[ face ] : 0.0;
    const Real faceArea = area ? area[ face ] : 1.0;

    Real qL[ 5 ] = {};
    Real qR[ 5 ] = {};
    PrimitiveToConservedFace(
        primitiveLeft, face, nFaces, nEq, gamma, qL );
    PrimitiveToConservedFace(
        primitiveRight, face, nFaces, nEq, gamma, qR );

    Primitive pL, pR;
    ConservedValuesToPrimitive( qL, nEq, gamma, pL );
    ConservedValuesToPrimitive( qR, nEq, gamma, pR );

    Real fL[ 5 ] = {};
    Real fR[ 5 ] = {};
    Real lambdaMax = 0.0;
    if ( scheme == 1 )
    {
        const Real ratio = sqrt( pR.density / pL.density );
        const Real coef = 1.0 / ( 1.0 + ratio );
        const Real densityMean = sqrt( pL.density * pR.density );
        const Real uMean = ( pL.u + pR.u * ratio ) * coef;
        const Real vMean = ( pL.v + pR.v * ratio ) * coef;
        const Real wMean = ( pL.w + pR.w * ratio ) * coef;
        const Real pressureMean =
            ( pL.pressure + pR.pressure * ratio ) * coef;
        const Real normalVelocityMean =
            nx * uMean + ny * vMean + nz * wMean - vfn;
        const Real soundSpeedMean = sqrt(
            fabs( gamma * pressureMean / densityMean ) );
        lambdaMax = fabs( normalVelocityMean ) + soundSpeedMean;

        const Real normalVelocityLeft =
            nx * pL.u + ny * pL.v + nz * pL.w - vfn;
        const Real normalVelocityRight =
            nx * pR.u + ny * pR.v + nz * pR.w - vfn;
        const Real massFluxLeft = pL.density * normalVelocityLeft;
        const Real massFluxRight = pR.density * normalVelocityRight;
        fL[ 0 ] = massFluxLeft;
        fL[ 1 ] = massFluxLeft * pL.u + nx * pL.pressure;
        fR[ 0 ] = massFluxRight;
        fR[ 1 ] = massFluxRight * pR.u + nx * pR.pressure;
        if ( nEq >= 4 )
        {
            fL[ 2 ] = massFluxLeft * pL.v + ny * pL.pressure;
            fL[ 3 ] = massFluxLeft * pL.w + nz * pL.pressure;
            fL[ 4 ] = massFluxLeft * pL.enthalpy + vfn * pL.pressure;
            fR[ 2 ] = massFluxRight * pR.v + ny * pR.pressure;
            fR[ 3 ] = massFluxRight * pR.w + nz * pR.pressure;
            fR[ 4 ] = massFluxRight * pR.enthalpy + vfn * pR.pressure;
        }
        else
        {
            fL[ 2 ] = massFluxLeft * pL.enthalpy + vfn * pL.pressure;
            fR[ 2 ] = massFluxRight * pR.enthalpy + vfn * pR.pressure;
        }
    }
    else
    {
        PhysicalFlux( pL, nx, ny, nz, vfn, fL, nEq );
        PhysicalFlux( pR, nx, ny, nz, vfn, fR, nEq );
        const Real lambdaL = MaxWaveSpeed( pL, nx, ny, nz, vfn );
        const Real lambdaR = MaxWaveSpeed( pR, nx, ny, nz, vfn );
        lambdaMax = lambdaL > lambdaR ? lambdaL : lambdaR;
    }

    const int left = leftCell[ face ];
    const int right = rightCell[ face ];
    const bool boundary = boundaryMask != nullptr
        ? boundaryMask[ face ] != 0
        : face < nBoundaryFaces;
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const Real faceFlux = faceArea * (
            0.5 * ( fL[ eq ] + fR[ eq ] )
            - 0.5 * lambdaMax * ( qR[ eq ] - qL[ eq ] ) );
        if ( flux != nullptr )
            flux[ eq * nFaces + face ] = faceFlux;
        atomicAdd(
            &residual[ eq * nCells + left ], -faceFlux );
        if ( ! boundary && right >= 0 && right < nCells )
            atomicAdd(
                &residual[ eq * nCells + right ], faceFlux );
    }
}

__global__ void AddFaceFluxEulerKernel(
    const Real * flux,
    const int * leftCell,
    const int * rightCell,
    const unsigned char * boundaryMask,
    int nBoundaryFaces,
    int nFaces,
    int nCells,
    int nEq,
    Real * residual )
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cell >= nCells ) return;

    for ( int eq = 0; eq < nEq; ++ eq )
    {
        Real value = 0.0;
        for ( int face = 0; face < nFaces; ++ face )
        {
            const bool boundary = boundaryMask != nullptr
                ? boundaryMask[ face ] != 0
                : face < nBoundaryFaces;
            if ( leftCell[ face ] == cell )
                value -= flux[ eq * nFaces + face ];
            if ( ! boundary && rightCell[ face ] == cell )
                value += flux[ eq * nFaces + face ];
        }
        residual[ eq * nCells + cell ] += value;
    }
}

// Scatter each face contribution to its owner cells.  The previous kernel
// assigned one thread per cell and scanned every face, which made the
// residual update O(nCells*nFaces) for an unstructured 3D grid.
__global__ void ScatterFaceFluxEulerKernel(
    const Real * flux,
    const int * leftCell,
    const int * rightCell,
    const unsigned char * boundaryMask,
    int nBoundaryFaces,
    int nFaces,
    int nCells,
    int nEq,
    Real * residual )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nFaces ) return;

    const int left = leftCell[ face ];
    const int right = rightCell[ face ];
    const bool boundary = boundaryMask != nullptr
        ? boundaryMask[ face ] != 0
        : face < nBoundaryFaces;
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const Real faceFlux = flux[ eq * nFaces + face ];
        atomicAdd( &residual[ eq * nCells + left ], -faceFlux );
        if ( ! boundary && right >= 0 && right < nCells )
            atomicAdd( &residual[ eq * nCells + right ], faceFlux );
    }
}

} // namespace

HipFluxBackend & HipFluxBackend::Shared()
{
    if ( ! sharedHipFluxBackend )
    {
        if ( ! AccelRuntime::Instance().IsInitialized() )
        {
            throw std::runtime_error(
                "cannot create shared HIP flux backend before runtime initialization." );
        }
        sharedHipFluxBackend = std::make_unique< HipFluxBackend >();
    }
    return *sharedHipFluxBackend;
}

void HipFluxBackend::ReleaseShared()
{
    sharedHipFluxBackend.reset();
}

void HipFluxBackend::CalcInvFlux(
    const FaceStateView & state,
    FaceFluxView & flux,
    int scheme )
{
    if ( state.nFaces != flux.nFaces || state.nEquations != flux.nEquations )
    {
        throw std::invalid_argument( "HIP flux view dimensions are inconsistent." );
    }
    ValidateFaceStateView( state );

    const int nFaces = state.nFaces;
    const int nEq = state.nEquations;
    const int totalValues = nFaces * nEq;

    if ( nEq == 1 )
    {
        qLeft_.CopyFromHost( state.qLeft, nFaces );
        qRight_.CopyFromHost( state.qRight, nFaces );
        if ( state.xNormal ) xNormal_.CopyFromHost( state.xNormal, nFaces );
        if ( state.faceArea ) area_.CopyFromHost( state.faceArea, nFaces );
        deviceFlux_.Resize( nFaces );

        const int blockSize = 256;
        const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
        hipLaunchKernelGGL(
            CalcInvFluxScalarKernel,
            dim3( gridSize ),
            dim3( blockSize ),
            0,
            0,
            qLeft_.Data(),
            qRight_.Data(),
            state.xNormal ? xNormal_.Data() : nullptr,
            state.faceArea ? area_.Data() : nullptr,
            deviceFlux_.Data(),
            nFaces );
        CheckHip( hipGetLastError(), "CalcInvFluxScalarKernel launch" );
        CheckHip( hipDeviceSynchronize(), "CalcInvFluxScalarKernel synchronize" );
        deviceFlux_.CopyToHost( flux.values, nFaces );
        deviceFluxFaces_ = nFaces;
        deviceFluxEquations_ = nEq;
    }
    else if ( nEq >= 3 )
    {
        qLeft_.CopyFromHost( state.qLeft, totalValues );
        qRight_.CopyFromHost( state.qRight, totalValues );
        if ( state.xNormal ) xNormal_.CopyFromHost( state.xNormal, nFaces );
        if ( state.yNormal ) yNormal_.CopyFromHost( state.yNormal, nFaces );
        if ( state.zNormal ) zNormal_.CopyFromHost( state.zNormal, nFaces );
        if ( state.meshVelocityNormal )
            meshVelocityNormal_.CopyFromHost( state.meshVelocityNormal, nFaces );
        if ( state.faceArea ) area_.CopyFromHost( state.faceArea, nFaces );
        deviceFlux_.Resize( totalValues );

        const int blockSize = 256;
        const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
        hipLaunchKernelGGL(
            CalcInvFluxEulerKernel,
            dim3( gridSize ),
            dim3( blockSize ),
            0,
            0,
            qLeft_.Data(),
            qRight_.Data(),
            state.xNormal ? xNormal_.Data() : nullptr,
            state.yNormal ? yNormal_.Data() : nullptr,
            state.zNormal ? zNormal_.Data() : nullptr,
            state.meshVelocityNormal ? meshVelocityNormal_.Data() : nullptr,
            state.faceArea ? area_.Data() : nullptr,
            deviceFlux_.Data(),
            nFaces,
            nEq,
            state.gamma,
            scheme );
        CheckHip( hipGetLastError(), "CalcInvFluxEulerKernel launch" );
        CheckHip( hipDeviceSynchronize(), "CalcInvFluxEulerKernel synchronize" );
        deviceFlux_.CopyToHost( flux.values, totalValues );
        deviceFluxFaces_ = nFaces;
        deviceFluxEquations_ = nEq;
    }
    else
    {
        throw std::invalid_argument(
            "HIP flux backend: unsupported number of equations." );
    }
}

void HipFluxBackend::AddFaceFlux(
    const FaceFluxView & flux,
    const FaceConnectivityView & connectivity,
    ResidualView & residual )
{
    if ( flux.nFaces != connectivity.nFaces
         || flux.nEquations != residual.nEquations )
    {
        throw std::invalid_argument( "HIP residual view dimensions are inconsistent." );
    }
    ValidateFaceConnectivityView( connectivity );

    const int nFaces = flux.nFaces;
    const int nCells = residual.nCells;
    const int nEq = flux.nEquations;
    const int totalFlux = nFaces * nEq;
    const int totalResidual = nCells * nEq;

    if ( nEq == 1 )
    {
        DeviceBuffer< Real > deviceFlux( nFaces );
        DeviceBuffer< int > leftCell( nFaces );
        DeviceBuffer< int > rightCell( nFaces );
        DeviceBuffer< unsigned char > boundaryMask( nFaces );
        DeviceBuffer< Real > deviceResidual( nCells );

        deviceFlux.CopyFromHost( flux.values, nFaces );
        leftCell.CopyFromHost( connectivity.leftCell, nFaces );
        rightCell.CopyFromHost( connectivity.rightCell, nFaces );
        if ( connectivity.boundaryMask )
            boundaryMask.CopyFromHost( connectivity.boundaryMask, nFaces );
        deviceResidual.CopyFromHost( residual.values, nCells );

        const int blockSize = 256;
        const int gridSize = ( nCells + blockSize - 1 ) / blockSize;
        hipLaunchKernelGGL(
            AddFaceFluxScalarKernel,
            dim3( gridSize ),
            dim3( blockSize ),
            0,
            0,
            deviceFlux.Data(),
            leftCell.Data(),
            rightCell.Data(),
            connectivity.boundaryMask ? boundaryMask.Data() : nullptr,
            connectivity.nBoundaryFaces,
            nFaces,
            nCells,
            deviceResidual.Data() );
        CheckHip( hipGetLastError(), "AddFaceFluxScalarKernel launch" );
        CheckHip( hipDeviceSynchronize(), "AddFaceFluxScalarKernel synchronize" );
        deviceResidual.CopyToHost( residual.values, nCells );
    }
    else
    {
        DeviceBuffer< Real > deviceFlux( totalFlux );
        DeviceBuffer< int > leftCell( nFaces );
        DeviceBuffer< int > rightCell( nFaces );
        DeviceBuffer< unsigned char > boundaryMask( nFaces );
        DeviceBuffer< Real > deviceResidual( totalResidual );

        deviceFlux.CopyFromHost( flux.values, totalFlux );
        leftCell.CopyFromHost( connectivity.leftCell, nFaces );
        rightCell.CopyFromHost( connectivity.rightCell, nFaces );
        if ( connectivity.boundaryMask )
            boundaryMask.CopyFromHost( connectivity.boundaryMask, nFaces );
        deviceResidual.CopyFromHost( residual.values, totalResidual );

        const int blockSize = 256;
        const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
        hipLaunchKernelGGL(
            ScatterFaceFluxEulerKernel,
            dim3( gridSize ),
            dim3( blockSize ),
            0,
            0,
            deviceFlux.Data(),
            leftCell.Data(),
            rightCell.Data(),
            connectivity.boundaryMask ? boundaryMask.Data() : nullptr,
            connectivity.nBoundaryFaces,
            nFaces,
            nCells,
            nEq,
            deviceResidual.Data() );
        CheckHip( hipGetLastError(), "AddFaceFluxEulerKernel launch" );
        CheckHip( hipDeviceSynchronize(), "AddFaceFluxEulerKernel synchronize" );
        deviceResidual.CopyToHost( residual.values, totalResidual );
    }
}

void HipFluxBackend::AddCurrentFaceFlux(
    const FaceConnectivityView & connectivity,
    ResidualView & residual )
{
    ValidateFaceConnectivityView( connectivity );
    if ( deviceFluxFaces_ != connectivity.nFaces
         || deviceFluxEquations_ != residual.nEquations
         || deviceFluxEquations_ < 3
         || residual.nCells <= 0
         || residual.values == nullptr )
    {
        throw std::invalid_argument(
            "HIP cached residual request does not match the current flux." );
    }

    const int nFaces = connectivity.nFaces;
    const int nCells = residual.nCells;
    const int nEq = residual.nEquations;
    const int totalResidual = nCells * nEq;

    DeviceBuffer< int > leftCell( nFaces );
    DeviceBuffer< int > rightCell( nFaces );
    DeviceBuffer< unsigned char > boundaryMask( nFaces );
    DeviceBuffer< Real > deviceResidual( totalResidual );

    leftCell.CopyFromHost( connectivity.leftCell, nFaces );
    rightCell.CopyFromHost( connectivity.rightCell, nFaces );
    if ( connectivity.boundaryMask )
        boundaryMask.CopyFromHost( connectivity.boundaryMask, nFaces );
    deviceResidual.CopyFromHost( residual.values, totalResidual );

    const int blockSize = 256;
    const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
    hipLaunchKernelGGL(
        ScatterFaceFluxEulerKernel,
        dim3( gridSize ),
        dim3( blockSize ),
        0,
        0,
        deviceFlux_.Data(),
        leftCell.Data(),
        rightCell.Data(),
        connectivity.boundaryMask ? boundaryMask.Data() : nullptr,
        connectivity.nBoundaryFaces,
        nFaces,
        nCells,
        nEq,
        deviceResidual.Data() );
    CheckHip( hipGetLastError(), "ScatterFaceFluxEulerKernel launch" );
    CheckHip(
        hipDeviceSynchronize(),
        "ScatterFaceFluxEulerKernel synchronize" );
    deviceResidual.CopyToHost( residual.values, totalResidual );
}

void HipFluxBackend::CalcAndAddPrimitiveFaceFlux(
    const PrimitiveFaceStateView & state,
    const FaceConnectivityView & connectivity,
    ResidualView & residual,
    int scheme,
    FaceFluxView * hostFlux,
    const void * cacheKey )
{
    if ( state.nFaces <= 0 || state.nEquations < 3
         || state.primitiveLeft == nullptr
         || state.primitiveRight == nullptr
         || state.gamma <= 1.0
         || state.nFaces != connectivity.nFaces
         || residual.nCells <= 0
         || residual.nEquations != state.nEquations
         || residual.values == nullptr )
    {
        throw std::invalid_argument(
            "invalid fused HIP primitive residual request." );
    }
    if ( hostFlux != nullptr
         && ( hostFlux->nFaces != state.nFaces
              || hostFlux->nEquations != state.nEquations
              || hostFlux->values == nullptr ) )
    {
        throw std::invalid_argument(
            "invalid fused HIP host flux request." );
    }
    ValidateFaceConnectivityView( connectivity );

    const int nFaces = state.nFaces;
    const int nCells = residual.nCells;
    const int nEq = state.nEquations;
    const int totalFlux = nFaces * nEq;
    const int totalResidual = nCells * nEq;

    qLeft_.CopyFromHost( state.primitiveLeft, totalFlux );
    qRight_.CopyFromHost( state.primitiveRight, totalFlux );

    const bool geometryCacheMatches =
        geometryCacheValid_
        && cachedFaces_ == nFaces
        && cachedBoundaryFaces_ == connectivity.nBoundaryFaces
        && cachedOwner_ == cacheKey
        && cachedXNormal_ == state.xNormal
        && cachedYNormal_ == state.yNormal
        && cachedZNormal_ == state.zNormal
        && cachedMeshVelocityNormal_ == state.meshVelocityNormal
        && cachedArea_ == state.faceArea
        && cachedLeftCell_ == connectivity.leftCell
        && cachedRightCell_ == connectivity.rightCell
        && cachedBoundaryMask_ == connectivity.boundaryMask;
    if ( ! geometryCacheMatches )
    {
        if ( state.xNormal )
            xNormal_.CopyFromHost( state.xNormal, nFaces );
        if ( state.yNormal )
            yNormal_.CopyFromHost( state.yNormal, nFaces );
        if ( state.zNormal )
            zNormal_.CopyFromHost( state.zNormal, nFaces );
        if ( state.meshVelocityNormal )
        {
            meshVelocityNormal_.CopyFromHost(
                state.meshVelocityNormal, nFaces );
        }
        if ( state.faceArea )
            area_.CopyFromHost( state.faceArea, nFaces );
        leftCell_.CopyFromHost( connectivity.leftCell, nFaces );
        rightCell_.CopyFromHost( connectivity.rightCell, nFaces );
        if ( connectivity.boundaryMask )
        {
            boundaryMask_.CopyFromHost(
                connectivity.boundaryMask, nFaces );
        }
        cachedXNormal_ = state.xNormal;
        cachedYNormal_ = state.yNormal;
        cachedZNormal_ = state.zNormal;
        cachedMeshVelocityNormal_ = state.meshVelocityNormal;
        cachedArea_ = state.faceArea;
        cachedLeftCell_ = connectivity.leftCell;
        cachedRightCell_ = connectivity.rightCell;
        cachedBoundaryMask_ = connectivity.boundaryMask;
        cachedOwner_ = cacheKey;
        cachedFaces_ = nFaces;
        cachedBoundaryFaces_ = connectivity.nBoundaryFaces;
        geometryCacheValid_ = true;
    }
    deviceResidual_.CopyFromHost( residual.values, totalResidual );
    if ( hostFlux != nullptr ) deviceFlux_.Resize( totalFlux );

    const int blockSize = 256;
    const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
    hipLaunchKernelGGL(
        CalcAndScatterPrimitiveEulerKernel,
        dim3( gridSize ),
        dim3( blockSize ),
        0,
        0,
        qLeft_.Data(),
        qRight_.Data(),
        state.xNormal ? xNormal_.Data() : nullptr,
        state.yNormal ? yNormal_.Data() : nullptr,
        state.zNormal ? zNormal_.Data() : nullptr,
        state.meshVelocityNormal ? meshVelocityNormal_.Data() : nullptr,
        state.faceArea ? area_.Data() : nullptr,
        leftCell_.Data(),
        rightCell_.Data(),
        connectivity.boundaryMask ? boundaryMask_.Data() : nullptr,
        connectivity.nBoundaryFaces,
        nFaces,
        nCells,
        nEq,
        state.gamma,
        scheme,
        hostFlux != nullptr ? deviceFlux_.Data() : nullptr,
        deviceResidual_.Data() );
    CheckHip(
        hipGetLastError(), "CalcAndScatterPrimitiveEulerKernel launch" );
    CheckHip(
        hipDeviceSynchronize(),
        "CalcAndScatterPrimitiveEulerKernel synchronize" );
    deviceResidual_.CopyToHost( residual.values, totalResidual );
    if ( hostFlux != nullptr )
    {
        deviceFlux_.CopyToHost( hostFlux->values, totalFlux );
        deviceFluxFaces_ = nFaces;
        deviceFluxEquations_ = nEq;
    }
}

}
