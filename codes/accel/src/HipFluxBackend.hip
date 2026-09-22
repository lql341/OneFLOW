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
#include "StageProfiler.h"

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

class HipNs3DBackendState final : public Ns3DBackendState
{
public:
    explicit HipNs3DBackendState( int deviceIdValue )
        : deviceId_( deviceIdValue )
    {
    }

    AccelBackendKind Backend() const noexcept override
    {
        return AccelBackendKind::HIP;
    }

    int DeviceId() const noexcept override
    {
        return deviceId_;
    }

    HipGradientStorage & Gradient() noexcept
    {
        return gradient_;
    }

private:
    int deviceId_ = -1;
    HipGradientStorage gradient_;
};

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

__global__ void GradientZeroKernel( Real * gradient, int nValues )
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index < nValues ) gradient[ index ] = 0.0;
}

__global__ void GradientAccumulateKernel(
    const Real * q, const Real * xFace, const Real * yFace,
    const Real * zFace, const Real * xNormal,
    const Real * yNormal, const Real * zNormal, const Real * faceArea,
    const Real * xCell, const Real * yCell,
    const Real * zCell, const int * leftCell, const int * rightCell,
    int nFaces, int nBoundaryFaces, int nCells, int nEq,
    Real * gradient )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nFaces ) return;
    const int left = leftCell[ face ];
    const int right = rightCell[ face ];
    const Real dxl = xFace[ face ] - xCell[ left ];
    const Real dyl = yFace[ face ] - yCell[ left ];
    const Real dzl = zFace[ face ] - zCell[ left ];
    const Real dxr = xFace[ face ] - xCell[ right ];
    const Real dyr = yFace[ face ] - yCell[ right ];
    const Real dzr = zFace[ face ] - zCell[ right ];
    const Real delt1 = sqrt( dxl * dxl + dyl * dyl + dzl * dzl );
    const Real delt2 = sqrt( dxr * dxr + dyr * dyr + dzr * dzr );
    const Real delta = 1.0 / ( delt1 + delt2 + 1.0e-40 );
    const Real cl = delt2 * delta;
    const Real cr = delt1 * delta;
    const Real ax = xNormal[ face ] * faceArea[ face ];
    const Real ay = yNormal[ face ] * faceArea[ face ];
    const Real az = zNormal[ face ] * faceArea[ face ];
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const Real value = cl * q[ eq * ( nCells + nBoundaryFaces ) + left ]
            + cr * q[ eq * ( nCells + nBoundaryFaces ) + right ];
        const int base = eq * 3 * ( nCells + nBoundaryFaces );
        atomicAdd( &gradient[ base + left ], ax * value );
        atomicAdd( &gradient[ base + ( nCells + nBoundaryFaces ) + left ], ay * value );
        atomicAdd( &gradient[ base + 2 * ( nCells + nBoundaryFaces ) + left ], az * value );
        if ( face >= nBoundaryFaces )
        {
            atomicAdd( &gradient[ base + right ], -ax * value );
            atomicAdd( &gradient[ base + ( nCells + nBoundaryFaces ) + right ], -ay * value );
            atomicAdd( &gradient[ base + 2 * ( nCells + nBoundaryFaces ) + right ], -az * value );
        }
    }
}

__global__ void GradientNormalizeKernel(
    Real * gradient, const Real * volume, int nCells,
    int nBoundaryFaces, int nEq )
{
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cell >= nCells ) return;
    const int totalCells = nCells + nBoundaryFaces;
    const Real inverseVolume = 1.0 / volume[ cell ];
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const int base = eq * 3 * totalCells;
        gradient[ base + cell ] *= inverseVolume;
        gradient[ base + totalCells + cell ] *= inverseVolume;
        gradient[ base + 2 * totalCells + cell ] *= inverseVolume;
    }
}

__global__ void GradientCopyBoundaryKernel(
    Real * gradient, const int * leftCell, const int * rightCell,
    int nCells, int nBoundaryFaces, int nEq )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nBoundaryFaces ) return;
    const int totalCells = nCells + nBoundaryFaces;
    const int left = leftCell[ face ];
    const int ghost = rightCell[ face ];
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const int base = eq * 3 * totalCells;
        gradient[ base + ghost ] = gradient[ base + left ];
        gradient[ base + totalCells + ghost ] =
            gradient[ base + totalCells + left ];
        gradient[ base + 2 * totalCells + ghost ] =
            gradient[ base + 2 * totalCells + left ];
    }
}


__global__ void ReconstructFaceEulerKernel(
    const Real * q,
    const Real * gradient,
    const Real * xFace,
    const Real * yFace,
    const Real * zFace,
    const Real * xCell,
    const Real * yCell,
    const Real * zCell,
    const int * leftCell,
    const int * rightCell,
    const unsigned char * boundaryMask,
    const ReconstructionBoundaryOperation * boundaryOperation,
    const Real * bcQ,
    int nCells,
    int nBoundaryFaces,
    int nFaces,
    int nEq,
    int physicality,
    int densityComponent,
    int pressureComponent,
    Real * qf1,
    Real * qf2 )
{
    const int face = blockIdx.x * blockDim.x + threadIdx.x;
    if ( face >= nFaces ) return;

    const int totalCells = nCells + nBoundaryFaces;
    const int left = leftCell[ face ];
    const int right = rightCell[ face ];
    Real leftValue[ 5 ] = {};
    Real rightValue[ 5 ] = {};
    Real leftCandidate[ 5 ] = {};
    Real rightCandidate[ 5 ] = {};
    const Real dxl = xFace[ face ] - xCell[ left ];
    const Real dyl = yFace[ face ] - yCell[ left ];
    const Real dzl = zFace[ face ] - zCell[ left ];
    const Real dxr = xFace[ face ] - xCell[ right ];
    const Real dyr = yFace[ face ] - yCell[ right ];
    const Real dzr = zFace[ face ] - zCell[ right ];

    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const int qBase = eq * totalCells;
        const int gradientBase = eq * 3 * totalCells;
        leftValue[ eq ] = q[ qBase + left ];
        rightValue[ eq ] = q[ qBase + right ];
        leftCandidate[ eq ] = leftValue[ eq ]
            + gradient[ gradientBase + left ] * dxl
            + gradient[ gradientBase + totalCells + left ] * dyl
            + gradient[ gradientBase + 2 * totalCells + left ] * dzl;
        rightCandidate[ eq ] = rightValue[ eq ]
            + gradient[ gradientBase + right ] * dxr
            + gradient[ gradientBase + totalCells + right ] * dyr
            + gradient[ gradientBase + 2 * totalCells + right ] * dzr;
    }

    bool leftPhysical = true;
    bool rightPhysical = true;
    if ( physicality != 0 )
    {
        leftPhysical = leftCandidate[ densityComponent ] > 0.0
            && leftCandidate[ pressureComponent ] > 0.0;
        rightPhysical = rightCandidate[ densityComponent ] > 0.0
            && rightCandidate[ pressureComponent ] > 0.0;
    }
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        qf1[ eq * nFaces + face ] =
            leftPhysical ? leftCandidate[ eq ] : leftValue[ eq ];
        qf2[ eq * nFaces + face ] =
            rightPhysical ? rightCandidate[ eq ] : rightValue[ eq ];
    }

    const bool boundary = boundaryMask != nullptr
        ? boundaryMask[ face ] != 0
        : face < nBoundaryFaces;
    if ( ! boundary ) return;

    ReconstructionBoundaryOperation operation =
        ReconstructionBoundaryOperation::Average;
    if ( boundaryOperation != nullptr ) operation = boundaryOperation[ face ];
    if ( operation == ReconstructionBoundaryOperation::Preserve ) return;
    if ( operation == ReconstructionBoundaryOperation::SolidOverride )
    {
        for ( int eq = 0; eq < nEq; ++ eq )
        {
            const Real value = bcQ[ eq * nFaces + face ];
            qf1[ eq * nFaces + face ] = value;
            qf2[ eq * nFaces + face ] = value;
        }
        return;
    }
    for ( int eq = 0; eq < nEq; ++ eq )
    {
        const Real value = 0.5 * ( leftValue[ eq ] + rightValue[ eq ] );
        qf1[ eq * nFaces + face ] = value;
        qf2[ eq * nFaces + face ] = value;
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

std::unique_ptr< Ns3DBackendState > CreateHipNs3DBackendState(
    int deviceId )
{
    if ( deviceId < 0 )
    {
        throw std::invalid_argument(
            "cannot create HIP Ns3D state with an invalid device" );
    }
    return std::make_unique< HipNs3DBackendState >( deviceId );
}

void HipFluxBackend::BindState(
    const void * cacheKey,
    Ns3DDeviceState & state )
{
    if ( cacheKey == nullptr )
    {
        throw std::invalid_argument(
            "cannot bind HIP Ns3D state without a cache key" );
    }
    if ( state.key.backend != AccelBackendKind::HIP
         || state.key.deviceId
             != AccelRuntime::Instance().Context().selectedDevice )
    {
        throw std::invalid_argument(
            "HIP Ns3D state key does not match the active device" );
    }
    auto * hipState =
        dynamic_cast< HipNs3DBackendState * >( state.BackendState() );
    if ( hipState == nullptr
         || hipState->DeviceId() != state.key.deviceId )
    {
        throw std::invalid_argument(
            "HIP Ns3D state has no matching backend ownership" );
    }
    const auto found = boundStates_.find( cacheKey );
    if ( found != boundStates_.end() && found->second != &state )
    {
        throw std::logic_error(
            "HIP grid cache key is already bound to another Ns3D state" );
    }
    boundStates_[ cacheKey ] = &state;
}

void HipFluxBackend::UnbindState( const void * cacheKey ) noexcept
{
    if ( cacheKey != nullptr ) boundStates_.erase( cacheKey );
}

void HipFluxBackend::CalcGradient( const CellGradientView & view )
{
    ValidateCellGradientView( view );
    HipGradientStorage * storage = &fallbackGradientStorage_;
    Ns3DDeviceState * owner = nullptr;
    const auto binding = boundStates_.find( view.cacheKey );
    if ( binding != boundStates_.end() )
    {
        owner = binding->second;
        auto * hipState =
            dynamic_cast< HipNs3DBackendState * >( owner->BackendState() );
        if ( hipState == nullptr )
        {
            throw std::logic_error(
                "bound HIP Ns3D state lost its backend ownership" );
        }
        storage = &hipState->Gradient();
    }
    const int totalCells = view.nCells + view.nGhostCells;
    const int totalState = view.nEquations * totalCells;
    const int totalGradient = view.nEquations * 3 * totalCells;
    storage->cellState.Resize( totalState );
    for ( int equation = 0; equation < view.nEquations; ++ equation )
    {
        AccelRuntime::Instance().Backend().Copy(
            storage->cellState.Data() + equation * totalCells,
            view.q[ equation ], totalCells * sizeof( Real ),
            AccelCopyKind::HostToDevice );
    }
    const void * ownerToken =
        owner != nullptr ? owner->OwnerToken() : view.cacheKey;
    const std::uint64_t topologyGeneration =
        owner != nullptr ? owner->TopologyGeneration() : 0;
    const bool cacheMatches = storage->geometryCacheValid
        && storage->cachedOwnerToken == ownerToken
        && storage->cachedTopologyGeneration == topologyGeneration
        && storage->cachedXFace == view.xFace
        && storage->cachedYFace == view.yFace
        && storage->cachedZFace == view.zFace
        && storage->cachedXNormal == view.xNormal
        && storage->cachedYNormal == view.yNormal
        && storage->cachedZNormal == view.zNormal
        && storage->cachedFaceArea == view.faceArea
        && storage->cachedXCell == view.xCell
        && storage->cachedYCell == view.yCell
        && storage->cachedZCell == view.zCell
        && storage->cachedVolume == view.cellVolume
        && storage->cachedLeftCell == view.leftCell
        && storage->cachedRightCell == view.rightCell
        && storage->cachedCells == view.nCells
        && storage->cachedGhostCells == view.nGhostCells
        && storage->cachedFaces == view.nFaces
        && storage->cachedBoundaryFaces == view.nBoundaryFaces;
    if ( ! cacheMatches )
    {
        storage->xFaceCenter.CopyFromHost( view.xFace, view.nFaces );
        storage->yFaceCenter.CopyFromHost( view.yFace, view.nFaces );
        storage->zFaceCenter.CopyFromHost( view.zFace, view.nFaces );
        storage->xNormal.CopyFromHost( view.xNormal, view.nFaces );
        storage->yNormal.CopyFromHost( view.yNormal, view.nFaces );
        storage->zNormal.CopyFromHost( view.zNormal, view.nFaces );
        storage->faceArea.CopyFromHost( view.faceArea, view.nFaces );
        storage->xCellCenter.CopyFromHost( view.xCell, totalCells );
        storage->yCellCenter.CopyFromHost( view.yCell, totalCells );
        storage->zCellCenter.CopyFromHost( view.zCell, totalCells );
        storage->cellVolume.CopyFromHost( view.cellVolume, view.nCells );
        storage->leftCell.CopyFromHost( view.leftCell, view.nFaces );
        storage->rightCell.CopyFromHost( view.rightCell, view.nFaces );
        storage->cachedOwnerToken = ownerToken;
        storage->cachedTopologyGeneration = topologyGeneration;
        storage->cachedXFace = view.xFace;
        storage->cachedYFace = view.yFace;
        storage->cachedZFace = view.zFace;
        storage->cachedXNormal = view.xNormal;
        storage->cachedYNormal = view.yNormal;
        storage->cachedZNormal = view.zNormal;
        storage->cachedFaceArea = view.faceArea;
        storage->cachedXCell = view.xCell;
        storage->cachedYCell = view.yCell;
        storage->cachedZCell = view.zCell;
        storage->cachedVolume = view.cellVolume;
        storage->cachedLeftCell = view.leftCell;
        storage->cachedRightCell = view.rightCell;
        storage->cachedCells = view.nCells;
        storage->cachedGhostCells = view.nGhostCells;
        storage->cachedFaces = view.nFaces;
        storage->cachedBoundaryFaces = view.nBoundaryFaces;
        storage->geometryCacheValid = true;
    }
    storage->cellGradient.Resize( totalGradient );
    const int blockSize = 256;
    hipLaunchKernelGGL(
        GradientZeroKernel,
        dim3( ( totalGradient + blockSize - 1 ) / blockSize ),
        dim3( blockSize ), 0, 0, storage->cellGradient.Data(),
        totalGradient );
    hipLaunchKernelGGL(
        GradientAccumulateKernel,
        dim3( ( view.nFaces + blockSize - 1 ) / blockSize ),
        dim3( blockSize ), 0, 0, storage->cellState.Data(),
        storage->xFaceCenter.Data(), storage->yFaceCenter.Data(),
        storage->zFaceCenter.Data(), storage->xNormal.Data(),
        storage->yNormal.Data(), storage->zNormal.Data(),
        storage->faceArea.Data(), storage->xCellCenter.Data(),
        storage->yCellCenter.Data(), storage->zCellCenter.Data(),
        storage->leftCell.Data(), storage->rightCell.Data(),
        view.nFaces, view.nBoundaryFaces, view.nCells,
        view.nEquations, storage->cellGradient.Data() );
    hipLaunchKernelGGL(
        GradientNormalizeKernel,
        dim3( ( view.nCells + blockSize - 1 ) / blockSize ),
        dim3( blockSize ), 0, 0, storage->cellGradient.Data(),
        storage->cellVolume.Data(), view.nCells,
        view.nBoundaryFaces, view.nEquations );
    if ( view.nBoundaryFaces > 0 )
    {
        hipLaunchKernelGGL(
            GradientCopyBoundaryKernel,
            dim3( ( view.nBoundaryFaces + blockSize - 1 ) / blockSize ),
            dim3( blockSize ), 0, 0, storage->cellGradient.Data(),
            storage->leftCell.Data(), storage->rightCell.Data(),
            view.nCells, view.nBoundaryFaces, view.nEquations );
    }
    CheckHip( hipGetLastError(), "gradient kernel launch" );
    CheckHip( hipDeviceSynchronize(), "gradient kernel synchronize" );
    for ( int equation = 0; equation < view.nEquations; ++ equation )
    {
        const int base = equation * 3 * totalCells;
        AccelRuntime::Instance().Backend().Copy(
            view.dqdx[ equation ], storage->cellGradient.Data() + base,
            totalCells * sizeof( Real ), AccelCopyKind::DeviceToHost );
        AccelRuntime::Instance().Backend().Copy(
            view.dqdy[ equation ],
            storage->cellGradient.Data() + base + totalCells,
            totalCells * sizeof( Real ), AccelCopyKind::DeviceToHost );
        AccelRuntime::Instance().Backend().Copy(
            view.dqdz[ equation ],
            storage->cellGradient.Data() + base + 2 * totalCells,
            totalCells * sizeof( Real ), AccelCopyKind::DeviceToHost );
    }
}


void HipFluxBackend::ReconstructFaceValues(
    const CellFaceReconstructionView & view,
    bool downloadToHost )
{
    ValidateCellFaceReconstructionView( view );
    if ( view.limiterMode != ReconstructionLimiterMode::Disabled )
    {
        throw std::invalid_argument(
            "HIP device reconstruction currently requires limiter off" );
    }
    const void * cacheKey = view.cacheKey != nullptr
        ? view.cacheKey : view.ownerToken;
    HipGradientStorage * storage = &fallbackGradientStorage_;
    Ns3DDeviceState * owner = nullptr;
    const auto binding = boundStates_.find( cacheKey );
    if ( binding != boundStates_.end() )
    {
        owner = binding->second;
        auto * hipState =
            dynamic_cast< HipNs3DBackendState * >( owner->BackendState() );
        if ( hipState == nullptr )
        {
            throw std::logic_error(
                "bound HIP Ns3D state lost its backend ownership" );
        }
        storage = &hipState->Gradient();
    }
    if ( storage->cellState.Size() <
             static_cast< std::size_t >( view.nEquations )
                 * ( view.nCells + view.nGhostCells )
         || storage->cellGradient.Size() <
             static_cast< std::size_t >( view.nEquations ) * 3
                 * ( view.nCells + view.nGhostCells )
         || ! storage->geometryCacheValid
         || storage->cachedFaces != view.nFaces
         || storage->cachedBoundaryFaces != view.nBoundaryFaces )
    {
        throw std::logic_error(
            "HIP reconstruction requires a preceding compatible gradient" );
    }

    const int totalCells = view.nCells + view.nGhostCells;
    const int totalFaces = view.nFaces;
    const int totalValues = view.nEquations * totalFaces;
    storage->qf1.Resize( totalValues );
    storage->qf2.Resize( totalValues );
    storage->boundaryOperation.Resize( totalFaces );
    if ( view.boundaryOperation != nullptr )
    {
        storage->boundaryOperation.CopyFromHost(
            view.boundaryOperation, totalFaces );
    }
    else
    {
        std::vector< ReconstructionBoundaryOperation > defaultOperation(
            totalFaces, ReconstructionBoundaryOperation::Preserve );
        storage->boundaryOperation.CopyFromHost(
            defaultOperation.data(), totalFaces );
    }
    if ( view.boundaryMask != nullptr )
    {
        storage->boundaryMask.CopyFromHost(
            view.boundaryMask, totalFaces );
        storage->boundaryMaskValid = true;
    }
    else
    {
        storage->boundaryMask.Reset();
        storage->boundaryMaskValid = false;
    }
    const bool hasSolid = view.hasSolidBoundary;
    if ( hasSolid )
    {
        storage->bcQ.Resize( totalValues );
        for ( int equation = 0; equation < view.nEquations; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                storage->bcQ.Data() + equation * totalFaces,
                view.bcQ[ equation ], totalFaces * sizeof( Real ),
                AccelCopyKind::HostToDevice );
        }
    }
    else
    {
        storage->bcQ.Reset();
    }
    storage->cachedEquations = view.nEquations;
    storage->cachedFieldGeneration = view.fieldGeneration;

    const int blockSize = 256;
    hipLaunchKernelGGL(
        ReconstructFaceEulerKernel,
        dim3( ( totalFaces + blockSize - 1 ) / blockSize ),
        dim3( blockSize ), 0, 0,
        storage->cellState.Data(), storage->cellGradient.Data(),
        storage->xFaceCenter.Data(), storage->yFaceCenter.Data(),
        storage->zFaceCenter.Data(), storage->xCellCenter.Data(),
        storage->yCellCenter.Data(), storage->zCellCenter.Data(),
        storage->leftCell.Data(), storage->rightCell.Data(),
        storage->boundaryMaskValid ? storage->boundaryMask.Data() : nullptr,
        storage->boundaryOperation.Data(),
        hasSolid ? storage->bcQ.Data() : nullptr,
        view.nCells, view.nBoundaryFaces, view.nFaces, view.nEquations,
        view.physicality == ReconstructionPhysicalityPolicy::None ? 0 : 1,
        view.densityComponent, view.pressureComponent,
        storage->qf1.Data(), storage->qf2.Data() );
    CheckHip( hipGetLastError(), "reconstruction kernel launch" );
    CheckHip( hipDeviceSynchronize(), "reconstruction kernel synchronize" );

    if ( downloadToHost )
    {
        for ( int equation = 0; equation < view.nEquations; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                view.qLeft[ equation ], storage->qf1.Data() + equation * totalFaces,
                totalFaces * sizeof( Real ), AccelCopyKind::DeviceToHost );
            AccelRuntime::Instance().Backend().Copy(
                view.qRight[ equation ], storage->qf2.Data() + equation * totalFaces,
                totalFaces * sizeof( Real ), AccelCopyKind::DeviceToHost );
        }
    }
}

void HipFluxBackend::CalcAndAddReconstructedFaceFlux(
    const FaceConnectivityView & connectivity,
    ResidualView & residual,
    int scheme,
    Real gamma,
    FaceFluxView * hostFlux,
    const void * cacheKey )
{
    ValidateFaceConnectivityView( connectivity );
    if ( residual.nCells <= 0 || residual.nEquations < 3 || gamma <= 1.0 )
    {
        throw std::invalid_argument(
            "invalid reconstructed HIP residual request" );
    }
    const void * key = cacheKey;
    if ( key == nullptr )
    {
        throw std::invalid_argument(
            "reconstructed HIP flux requires a bound cache key" );
    }
    auto binding = boundStates_.find( key );
    HipGradientStorage * storage = &fallbackGradientStorage_;
    if ( binding != boundStates_.end() )
    {
        auto * hipState = dynamic_cast< HipNs3DBackendState * >(
            binding->second->BackendState() );
        if ( hipState == nullptr )
            throw std::logic_error( "bound HIP state is not HIP-owned" );
        storage = &hipState->Gradient();
    }
    const int nFaces = storage->cachedFaces;
    const int nEq = storage->cachedEquations;
    const int nCells = residual.nCells;
    if ( nFaces <= 0 || nEq < 3 || residual.nEquations != nEq
         || connectivity.nFaces != nFaces
         || storage->qf1.Size() < static_cast< std::size_t >( nFaces * nEq )
         || storage->qf2.Size() < static_cast< std::size_t >( nFaces * nEq )
         || storage->cachedBoundaryFaces != connectivity.nBoundaryFaces )
    {
        throw std::invalid_argument(
            "reconstructed HIP flux dimensions do not match state" );
    }
    if ( storage->cachedLeftCell != connectivity.leftCell
         || storage->cachedRightCell != connectivity.rightCell )
    {
        throw std::invalid_argument(
            "reconstructed HIP flux connectivity does not match state" );
    }
    if ( hostFlux != nullptr
         && ( hostFlux->nFaces != nFaces || hostFlux->nEquations != nEq
              || hostFlux->values == nullptr ) )
    {
        throw std::invalid_argument( "invalid reconstructed HIP host flux" );
    }

    const int totalResidual = nCells * nEq;
    deviceResidual_.Resize( totalResidual );
    if ( residual.values != nullptr )
    {
        deviceResidual_.CopyFromHost(
            residual.values, totalResidual );
    }
    else
    {
        for ( int equation = 0; equation < nEq; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                deviceResidual_.Data() + equation * nCells,
                residual.components[ equation ], nCells * sizeof( Real ),
                AccelCopyKind::HostToDevice );
        }
    }
    if ( hostFlux != nullptr ) deviceFlux_.Resize( nFaces * nEq );

    const unsigned char * deviceBoundaryMask = nullptr;
    if ( connectivity.boundaryMask != nullptr )
    {
        boundaryMask_.CopyFromHost(
            connectivity.boundaryMask, nFaces );
        deviceBoundaryMask = boundaryMask_.Data();
    }
    const int blockSize = 256;
    hipLaunchKernelGGL(
        CalcAndScatterPrimitiveEulerKernel,
        dim3( ( nFaces + blockSize - 1 ) / blockSize ),
        dim3( blockSize ), 0, 0,
        storage->qf1.Data(), storage->qf2.Data(),
        storage->xNormal.Data(), storage->yNormal.Data(),
        storage->zNormal.Data(), nullptr, storage->faceArea.Data(),
        storage->leftCell.Data(), storage->rightCell.Data(),
        deviceBoundaryMask, connectivity.nBoundaryFaces, nFaces, nCells,
        nEq, gamma, scheme,
        hostFlux != nullptr ? deviceFlux_.Data() : nullptr,
        deviceResidual_.Data() );
    CheckHip( hipGetLastError(),
              "reconstructed primitive flux kernel launch" );
    CheckHip( hipDeviceSynchronize(),
              "reconstructed primitive flux kernel synchronize" );
    if ( residual.values != nullptr )
    {
        deviceResidual_.CopyToHost( residual.values, totalResidual );
    }
    else
    {
        for ( int equation = 0; equation < nEq; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                residual.components[ equation ],
                deviceResidual_.Data() + equation * nCells,
                nCells * sizeof( Real ), AccelCopyKind::DeviceToHost );
        }
    }
    if ( hostFlux != nullptr )
    {
        deviceFlux_.CopyToHost( hostFlux->values, nFaces * nEq );
        deviceFluxFaces_ = nFaces;
        deviceFluxEquations_ = nEq;
    }
}

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

    const bool connectivityCacheMatches =
        geometryCacheValid_
        && cachedFaces_ == nFaces
        && cachedBoundaryFaces_ == connectivity.nBoundaryFaces
        && cachedLeftCell_ == connectivity.leftCell
        && cachedRightCell_ == connectivity.rightCell
        && cachedBoundaryMask_ == connectivity.boundaryMask;
    if ( ! connectivityCacheMatches )
    {
        ScopedStageTimer timer( "geometry_connectivity_H2D" );
        leftCell_.CopyFromHost( connectivity.leftCell, nFaces );
        rightCell_.CopyFromHost( connectivity.rightCell, nFaces );
        if ( connectivity.boundaryMask )
            boundaryMask_.CopyFromHost( connectivity.boundaryMask, nFaces );
        cachedLeftCell_ = connectivity.leftCell;
        cachedRightCell_ = connectivity.rightCell;
        cachedBoundaryMask_ = connectivity.boundaryMask;
        cachedFaces_ = nFaces;
        cachedBoundaryFaces_ = connectivity.nBoundaryFaces;
        geometryCacheValid_ = true;
    }
    deviceResidual_.Resize( totalResidual );
    { ScopedStageTimer timer( "H2D" );
        deviceResidual_.CopyFromHost( residual.values, totalResidual );
    }

    const int blockSize = 256;
    const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
    { ScopedStageTimer timer( "HIP kernel" );
    hipLaunchKernelGGL(
        ScatterFaceFluxEulerKernel,
        dim3( gridSize ),
        dim3( blockSize ),
        0,
        0,
        deviceFlux_.Data(),
        leftCell_.Data(),
        rightCell_.Data(),
        connectivity.boundaryMask ? boundaryMask_.Data() : nullptr,
        connectivity.nBoundaryFaces,
        nFaces,
        nCells,
        nEq,
        deviceResidual_.Data() );
    CheckHip( hipGetLastError(), "ScatterFaceFluxEulerKernel launch" );
    CheckHip(
        hipDeviceSynchronize(),
        "ScatterFaceFluxEulerKernel synchronize" );
    }
    { ScopedStageTimer timer( "D2H" );
        deviceResidual_.CopyToHost( residual.values, totalResidual );
    }
}

void HipFluxBackend::CalcAndAddPrimitiveFaceFlux(
    const PrimitiveFaceStateView & state,
    const FaceConnectivityView & connectivity,
    ResidualView & residual,
    int scheme,
    FaceFluxView * hostFlux,
    const void * cacheKey )
{
    const bool componentInput =
        state.primitiveLeft == nullptr && state.primitiveRight == nullptr
        && state.primitiveLeftComponents[ 0 ] != nullptr
        && state.primitiveRightComponents[ 0 ] != nullptr;
    if ( state.nFaces <= 0 || state.nEquations < 3
         || ( !componentInput
              && ( state.primitiveLeft == nullptr
                   || state.primitiveRight == nullptr ) )
         || state.gamma <= 1.0
         || state.nFaces != connectivity.nFaces
         || residual.nCells <= 0
         || residual.nEquations != state.nEquations
         || ( residual.values == nullptr
              && residual.components[ 0 ] == nullptr ) )
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

    if ( componentInput )
    {
        ScopedStageTimer timer( "H2D" );
        qLeft_.Resize( totalFlux );
        qRight_.Resize( totalFlux );
        for ( int equation = 0; equation < nEq; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                qLeft_.Data() + equation * nFaces,
                state.primitiveLeftComponents[ equation ],
                nFaces * sizeof( Real ), AccelCopyKind::HostToDevice );
            AccelRuntime::Instance().Backend().Copy(
                qRight_.Data() + equation * nFaces,
                state.primitiveRightComponents[ equation ],
                nFaces * sizeof( Real ), AccelCopyKind::HostToDevice );
        }
    }
    else
    {
        ScopedStageTimer timer( "H2D" );
        qLeft_.CopyFromHost( state.primitiveLeft, totalFlux );
        qRight_.CopyFromHost( state.primitiveRight, totalFlux );
    }

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
        ScopedStageTimer timer( "geometry_connectivity_H2D" );
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
    deviceResidual_.Resize( totalResidual );
    { ScopedStageTimer timer( "H2D" );
    if ( residual.values != nullptr )
    {
        deviceResidual_.CopyFromHost( residual.values, totalResidual );
    }
    else
    {
        for ( int equation = 0; equation < nEq; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                deviceResidual_.Data() + equation * nCells,
                residual.components[ equation ],
                nCells * sizeof( Real ), AccelCopyKind::HostToDevice );
        }
    }
    if ( hostFlux != nullptr ) deviceFlux_.Resize( totalFlux );
    }

    const int blockSize = 256;
    const int gridSize = ( nFaces + blockSize - 1 ) / blockSize;
    { ScopedStageTimer kernelTimer( "HIP kernel" );
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
    }
    { ScopedStageTimer timer( "D2H" );
    if ( residual.values != nullptr )
    {
        deviceResidual_.CopyToHost( residual.values, totalResidual );
    }
    else
    {
        for ( int equation = 0; equation < nEq; ++ equation )
        {
            AccelRuntime::Instance().Backend().Copy(
                residual.components[ equation ],
                deviceResidual_.Data() + equation * nCells,
                nCells * sizeof( Real ), AccelCopyKind::DeviceToHost );
        }
    }
    }
    if ( hostFlux != nullptr )
    {
        ScopedStageTimer timer( "D2H" );
        deviceFlux_.CopyToHost( hostFlux->values, totalFlux );
        deviceFluxFaces_ = nFaces;
        deviceFluxEquations_ = nEq;
    }
}

}
