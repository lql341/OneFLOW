/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.
\---------------------------------------------------------------------------*/

#pragma once

#include "FluxBackend.h"
#include "DeviceBuffer.h"

namespace ONEFLOW
{

struct PrimitiveFaceStateView;

class HipFluxBackend final : public FluxBackend
{
public:
    static HipFluxBackend & Shared();
    static void ReleaseShared();

    void CalcInvFlux(
        const FaceStateView & state,
        FaceFluxView & flux,
        int scheme ) override;

    void AddFaceFlux(
        const FaceFluxView & flux,
        const FaceConnectivityView & connectivity,
        ResidualView & residual ) override;

    // Add the flux produced by the immediately preceding CalcInvFlux call
    // without uploading the host flux array again. This is the production
    // seam used by the 3D main solver HIP batch path.
    void AddCurrentFaceFlux(
        const FaceConnectivityView & connectivity,
        ResidualView & residual );

    // Fused production path for primitive face data. The kernel converts
    // primitive states, evaluates the numerical flux, and scatters directly
    // into the device residual. hostFlux may be null in NoTrace mode.
    void CalcAndAddPrimitiveFaceFlux(
        const PrimitiveFaceStateView & state,
        const FaceConnectivityView & connectivity,
        ResidualView & residual,
        int scheme,
        FaceFluxView * hostFlux = nullptr,
        const void * cacheKey = nullptr );

private:
    DeviceBuffer< Real > qLeft_;
    DeviceBuffer< Real > qRight_;
    DeviceBuffer< Real > xNormal_;
    DeviceBuffer< Real > yNormal_;
    DeviceBuffer< Real > zNormal_;
    DeviceBuffer< Real > meshVelocityNormal_;
    DeviceBuffer< Real > area_;
    DeviceBuffer< Real > deviceFlux_;
    DeviceBuffer< int > leftCell_;
    DeviceBuffer< int > rightCell_;
    DeviceBuffer< unsigned char > boundaryMask_;
    DeviceBuffer< Real > deviceResidual_;
    int deviceFluxFaces_ = 0;
    int deviceFluxEquations_ = 0;
    const Real * cachedXNormal_ = nullptr;
    const Real * cachedYNormal_ = nullptr;
    const Real * cachedZNormal_ = nullptr;
    const Real * cachedMeshVelocityNormal_ = nullptr;
    const Real * cachedArea_ = nullptr;
    const int * cachedLeftCell_ = nullptr;
    const int * cachedRightCell_ = nullptr;
    const unsigned char * cachedBoundaryMask_ = nullptr;
    const void * cachedOwner_ = nullptr;
    int cachedFaces_ = 0;
    int cachedBoundaryFaces_ = -1;
    bool geometryCacheValid_ = false;
};

}
