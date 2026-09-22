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
#include "EulerDomain.h"

#include <cstdint>
#include <memory>
#include <unordered_map>

namespace ONEFLOW
{

struct PrimitiveFaceStateView;

// HIP-only storage owned either by a solver/grid Ns3DBackendState or by the
// compatibility fallback in HipFluxBackend. Keeping this type in the HIP
// header avoids leaking DeviceBuffer through backend-neutral contracts.
struct HipGradientStorage
{
    DeviceBuffer< Real > cellState;
    DeviceBuffer< Real > cellGradient;
    DeviceBuffer< Real > xFaceCenter;
    DeviceBuffer< Real > yFaceCenter;
    DeviceBuffer< Real > zFaceCenter;
    DeviceBuffer< Real > xNormal;
    DeviceBuffer< Real > yNormal;
    DeviceBuffer< Real > zNormal;
    DeviceBuffer< Real > faceArea;
    DeviceBuffer< Real > xCellCenter;
    DeviceBuffer< Real > yCellCenter;
    DeviceBuffer< Real > zCellCenter;
    DeviceBuffer< Real > cellVolume;
    DeviceBuffer< int > leftCell;
    DeviceBuffer< int > rightCell;
    const void * cachedOwnerToken = nullptr;
    std::uint64_t cachedTopologyGeneration = 0;
    const Real * cachedXFace = nullptr;
    const Real * cachedYFace = nullptr;
    const Real * cachedZFace = nullptr;
    const Real * cachedXNormal = nullptr;
    const Real * cachedYNormal = nullptr;
    const Real * cachedZNormal = nullptr;
    const Real * cachedFaceArea = nullptr;
    const Real * cachedXCell = nullptr;
    const Real * cachedYCell = nullptr;
    const Real * cachedZCell = nullptr;
    const Real * cachedVolume = nullptr;
    const int * cachedLeftCell = nullptr;
    const int * cachedRightCell = nullptr;
    int cachedCells = 0;
    int cachedGhostCells = -1;
    int cachedFaces = 0;
    int cachedBoundaryFaces = -1;
    bool geometryCacheValid = false;
    DeviceBuffer< Real > qf1;
    DeviceBuffer< Real > qf2;
    DeviceBuffer< unsigned char > boundaryMask;
    DeviceBuffer< ReconstructionBoundaryOperation > boundaryOperation;
    DeviceBuffer< Real > bcQ;
    int cachedEquations = 0;
    std::uint64_t cachedFieldGeneration = 0;
    bool boundaryMaskValid = false;
};

std::unique_ptr< Ns3DBackendState > CreateHipNs3DBackendState(
    int deviceId );

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

    void CalcGradient( const CellGradientView & view );

    // Reconstruct face primitive values into state-owned device buffers. The
    // host view supplies metadata and optional diagnostic destinations; the
    // numerical path consumes q/gradient/geometry already resident on device.
    void ReconstructFaceValues(
        const CellFaceReconstructionView & view,
        bool downloadToHost = false );

    // Consume the device qf1/qf2 produced by ReconstructFaceValues and scatter
    // the resulting flux directly into the device residual. hostFlux is only
    // populated for explicit trace/diagnostic requests.
    void CalcAndAddReconstructedFaceFlux(
        const FaceConnectivityView & connectivity,
        ResidualView & residual,
        int scheme,
        Real gamma,
        FaceFluxView * hostFlux = nullptr,
        const void * cacheKey = nullptr );

    void BindState( const void * cacheKey, Ns3DDeviceState & state );
    void UnbindState( const void * cacheKey ) noexcept;

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
    HipGradientStorage fallbackGradientStorage_;
    std::unordered_map< const void *, Ns3DDeviceState * > boundStates_;
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
