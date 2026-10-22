/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.
\*---------------------------------------------------------------------------*/

#pragma once

#include "AccelRuntime.h"

#include <cstdlib>
#include <cstring>

BeginNameSpace( ONEFLOW )

// Production HIP follows the runtime backend selection. Legacy per-stage
// variables remain overrides: unset selects automatically, "0" disables the
// stage, and "1" requests it subject to capability checks.
inline bool HipProductionBackendSelected()
{
#ifdef ONEFLOW_ENABLE_HIP
    AccelRuntime & runtime = AccelRuntime::Instance();
    return runtime.IsInitialized()
        && runtime.Backend().Kind() == AccelBackendKind::HIP;
#else
    return false;
#endif
}

inline bool HipProductionFeatureExplicitlyEnabled( const char * name )
{
    const char * value = std::getenv( name );
    return value != nullptr && value[ 0 ] == '1';
}

inline bool HipProductionFeatureEnabled( const char * legacyName )
{
    const char * value = std::getenv( legacyName );
    if ( value != nullptr ) return value[ 0 ] == '1';
    if ( ! HipProductionBackendSelected() ) return false;

    // Stages enabled by `ONEFLOW_ACCEL_BACKEND=HIP` alone.  Every one of them
    // has passed the same target-node gates (3-step 36-record trace against
    // the legacy oracle plus fixed-CFL 50-step stability, single- and
    // multi-zone, and the resident-state contract at 0 difference):
    //   * fused primitive flux/residual,
    //   * the state-owned interface/halo exchange,
    //   * device Green-Gauss gradient,
    //   * device reconstruction (cross-zone ghost gradient still travels
    //     through the host-staged exchange contract),
    //   * device primitive state update plus the resident residual chain,
    //   * device local boundary model,
    //   * resident local time-step.
    // The per-stage UseHip*/Supported guards still fall back to the host path
    // when the configuration is outside the validated scope (viscous,
    // limited, multi-grid, non-Lax-Friedrichs, more than one local grid, ...),
    // and an explicitly requested but unsupported stage still fails fast.
    return std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_BATCH" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_GRADIENT" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_INTERFACE" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_RECONSTRUCTION" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_STATE_UPDATE" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_BOUNDARY" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_TIME_STEP" ) == 0;
}

EndNameSpace
