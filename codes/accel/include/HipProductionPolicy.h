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

    // Stages re-opened by `ONEFLOW_ACCEL_BACKEND=HIP` alone, each with a
    // target-node trace gate behind it:
    //   * fused primitive flux/residual (validated single- and multi-zone),
    //   * the state-owned interface/halo exchange, whose device q ghost
    //     region is now bound by ResidentStateContract (device/host ghost
    //     equality verified at 0 difference on the 8-zone fixture),
    //   * device Green-Gauss gradient, which consumes that resident q.
    // Everything else stays explicit until it has the same evidence:
    // reconstruction, state update, boundary and resident timestep.
    return std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_BATCH" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_GRADIENT" ) == 0
        || std::strcmp( legacyName, "ONEFLOW_ENABLE_UNS_HIP_INTERFACE" ) == 0;
}

EndNameSpace
