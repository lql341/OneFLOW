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
    return HipProductionBackendSelected();
}

EndNameSpace
