/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.

    OneFLOW is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
\*---------------------------------------------------------------------------*/

#pragma once

// Boundary-condition arithmetic shared by the host oracle and the device
// kernels.
//
// Why this header exists
//   The symmetry ghost state is written by two implementations: the host chain
//   (NsBcSolver::SymmetryBc -> UNsBcSolver::UpdateBc) and the device kernel
//   (UpdateBoundaryPrimitiveKernel in HipFluxBackend.hip).  They disagreed.
//   The host copies the owner primitive and then *overwrites the three velocity
//   components with the reflection about the face*, so the symmetry ghost
//   carries a cancelled normal velocity.  The device branch copied the owner
//   unchanged and used the reflection only for bcQ, which left the normal
//   velocity in the ghost and changed the symmetry-face inviscid flux.  That is
//   the B.4.1 boundary residual.
//
//   Both callers now evaluate this one implementation, so the formulas cannot
//   drift again.  The host sources stay the reference; nothing here changes
//   their arithmetic.
//
// Host/device callable marker
//   Without the attribute the device compiler rejects a call from a __global__
//   function ("reference to __host__ function ... in __global__ function").
//   The guard is written on __HIPCC__/__CUDACC__ so the header also works in a
//   CUDA translation unit; on a plain host compiler the macro is empty and the
//   functions stay ordinary inline functions.

#include "HXTypeBasic.h"

#if defined( __HIPCC__ ) || defined( __CUDACC__ )
#define ONEFLOW_BC_ARITHMETIC_HD __host__ __device__
#else
#define ONEFLOW_BC_ARITHMETIC_HD
#endif

BeginNameSpace( ONEFLOW )

// Velocity relative to the face along its outward normal.
// Mirrors NsInvFluxPolicy.cpp:RelativeFaceNormalVelocity.
// @doc-algorithm boundary-velocity
ONEFLOW_BC_ARITHMETIC_HD inline Real RelativeFaceNormalVelocityHd(
    Real xfn, Real yfn, Real zfn, Real vfn,
    Real u, Real v, Real w )
{
    return xfn * u + yfn * v + zfn * w - vfn;
}

// Specular reflection of a velocity about a (possibly moving) face.
// Mirrors NsBcPolicy.cpp:ReflectVelocityRelativeToMovingFace.
// @doc-algorithm boundary-velocity
// @doc-stage symmetry-reflect
ONEFLOW_BC_ARITHMETIC_HD inline void ReflectVelocityRelativeToMovingFaceHd(
    Real u, Real v, Real w,
    Real xfn, Real yfn, Real zfn, Real vfn,
    Real & uReflected, Real & vReflected, Real & wReflected )
{
    const Real vnRel = RelativeFaceNormalVelocityHd(
        xfn, yfn, zfn, vfn, u, v, w );
    uReflected = u - 2.0 * xfn * vnRel;
    vReflected = v - 2.0 * yfn * vnRel;
    wReflected = w - 2.0 * zfn * vnRel;
}

// Arithmetic mean of two scalars, used for the face-averaged boundary state.
// Mirrors NsBcPolicy.cpp:ArithmeticAverage.
// @doc-algorithm boundary-velocity
// @doc-stage face-average
ONEFLOW_BC_ARITHMETIC_HD inline Real ArithmeticAverageHd( Real a, Real b )
{
    return 0.5 * ( a + b );
}

EndNameSpace
