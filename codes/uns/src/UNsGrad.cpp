/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
    Copyright (C) 2017-2026 He Xin and the OneFLOW contributors.
-------------------------------------------------------------------------------
License
    This file is part of OneFLOW.

    OneFLOW is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OneFLOW is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OneFLOW.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "UNsGrad.h"
#include "UCom.h"
#include "NsCom.h"
#include "DataBase.h"
#include "FieldImp.h"
#include "FaceTopo.h"
#include "BcRecord.h"
#include "UnsGrid.h"
#include "Zone.h"
#include "FaceMesh.h"
#include "CellMesh.h"
#include "CellTopo.h"
#ifdef ONEFLOW_ENABLE_HIP
#include "HipFluxBackend.h"
#endif
#include <stdexcept>

BeginNameSpace( ONEFLOW )

UNsGrad uns_grad;
UTGrad ut_grad;

UNsGrad::UNsGrad()
{
    ;
}

UNsGrad::~UNsGrad()
{
    ;
}

void UNsGrad::Init()
{
    UnsGrid * grid = Zone::GetUnsGrid();

    name  = "q";
    namex = "dqdx";
    namey = "dqdy";
    namez = "dqdz";

    q    = GetFieldPointer< MRField > ( grid, name  );
    dqdx = GetFieldPointer< MRField > ( grid, namex );
    dqdy = GetFieldPointer< MRField > ( grid, namey );
    dqdz = GetFieldPointer< MRField > ( grid, namez );
    bdqdx = GetFieldPointer< MRField > ( grid, "bcdqdx" );
    bdqdy = GetFieldPointer< MRField > ( grid, "bcdqdy" );
    bdqdz = GetFieldPointer< MRField > ( grid, "bcdqdz" );

    this->nEqu = nscom.nTEqu;

    this->istore = 1;
}

void UNsGrad::CalcGradHip()
{
#ifdef ONEFLOW_ENABLE_HIP
    if ( q == nullptr || dqdx == nullptr || dqdy == nullptr || dqdz == nullptr )
    {
        throw std::runtime_error( "UNs HIP gradient fields are unavailable" );
    }
    UnsGrid * grid = Zone::GetUnsGrid();
    CellGradientView view;
    view.nCells = ug.nCells;
    view.nGhostCells = ug.nTCell - ug.nCells;
    view.nFaces = ug.nFaces;
    view.nBoundaryFaces = ug.nBFaces;
    view.nEquations = nEqu;
    view.xFace = ug.xfc->data();
    view.yFace = ug.yfc->data();
    view.zFace = ug.zfc->data();
    view.xNormal = ug.xfn->data();
    view.yNormal = ug.yfn->data();
    view.zNormal = ug.zfn->data();
    view.faceArea = ug.farea->data();
    view.xCell = ug.xcc->data();
    view.yCell = ug.ycc->data();
    view.zCell = ug.zcc->data();
    view.cellVolume = ug.cvol->data();
    view.leftCell = ug.lcf->data();
    view.rightCell = ug.rcf->data();
    view.cacheKey = grid;
    for ( int equation = 0; equation < nEqu; ++ equation )
    {
        view.q[ equation ] = ( * q )[ equation ].data();
        view.dqdx[ equation ] = ( * dqdx )[ equation ].data();
        view.dqdy[ equation ] = ( * dqdy )[ equation ].data();
        view.dqdz[ equation ] = ( * dqdz )[ equation ].data();
    }
    HipFluxBackend::Shared().CalcGradient( view );
    this->SwapBcGrad();
#else
    throw std::runtime_error(
        "UNs HIP gradient was called without HIP support" );
#endif
}

UTGrad::UTGrad()
{
    ;
}

UTGrad::~UTGrad()
{
    ;
}

void UTGrad::Init()
{
    UnsGrid * grid = Zone::GetUnsGrid();

    name  = "tempr";
    namex = "dtdx";
    namey = "dtdy";
    namez = "dtdz";

    q    = GetFieldPointer< MRField > ( grid, name  );
    dqdx = GetFieldPointer< MRField > ( grid, namex );
    dqdy = GetFieldPointer< MRField > ( grid, namey );
    dqdz = GetFieldPointer< MRField > ( grid, namez );

    this->nEqu = nscom.nTModel;

    this->istore = 0;
}

EndNameSpace
