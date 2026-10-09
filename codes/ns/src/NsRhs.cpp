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

#include "NsRhs.h"
#include "UNsBcSolver.h"
#include "Zone.h"
#include "DataBase.h"
#include "Iteration.h"
#include "NsCom.h"
#include "UCom.h"
#include "UNsCom.h"
#include "UnsGrid.h"
#include "NsCom.h"
#include "NsIdx.h"
#include "UNsInvFlux.h"
#include "UNsVisFlux.h"
#include "UNsUnsteady.h"
#include "Ctrl.h"

#include <iostream>
#include <memory>


BeginNameSpace( ONEFLOW )

NsRhs::NsRhs()
{
    ;
}

NsRhs::~NsRhs()
{
    ;
}

void NsRhs::UpdateResiduals()
{
	NsCalcRHS();
}

void NsCalcBc()
{
	auto uNsBcSolver = std::make_unique<UNsBcSolver>();
	uNsBcSolver->Init();
	uNsBcSolver->CalcBc();
}

void NsCalcBcDebug( const std::string & title )
{
	std::cout << title << "\n";
	auto uNsBcSolver = std::make_unique<UNsBcSolver>();
	uNsBcSolver->Init();
	uNsBcSolver->CalcBc();

}

void NsCalcGamaT( int flag )
{
	UnsGrid * grid = Zone::GetUnsGrid();

	ug.Init();
	unsf.Init();
	ug.SetStEd(flag);

	if ( nscom.chemModel == 1 )
	{
	}
	else
	{
		Real oamw = one;
		Real * density = ( * unsf.q )[ IDX::IR ].data();
		Real * pressure = ( * unsf.q )[ IDX::IP ].data();
		Real * gama = ( * unsf.gama )[ 0 ].data();
		Real * temperature = ( * unsf.tempr )[ IDX::ITT ].data();
		for ( int cId = ug.ist; cId < ug.ied; ++ cId )
		{
			gama[ cId ] = nscom.gama_ref;
			temperature[ cId ] =
				pressure[ cId ] / ( nscom.statecoef * density[ cId ] * oamw );
		}
	}
}

void NsCalcRHS()
{
	NsCalcInvFlux();

	NsCalcVisFlux();

	NsCalcSrcFlux();
}


void NsCalcInvFlux()
{
    // Reuse the host-side limiter/face work buffers across RK stages.  The
    // fields are rebound in Init() for the active zone/grid, while qf1/qf2
    // and invflux retain their capacity when the topology is unchanged.
    static UNsInvFlux uNsInvFlux;
    uNsInvFlux.CalcFlux();
}

void NsCalcVisFlux()
{
	auto uNsVisFlux = std::make_unique<UNsVisFlux>();
	uNsVisFlux->CalcFlux();
}

void NsCalcSrcFlux()
{
	if (nscom.chemModel == 1)
	{
		NsCalcChemSrc();
	}

	NsCalcTurbEnergy();

	//dual time step source
	if ( ctrl.idualtime == 1 )
	{
		NsCalcDualTimeStepSrc();
	}
}

void NsCalcChemSrc()
{
	;
}

void NsCalcTurbEnergy()
{
	;
}

void NsCalcDualTimeStepSrc()
{
	auto unsUnsteady = std::make_unique<UNsUnsteady>();
	unsUnsteady->CalcDualTimeSrc();
}

EndNameSpace
