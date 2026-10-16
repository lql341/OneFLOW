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
#include "InterfaceFieldProperty.h"
#include "FieldBase.h"
#include "Fatal.h"
#include "Grid.h"
#include "GridState.h"
#include "DataBase.h"
#include "DataStorage.h"
#include "UnsGrid.h"
#include "Zone.h"
#include "ZoneState.h"
#include "InterFace.h"
#include "FaceTopo.h"
#include "BoundaryExchangeContract.h"

#include <vector>

BeginNameSpace( ONEFLOW )

void InterfaceFieldProperty::AddField(
    const std::string & fieldName,
    int nEqu )
{
    this->fieldDefinitions.AddField(
        fieldName,
        nEqu );
}

bool InterfaceFieldProperty::HasField(
    const std::string & fieldName ) const
{
    return this->fieldDefinitions.HasField(
        fieldName );
}

int InterfaceFieldProperty::GetNEqu(
    const std::string & fieldName ) const
{
    return this->fieldDefinitions.GetNEqu(
        fieldName );
}

bool InterfaceFieldProperty::Empty() const
{
    return this->fieldDefinitions.Empty();
}

const FieldDefinitionTable::Data &
InterfaceFieldProperty::GetData() const
{
    return this->fieldDefinitions.GetData();
}

void InterfaceFieldProperty::Dump(
    std::ostream & output ) const
{
    this->fieldDefinitions.Dump( output );
}

void UploadInterfaceValue( UnsGrid * grid, MRField * field2D, const std::string & name, int nEqu )
{
    InterFace * interFace = grid->interFace.get();
    if ( ! ONEFLOW::IsValid( interFace ) ) return;

    int nIFaces = interFace->nIFaces;

    if ( field2D == nullptr ) return;

    for ( int ghostId = MAX_GHOST_LEVELS - 1; ghostId >= 0; -- ghostId )
    {
        DataStorage * dataSend = &interFace->GetSendStorage( ghostId );

        MRField * fieldStorage =
            ONEFLOW::GetFieldPointer< MRField >( dataSend, name );

        if ( fieldStorage == nullptr )
        {
            Fatal(
                "Interface send field is not allocated: "
                + name );
        }

        BoundaryEndpointMap endpoint;
        endpoint.cellCount =
            static_cast< int >( ( * field2D )[ 0 ].size() );
        endpoint.cellIds.resize( nIFaces );
        for ( int iFace = 0; iFace < nIFaces; ++ iFace )
        {
            int iCell;
            grid->GetFaceTopo().GetSId(iFace, ghostId + 1, iCell);
            endpoint.cellIds[ iFace ] = iCell;
        }

        std::vector< const Real * > components;
        components.reserve( nEqu );
        for ( int iEqu = 0; iEqu < nEqu; ++ iEqu )
        {
            components.push_back( ( * field2D )[ iEqu ].data() );
        }
        std::vector< Real > packed;
        PackBoundaryHost( endpoint, nEqu, components, packed );
        for ( int iEqu = 0; iEqu < nEqu; ++ iEqu )
        {
            for ( int iFace = 0; iFace < nIFaces; ++ iFace )
            {
                ( * fieldStorage )[ iEqu ][ iFace ] =
                    packed[ iEqu * nIFaces + iFace ];
            }
        }
    }
}

void DownloadInterfaceValue( UnsGrid * grid, MRField * field2D, const std::string & name, int nEqu )
{
    InterFace * interFace = grid->interFace.get();
    if ( ! ONEFLOW::IsValid( interFace ) ) return;

    if ( field2D == nullptr ) return;

    for ( int ghostId = MAX_GHOST_LEVELS - 1; ghostId >= 0; -- ghostId )
    {
        DataStorage * dataRecv = &interFace->GetRecvStorage( ghostId );

        MRField * fieldStorage =
            ONEFLOW::GetFieldPointer< MRField >( dataRecv, name );

        if ( fieldStorage == nullptr )
        {
            Fatal(
                "Interface recv field is not allocated: "
                + name );
        }

        const int nIFaces = interFace->nIFaces;
        std::vector< Real > packed(
            static_cast< std::size_t >( nEqu ) * nIFaces );
        for ( int iEqu = 0; iEqu < nEqu; ++ iEqu )
        {
            for ( int iFace = 0; iFace < nIFaces; ++ iFace )
            {
                packed[ iEqu * nIFaces + iFace ] =
                    ( * fieldStorage )[ iEqu ][ iFace ];
            }
        }

        BoundaryEndpointMap endpoint;
        endpoint.cellCount =
            static_cast< int >( ( * field2D )[ 0 ].size() );
        endpoint.cellIds.resize( nIFaces );
        for ( int iFace = 0; iFace < nIFaces; ++ iFace )
        {
            int iCell;
            grid->GetFaceTopo().GetTId(iFace, ghostId + 1, iCell);
            endpoint.cellIds[ iFace ] = iCell;
        }

        std::vector< Real * > components;
        components.reserve( nEqu );
        for ( int iEqu = 0; iEqu < nEqu; ++ iEqu )
        {
            components.push_back( ( * field2D )[ iEqu ].data() );
        }
        UnpackBoundaryHost(
            endpoint,
            nEqu,
            packed,
            components );
    }
}

void UploadOversetValue( UnsGrid * grid, MRField * field2D, const std::string & name, int nEqu )
{
    // Reserved: overset interface transfer is not implemented yet.
    (void) grid;
    (void) field2D;
    (void) name;
    (void) nEqu;
}

void DownloadOversetValue( UnsGrid * grid, MRField * field2D, const std::string & name, int nEqu )
{
    // Reserved: overset interface transfer is not implemented yet.
    (void) grid;
    (void) field2D;
    (void) name;
    (void) nEqu;
}

EndNameSpace
