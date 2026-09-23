/*---------------------------------------------------------------------------*\
    OneFLOW - LargeScale Multiphysics Scientific Simulation Environment
-------------------------------------------------------------------------------
    Opt-in stage timing for accelerator/dataflow investigations.
\*---------------------------------------------------------------------------*/

#include "StageProfiler.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

BeginNameSpace( ONEFLOW )

namespace
{

std::mutex & OutputMutex()
{
    static std::mutex mutex;
    return mutex;
}

struct CategoryTotal
{
    const char * name;
    double elapsedMs;
};

std::array< CategoryTotal, 32 > & Totals()
{
    static std::array< CategoryTotal, 32 > totals = {{
        { "initialization", 0.0 },
        { "gradient", 0.0 },
        { "limiter", 0.0 },
        { "reconstruction", 0.0 },
        { "boundary_reconstruction", 0.0 },
        { "host_pack", 0.0 },
        { "flux_backend", 0.0 },
        { "geometry_connectivity_H2D", 0.0 },
        { "H2D", 0.0 },
        { "HIP kernel", 0.0 },
        { "D2H", 0.0 },
        { "viscous_turbulence", 0.0 },
        { "rk_update", 0.0 },
        { "residual_state_update", 0.0 },
        { "mpi_interface", 0.0 },
        { "output", 0.0 },
        { "load_q", 0.0 },
        { "calc_time_step", 0.0 },
        { "load_residuals", 0.0 },
        { "update_residuals", 0.0 },
        { "calc_lhs", 0.0 },
        { "update_flowfield", 0.0 },
        { "calc_boundary", 0.0 },
        { "boundary_gamma_inner", 0.0 },
        { "boundary_viscosity_inner", 0.0 },
        { "boundary_bc", 0.0 },
        { "boundary_gamma_ghost", 0.0 },
        { "boundary_viscosity_ghost", 0.0 },
        { "boundary_set_id", 0.0 },
        { "boundary_prepare_data", 0.0 },
        { "boundary_calc_face_bc", 0.0 },
        { "boundary_update_bc", 0.0 },
    }};
    return totals;
}

std::string RankAwarePath( const char * configured )
{
    std::string path( configured == nullptr ? "" : configured );
    const std::string token( "%r" );
    const std::size_t tokenPos = path.find( token );
    if ( tokenPos == std::string::npos ) return path;

    const char * rankNames[] = {
        "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "SLURM_PROCID" };
    const char * rank = "0";
    for ( const char * name : rankNames )
    {
        const char * value = std::getenv( name );
        if ( value != nullptr && value[ 0 ] != '\0' )
        {
            rank = value;
            break;
        }
    }
    path.replace( tokenPos, token.size(), rank );
    return path;
}

}

bool StageProfiler::Enabled()
{
    const char * path = std::getenv( "ONEFLOW_STAGE_BREAKDOWN_FILE" );
    return path != nullptr && path[ 0 ] != '\0';
}

void StageProfiler::Record( const char * category, double elapsedMs ) noexcept
{
    if ( category == nullptr || category[ 0 ] == '\0' || elapsedMs < 0.0 )
        return;

    std::lock_guard< std::mutex > lock( OutputMutex() );
    for ( CategoryTotal & total : Totals() )
    {
        if ( std::strcmp( total.name, category ) == 0 )
        {
            total.elapsedMs += elapsedMs;
            return;
        }
    }
}

void StageProfiler::Flush()
{
    const char * configured = std::getenv( "ONEFLOW_STAGE_BREAKDOWN_FILE" );
    if ( configured == nullptr || configured[ 0 ] == '\0' ) return;

    const std::string path = RankAwarePath( configured );
    if ( path.empty() ) return;

    std::lock_guard< std::mutex > lock( OutputMutex() );
    std::ofstream output( path, std::ios::out | std::ios::trunc );
    if ( ! output )
    {
        throw std::runtime_error(
            "cannot open ONEFLOW_STAGE_BREAKDOWN_FILE: " + path );
    }
    output << "category\telapsed_ms\n";
    for ( const CategoryTotal & total : Totals() )
    {
        output << total.name << "\t" << total.elapsedMs << "\n";
    }
    if ( ! output )
    {
        throw std::runtime_error(
            "failed while writing ONEFLOW_STAGE_BREAKDOWN_FILE: " + path );
    }
}

const char * StageProfiler::CategoryForTask( const std::string & taskName )
{
    if ( taskName == "LOAD_Q" ) return "load_q";
    if ( taskName == "CALC_TIME_STEP" ) return "calc_time_step";
    if ( taskName == "LOAD_RESIDUALS" ) return "load_residuals";
    if ( taskName == "UPDATE_RESIDUALS" ) return "update_residuals";
    if ( taskName == "CALC_LHS" ) return "calc_lhs";
    if ( taskName == "UPDATE_FLOWFIELD" ) return "update_flowfield";
    if ( taskName == "CALC_BOUNDARY" ) return "calc_boundary";
    if ( taskName.find( "INIT" ) != std::string::npos
         || taskName.find( "READ_RESTART" ) != std::string::npos )
    {
        return "initialization";
    }
    if ( taskName.find( "INTERFACE" ) != std::string::npos
         || taskName.find( "EXCHANGE" ) != std::string::npos
         || taskName.find( "UPLOAD_INTERFACE" ) != std::string::npos
         || taskName.find( "DOWNLOAD_INTERFACE" ) != std::string::npos )
    {
        return "mpi_interface";
    }
    if ( taskName.find( "DUMP" ) != std::string::npos
         || taskName.find( "WRITE" ) != std::string::npos
         || taskName.find( "VISUALIZATION" ) != std::string::npos )
    {
        return "output";
    }
    return "";
}

ScopedStageTimer::ScopedStageTimer( const char * category )
    : category_( category ), enabled_( StageProfiler::Enabled() )
{
    if ( enabled_ )
    {
        start_ = std::chrono::steady_clock::now();
    }
}

ScopedStageTimer::~ScopedStageTimer() noexcept
{
    if ( ! enabled_ ) return;
    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration< double, std::milli > elapsed = end - start_;
    StageProfiler::Record( category_, elapsed.count() );
}

EndNameSpace
