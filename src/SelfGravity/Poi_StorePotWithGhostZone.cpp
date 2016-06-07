#include "Copyright.h"
#include "GAMER.h"

#if ( defined GRAVITY  &&  defined STORE_POT_GHOST )




//-------------------------------------------------------------------------------------------------------
// Function    :  Poi_StorePotWithGhostZone
// Description :  Fill up the potential array with ghost zones, "pot_ext", for each target patch
//
// Note        :  1. Called by Gra_AdvancedDt after the base-level FFT solver, EvolveLevel after grid
//                   refinement, and Main after the base-level FFT when GAMER_DEBUG is on
//                2. For potential at lv>0, the "pot_ext" array is filled by "Poi_Close" directly
//                   (except after grid refinement). Therefore, one does NOT need to call this function
//                   for that.
//                3. After grid refinement, only newly-allocated patches need to get pot_ext
//                   --> We set pot_ext[0][0][0] == POT_EXT_NEED_INIT for newly-allocated patches
//                       so as to distinguish them from other existing patches
//
// Parameter   :  lv       : Targeted refinement level
//                PotSg    : Target potential sandglass
//                AllPatch : true  --> work on all patches at lv
//                                     (used after the root-level FFT solver)
//                           false --> only work on patches with pot_ext[0][0][0] == POT_EXT_NEED_INIT
//                                     (used after grid refinement)
//-------------------------------------------------------------------------------------------------------
void Poi_StorePotWithGhostZone( const int lv, const int PotSg, const bool AllPatch )
{

   const OptFluBC_t *FluBC_None = NULL;
   const bool IntPhase_No       = false;
   const bool GetTotDens_No     = false;

   const double PrepPotTime = amr->PotSgTime[lv][PotSg];
   const int    PotGhost    = GRA_GHOST_SIZE;
   const int    PotSize     = PS1 + 2*PotGhost;
   const int    PotSizeCube = CUBE(PotSize);

   real *Pot = new real [ 8*PotSizeCube ];   // 8: number of patches per patch group

   for (int PID0=0; PID0<amr->NPatchComma[lv][1]; PID0+=8)
   {
      if ( AllPatch  ||  amr->patch[PotSg][lv][PID0]->pot_ext[0][0][0] == POT_EXT_NEED_INIT )
      {
         Prepare_PatchData( lv, PrepPotTime, Pot, PotGhost, 1, &PID0, _POTE, OPT__REF_POT_INT_SCHEME,
                            UNIT_PATCH, NSIDE_26, IntPhase_No, FluBC_None, OPT__BC_POT, GetTotDens_No );

         for (int PID=PID0, P=0; PID<PID0+8; PID++, P++)
            memcpy( amr->patch[PotSg][lv][PID]->pot_ext, Pot+P*PotSizeCube, PotSizeCube*sizeof(real) );
      }
   }

} // FUNCTION : Poi_StorePotWithGhostZone



#endif // #if ( defined GRAVITY  &&  defined STORE_POT_GHOST )