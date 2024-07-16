#include "GAMER.h"



// declare as static so that other functions cannot invoke it directly and must use the function pointer
static void Flag_UserWorkBeforeFlag_Template( const int lv );

// to enable this feature, set this function pointer by a test problem initializer
void (*Flag_UserWorkBeforeFlag_Ptr)( const int lv ) = NULL;



//-------------------------------------------------------------------------------------------------------
// Function    :  Flag_UserWorkBeforeFlag_Template
// Description :  Template if user-specified work before flagging the real patches
//
// Note        :  1. Invoked by Flag_Real() using the function pointer "Flag_UserWorkBeforeFlag_Ptr"
//
// Parameter   :  lv : Target refinement level
//
// Return      :  none
//-------------------------------------------------------------------------------------------------------
void Flag_UserWorkBeforeFlag_Template( const int lv )
{
} // FUNCTION : Flag_UserWorkBeforeFlag_Template
