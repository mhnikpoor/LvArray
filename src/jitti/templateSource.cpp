#if !defined(JITTI_TEMPLATE_HEADER_FILE)
  #error JITTI_TEMPLATE_HEADER_FILE must be defined!
#endif

#if !defined(JITTI_TEMPLATE_FUNCTION)
  #error JITTI_TEMPLATE_FUNCTION must be defined!
#endif

#if !defined(JITTI_TEMPLATE_PARAMS)
  #error JITTI_TEMPLATE_PARAMS must be defined!
#endif

// Include the header the template is defined in
#include JITTI_TEMPLATE_HEADER_FILE

// Include the jitti header
#include "jitti.hpp"


auto instantiation = JITTI_TEMPLATE_FUNCTION< JITTI_TEMPLATE_PARAMS >;

jitti::SymbolTable exportedSymbols = {
  { std::string( STRINGIZE( JITTI_TEMPLATE_FUNCTION ) "< " STRINGIZE( JITTI_TEMPLATE_PARAMS ) " >" ),
     { reinterpret_cast< void * >( instantiation ), std::type_index( typeid( instantiation ) ) } }
};

extern "C"
{

jitti::SymbolTable const * getExportedSymbols()
{ 
  LVARRAY_LOG( "JITTI_TEMPLATE_PARAMS = " << STRINGIZE( JITTI_TEMPLATE_PARAMS ) );
  return &exportedSymbols;
}

} // extern "C"

