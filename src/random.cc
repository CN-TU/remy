#include <ctime>
#include <sys/types.h>
#include <unistd.h>

#include "random.hh"

PRNG & global_PRNG( void )
{
  static PRNG generator( time( NULL ) ^ getpid() );
  return generator;
}

PRNG* global_PRNG( const size_t thread_id )
{
  PRNG* generator = new PRNG( time( NULL ) ^ getpid() ^ (long) thread_id );
  return generator;
}
