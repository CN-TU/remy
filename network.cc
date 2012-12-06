#include <assert.h>
#include <stdlib.h>
#include <utility>

#include "network.hh"

Network::Network( const double s_rate )
  : _buffer(),
    _egress_process( s_rate )
{
}

void Network::accept( Packet && p ) noexcept
{
  _buffer.push( std::move( p ) );
}

void Network::tick( Receiver & rec, const int tickno )
{
  const int num = _egress_process.tick();

  for ( int i = 0; i < num; i++ ) {
    if ( _buffer.empty() ) {
      break;
    }

    rec.accept( std::move( _buffer.front() ), tickno );
    _buffer.pop();
  }
}
