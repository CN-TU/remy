#include "window-sender.hh"

WindowSender::WindowSender( const unsigned int s_id,
			    const unsigned int s_flow_id,
			    const unsigned int s_window,
			    const unsigned int tickno )
  : _id( s_id ),
    _flow_id( s_flow_id ),
    _window( s_window ),
    _packets_sent( 0 ),
    _packets_received( 0 ),
    _creation_tick( tickno ),
    _total_delay( 0 )
{
}

void WindowSender::tick( Network & net, Receiver & rec, const unsigned int tickno )
{
  /* Receive feedback */
  const std::vector< Packet > packets = rec.collect( _id );

  for ( auto &x : packets ) {
    _packets_received++;
    _total_delay += x.tick_received - x.tick_sent;
  }

  /* Send */
  assert( _packets_sent >= _packets_received );

  while ( _packets_sent < _packets_received + _window ) {
    net.accept( Packet( _id, _flow_id, _packets_sent++, tickno ) );
  }
}

void WindowSender::print_stats( const unsigned int tickno ) const
{
  const unsigned int age = tickno - _creation_tick;

  printf( "%d tick %d: avg throughput = %.4f, avg delay = %.4f\n",
	  _id,
	  tickno,
	  double( _packets_received ) / double( age ),
	  double( _total_delay ) / double( _packets_received ) );
}
