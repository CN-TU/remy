#include <boost/functional/hash.hpp>
#include <vector>
#include <cassert>

#include "memory.hh"

using namespace std;

static const double alpha = 1.0 / 8.0;

static const double slow_alpha = 1.0 / 256.0;

void Memory::lost(const int lost) {
  printf("lost: %d\n", lost);
  for (auto i=0; i<lost; i++) {
    _loss = (1 - alpha) * _loss + alpha * 1;
  }
}

void Memory::packets_received( const vector< Packet > & packets, const unsigned int flow_id,
  const int largest_ack )
{
  for ( const auto &x : packets ) {
    if ( x.flow_id != flow_id ) {
      continue;
    }
    const double rtt = x.tick_received - x.tick_sent;
    int pkt_outstanding = 1;
    if ( x.seq_num > largest_ack ) {
      pkt_outstanding = x.seq_num - largest_ack;
    }
    if ( _last_tick_sent == 0 || _last_tick_received == 0 ) {
      _last_tick_sent = x.tick_sent;
      _last_tick_received = x.tick_received;
      _min_rtt = rtt;
    } else {
      _send = (x.tick_sent - _last_tick_sent);
      _rec = (x.tick_received - _last_tick_received);
      _rec_send_ewma = (1 - alpha) * _rec_send_ewma + alpha * (x.tick_sent - _last_tick_sent);
      _rec_rec_ewma = (1 - alpha) * _rec_rec_ewma + alpha * (x.tick_received - _last_tick_received);
      _slow_rec_rec_ewma = (1 - slow_alpha) * _slow_rec_rec_ewma + slow_alpha * (x.tick_received - _last_tick_received);

      _last_tick_sent = x.tick_sent;
      _last_tick_received = x.tick_received;

      _min_rtt = min( _min_rtt, rtt );
      _rtt_ratio = double( rtt ) / double( _min_rtt );
      _rtt_diff = rtt - _min_rtt;
      assert( _rtt_ratio >= 1.0 );
      assert( _rtt_diff >= 0 );
      _queueing_delay = _rec_rec_ewma * pkt_outstanding;
      _loss = (1 - alpha) * _loss + alpha * 0;
    }
  }
}

string Memory::str( void ) const
{
  char tmp[ 256 ];
  snprintf( tmp, 256, "sewma=%f, rewma=%f, rttr=%f, slowrewma=%f, rttd=%f, qdelay=%f", _rec_send_ewma, _rec_rec_ewma, _rtt_ratio, _slow_rec_rec_ewma, _rtt_diff, _queueing_delay );
  return tmp;
}

string Memory::str( unsigned int num ) const
{
  char tmp[ 50 ];
  switch ( num ) {
    case 0:
      snprintf( tmp, 50, "sewma=%f ", _rec_send_ewma );
      break;
    case 1:
      snprintf( tmp, 50, "rewma=%f ", _rec_rec_ewma );
      break;
    case 2:
      snprintf( tmp, 50, "rttr=%f ", _rtt_ratio );
      break;
    case 3:
      snprintf( tmp, 50, "slowrewma=%f ", _slow_rec_rec_ewma );
      break;
    case 4:
      snprintf( tmp, 50, "rttd=%f ", _rtt_diff );
      break;
    case 5:
      snprintf( tmp, 50, "qdelay=%f ", _queueing_delay );
      break;
  }
  return tmp;
}

const Memory & MAX_MEMORY( void )
{
  static const Memory max_memory( { 163840, 163840, 163840, 163840, 163840, 163840 } );
  return max_memory;
}

RemyBuffers::Memory Memory::DNA( void ) const
{
  RemyBuffers::Memory ret;
  ret.set_rec_send_ewma( _rec_send_ewma );
  ret.set_rec_rec_ewma( _rec_rec_ewma );
  ret.set_rtt_ratio( _rtt_ratio );
  ret.set_slow_rec_rec_ewma( _slow_rec_rec_ewma );
  ret.set_rtt_diff( _rtt_diff );
  ret.set_queueing_delay( _queueing_delay );
  return ret;
}

/* If fields are missing in the DNA, we want to wildcard the resulting rule to match anything */
#define get_val_or_default( protobuf, field, limit ) \
  ( (protobuf).has_ ## field() ? (protobuf).field() : (limit) ? 0 : 163840 )

Memory::Memory( const bool is_lower_limit, const RemyBuffers::Memory & dna )
  : _rec_send_ewma( get_val_or_default( dna, rec_send_ewma, is_lower_limit ) ),
    _rec_rec_ewma( get_val_or_default( dna, rec_rec_ewma, is_lower_limit ) ),
    _rtt_ratio( get_val_or_default( dna, rtt_ratio, is_lower_limit ) ),
    _slow_rec_rec_ewma( get_val_or_default( dna, slow_rec_rec_ewma, is_lower_limit ) ),
    _rtt_diff( get_val_or_default( dna, rtt_diff, is_lower_limit ) ),
    _queueing_delay( get_val_or_default( dna, queueing_delay, is_lower_limit ) ),
    _loss(0), //FIXME: Should change "DNA" at some point
    _last_tick_sent( 0 ),
    _last_tick_received( 0 ),
    _min_rtt( 0 ),
    _send(0),
    _rec(0)
{
  assert(false);
}

size_t hash_value( const Memory & mem )
{
  size_t seed = 0;
  boost::hash_combine( seed, mem._rec_send_ewma );
  boost::hash_combine( seed, mem._rec_rec_ewma );
  boost::hash_combine( seed, mem._rtt_ratio );
  boost::hash_combine( seed, mem._slow_rec_rec_ewma );
  boost::hash_combine( seed, mem._rtt_diff );
  boost::hash_combine( seed, mem._queueing_delay );

  return seed;
}
