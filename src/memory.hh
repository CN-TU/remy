#ifndef MEMORY_HH
#define MEMORY_HH

#define MIN_WINDOW_UNICORN 1.0
#define MAX_WINDOW_UNICORN 1000.0

#include <vector>
#include <string>

#include "packet.hh"
#include "../protobufs/dna.pb.h"

class Memory {
public:
  typedef double DataType;

private:

public:
  DataType _rec_send_ewma;
  DataType _rec_rec_ewma;
  DataType _rtt_ratio;
  DataType _slow_rec_rec_ewma;
  DataType _rtt_diff;
  DataType _queueing_delay;
  DataType _slow_rec_send_ewma;
  DataType _loss;
  DataType _loss_ewma;
  DataType _slow_loss_ewma;
  double _last_tick_sent;
  double _last_tick_received;
  double _min_rtt;
  double _send;
  double _rec;
  int _lost_since_last_time;
  DataType _rtt;
  DataType _rtt_ewma;
  DataType _slow_rtt_ewma;
  DataType _window;
  DataType _window_ewma;
  DataType _slow_window_ewma;

  Memory( const std::vector< DataType > & s_data )
    : _rec_send_ewma( s_data.at( 0 ) ),
      _rec_rec_ewma( s_data.at( 1 ) ),
      _rtt_ratio( s_data.at( 2 ) ),
      _slow_rec_rec_ewma( s_data.at( 3 ) ),
      _rtt_diff( s_data.at(4) ),
      _queueing_delay( s_data.at(5) ),
      _slow_rec_send_ewma( s_data.at(6) ),
      _loss( 0 ),
      _loss_ewma(0),
      _slow_loss_ewma(0),
      _last_tick_sent( 0 ),
      _last_tick_received( 0 ),
      _min_rtt( 0 ),
      _send (0),
      _rec(0),
      _lost_since_last_time(0),
      _rtt(0),
      _rtt_ewma(0),
      _slow_rtt_ewma(0),
      _window(0),
      _window_ewma(0),
      _slow_window_ewma(0)
  {}

  Memory()
    : _rec_send_ewma( 0 ),
      _rec_rec_ewma( 0 ),
      _rtt_ratio( 0.0 ),
      _slow_rec_rec_ewma( 0 ),
      _rtt_diff( 0 ),
      _queueing_delay( 0 ),
      _slow_rec_send_ewma( 0 ),
      _loss( 0 ),
      _loss_ewma(0),
      _slow_loss_ewma(0),
      _last_tick_sent( 0 ),
      _last_tick_received( 0 ),
      _min_rtt( 0 ),
      _send(0),
      _rec(0),
      _lost_since_last_time(0),
      _rtt(0),
      _rtt_ewma(0),
      _slow_rtt_ewma(0),
      _window(0),
      _window_ewma(0),
      _slow_window_ewma(0)
  {}

  void reset( void ) { _rec_send_ewma = _rec_rec_ewma = _rtt_ratio = _slow_rec_rec_ewma = _slow_rec_send_ewma = _rtt_diff = _queueing_delay = _last_tick_sent = _last_tick_received = _min_rtt = _loss = _loss_ewma = _slow_loss_ewma = _send = _rec = _lost_since_last_time = _rtt = _rtt_ewma = _slow_rtt_ewma = _window = _window_ewma = _slow_window_ewma = 0; _window = _window_ewma = _slow_window_ewma = MIN_WINDOW_UNICORN; }

  static const unsigned int datasize = 6;

  const DataType & field( unsigned int num ) const { return num == 0 ? _rec_send_ewma : num == 1 ? _rec_rec_ewma : num == 2 ? _rtt_ratio : num == 3 ? _slow_rec_rec_ewma : num == 4 ? _rtt_diff : num == 5 ? _queueing_delay : num == 6 ? _slow_rec_send_ewma : _loss ; }
  DataType & mutable_field( unsigned int num )     { return num == 0 ? _rec_send_ewma : num == 1 ? _rec_rec_ewma : num == 2 ? _rtt_ratio : num == 3 ? _slow_rec_rec_ewma : num == 4 ? _rtt_diff : num == 5 ? _queueing_delay : num == 6 ? _slow_rec_send_ewma : _loss ; }

  void packet_sent( const remy::Packet & packet __attribute((unused)) ) {}
  void packets_received( const std::vector< remy::Packet > & packets, const unsigned int flow_id, const int largest_ack );
  void lost(const int lost);
  void window(const double& s_window);
  void advance_to( const unsigned int tickno __attribute((unused)) ) {}

  std::string str( void ) const;
  std::string str( unsigned int num ) const;

  bool operator>=( const Memory & other ) const {
    for (unsigned int i = 0; i < datasize; i ++) { if ( field(i) < other.field(i) ) return false; }
    return true;
  }
  bool operator<( const Memory & other ) const {
    for (unsigned int i = 0; i < datasize; i ++) { if ( field(i) >= other.field(i) ) return false; }
    return true;
  }
  bool operator==( const Memory & other ) const {
    for (unsigned int i = 0; i < datasize; i ++) { if ( field(i) != other.field(i) ) return false; }
    return true;
  }

  RemyBuffers::Memory DNA( void ) const;
  Memory( const bool is_lower_limit, const RemyBuffers::Memory & dna );

  friend size_t hash_value( const Memory & mem );
};

extern const Memory & MAX_MEMORY( void );

#endif
