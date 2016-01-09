#ifndef NETWORK_HH
#define NETWORK_HH

#include <string>

#include "sendergangofgangs.hh"
#include "link.hh"
#include "delay.hh"
#include "receiver.hh"
#include "random.hh"
#include "answer.pb.h"

class NetConfig
{
public:
  double mean_on_duration, mean_off_duration;
  double num_senders;
  double link_ppt;
  double delay;
  double buffer_size;

  NetConfig( void )
    : mean_on_duration( 5000.0 ),
      mean_off_duration( 5000.0 ),
      num_senders( 8 ),
      link_ppt( 1.0 ),
      delay( 150 ),
      buffer_size( std::numeric_limits<unsigned int>::max() )
  {}

  NetConfig( const RemyBuffers::NetConfig & dna )
    : mean_on_duration( dna.mean_on_duration() ),
      mean_off_duration( dna.mean_off_duration() ),
      num_senders( dna.num_senders() ),
      link_ppt( dna.link_ppt() ),
      delay( dna.delay() ),
      buffer_size( dna.buffer_size() )
  {}
  
  NetConfig & set_link_ppt( const double s_link_ppt ) { link_ppt = s_link_ppt; return *this; }
  NetConfig & set_delay( const double s_delay ) { delay = s_delay; return *this; }
  NetConfig & set_num_senders( const unsigned int n ) { num_senders = n; return *this; }
  NetConfig & set_on_duration( const double & duration ) { mean_on_duration = duration; return *this; }
  NetConfig & set_off_duration( const double & duration ) { mean_off_duration = duration; return *this; }
  NetConfig & set_buffer_size( const unsigned int n ) { buffer_size = n; return *this; }

  RemyBuffers::NetConfig DNA( void ) const
  {
      RemyBuffers::NetConfig ret;
      ret.set_mean_on_duration( mean_on_duration );
      ret.set_mean_off_duration( mean_off_duration );
      ret.set_num_senders( num_senders );
      ret.set_delay( delay );
      ret.set_link_ppt( link_ppt );
      ret.set_buffer_size( buffer_size );
      return ret;
  }

  std::string str( void ) const
  {
    char tmp[ 256 ];
    snprintf( tmp, 256, "mean_on=%f, mean_off=%f, nsrc=%f, link_ppt=%f, delay=%f, buffer_size=%f\n",
	     mean_on_duration, mean_off_duration, num_senders, link_ppt, delay, buffer_size );
    return tmp;
  }
};

template <class SenderType1, class SenderType2>
class Network
{
private:
  PRNG & _prng;
  SenderGangofGangs<SenderType1, SenderType2> _senders;
  Link _link;
  Delay _delay;
  Receiver _rec;

  double _tickno;

  void tick( void );

public:
  Network( const SenderType1 & example_sender1, const SenderType2 & example_sender2, PRNG & s_prng, const NetConfig & config );

  Network( const SenderType1 & example_sender1, PRNG & s_prng, const NetConfig & config );

  void run_simulation( const double & duration );

  const SenderGangofGangs<SenderType1,SenderType2> & senders( void ) const { return _senders; }
};

#endif
