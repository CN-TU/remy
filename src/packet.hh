#ifndef PACKET_HH
#define PACKET_HH

class Packet
{
public:
  unsigned int src;
  unsigned int flow_id;
  double tick_sent, tick_received;
  int seq_num;
  long int unsigned previous_attempts;

  Packet( const unsigned int & s_src,
	  const unsigned int & s_flow_id,
	  const double & s_tick_sent,
	  const int & s_seq_num )
    : src( s_src ),
      flow_id( s_flow_id ), tick_sent( s_tick_sent ),
      tick_received( -1 ),
      seq_num( s_seq_num ),
      previous_attempts(0)
  {}

  Packet( const unsigned int & s_src,
  const unsigned int & s_flow_id,
  const double & s_tick_sent,
  const int & s_seq_num,
  const long unsigned int & s_previous_attempts)
  : src( s_src ),
    flow_id( s_flow_id ), tick_sent( s_tick_sent ),
    tick_received( -1 ),
    seq_num( s_seq_num ),
    previous_attempts( s_previous_attempts )
{}
  
};

#endif
