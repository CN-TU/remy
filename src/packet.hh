#ifndef PACKET_HH
#define PACKET_HH

namespace remy {
  class Packet
  {
  public:
    unsigned int src;
    unsigned int flow_id;
    double tick_sent, tick_received;
    long int seq_num;

    Packet( const unsigned int & s_src,
      const unsigned int & s_flow_id,
      const double & s_tick_sent,
      const long int & s_seq_num )
      : src( s_src ),
        flow_id( s_flow_id ), tick_sent( s_tick_sent ),
        tick_received( -1 ),
        seq_num( s_seq_num )
    {}

    Packet(const long int & s_seq_num, const double & s_tick_sent, const double & s_tick_received )
    : src( 0 ),
      flow_id( 0 ), tick_sent( s_tick_sent ),
      tick_received( s_tick_received ),
      seq_num( s_seq_num )
    {}

  };
}

#endif
