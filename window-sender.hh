#ifndef WINDOWSENDER_HH
#define WINDOWSENDER_HH

#include <vector>

#include "packet.hh"

class WindowSender
{
private:
  unsigned int _window;
  unsigned int _packets_sent, _packets_received;

public:
  WindowSender( const unsigned int s_window );

  void packets_received( const std::vector< Packet > & packets );
  void dormant_tick( const unsigned int tickno ); /* do nothing */

  template <class NextHop>
  void send( const unsigned int id, NextHop & next, const unsigned int tickno );
};

#endif
