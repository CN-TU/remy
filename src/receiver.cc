#include <cassert>
#include <limits>

#include "receiver.hh"

Receiver::Receiver()
<<<<<<< HEAD
  : _collector()
  // _collector_lost()
=======
  : _collector(),
  _collector_lost()
>>>>>>> 331e785d61963efd7bb8e85fc9a5f1cecb7adea9
{
}

void Receiver::accept( const Packet & p, const double & tickno ) noexcept
{
  autosize( p.src );

  _collector[ p.src ].push_back( p );
  _collector[ p.src ].back().tick_received = tickno;
}

// void Receiver::accept_lost( Packet & p, const double & tickno ) noexcept
// {
//   p.lost = true;
//   autosize( p.src );

//   _collector_lost[ p.src ].push_back( p );
//   _collector_lost[ p.src ].back().tick_received = tickno;
// }

void Receiver::autosize( const unsigned int index )
{
  if ( index >= _collector.size() ) {
    _collector.resize( index + 1 );
  }
}

// void Receiver::autosize_lost( const unsigned int index )
// {
//   if ( index >= _collector_lost.size() ) {
//     _collector_lost.resize( index + 1 );
//   }
// }

double Receiver::next_event_time( const double & tickno ) const
{
  for ( const auto & x : _collector_lost ) {
    if ( not x.empty() ) {
      return tickno;
    }
  }
  for ( const auto & x : _collector ) {
    if ( not x.empty() ) {
      return tickno;
    }
  }
  return std::numeric_limits<double>::max();
}
