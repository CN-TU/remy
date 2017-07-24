#ifndef STOCHASTICLOSS_HH
#define STOCHASTICLOSS_HH

#include "packet.hh"
#include <deque>
#include <tuple>
#include <random>
#include "exponential.hh"

// FIXME: Why do I even need this?
#define UNUSED(x) (void)(x)

class StochasticLoss
{
  private:
    std::deque< std::tuple< double, Packet > > _buffer;
    double _loss_rate;
    PRNG & _prng;
    std::bernoulli_distribution _distr;
    Receiver & _rec;

  public:
    StochasticLoss( const double & rate, PRNG &prng, Receiver & s_rec ) :  _buffer(), _loss_rate( rate ), _prng( prng ), _distr(), _rec(s_rec) { _distr = std::bernoulli_distribution( rate );}
    template <class NextHop>
    void tick( NextHop & next, const double & tickno )
    {
      // pops items off buffer and sends them to the next item
      while ( (!_buffer.empty())) {
        next.accept(std::get< 1 >(_buffer.front()), tickno);
        _buffer.pop_front();
      }
    }
    void accept( Packet & p, const double & tickno ) noexcept
    {
      if (!(_distr( _prng ))) {
        _buffer.emplace_back( tickno, p );
      } else {
        _rec.accept_lost(p, tickno);
      }
    }

    double next_event_time( const double & tickno ) const
    {
      UNUSED(tickno);
      if ( _buffer.empty() ) {
        return std::numeric_limits<double>::max();
      }
      assert( std::get< 0 >( _buffer.front() ) >= tickno );
      return std::get< 0 >( _buffer.front() );
    }

};

#endif
