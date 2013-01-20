#include <assert.h>
#include <math.h>
#include <algorithm>

#include "whisker.hh"

using namespace std;

static const unsigned int MAX_WINDOW = 256;
static constexpr double MAX_RATE = 4.0;
static const unsigned int DEFAULT_WINDOW = 1;

Whiskers::Whiskers()
  : _domain( Memory(), MAX_MEMORY() ),
    _children(),
    _leaf( 1, Whisker( DEFAULT_WINDOW, MAX_RATE, _domain ) )
{
}

Whiskers::Whiskers( const Whisker & whisker, const bool bisect )
  : _domain( whisker.domain() ),
    _children(),
    _leaf()
{
  if ( !bisect ) {
    _leaf.push_back( whisker );
  } else {
    for ( auto &x : whisker.bisect() ) {
      _children.push_back( Whiskers( x, false ) );
    }
  }
}

vector< Whisker > Whisker::bisect( void ) const
{
  vector< Whisker > ret;
  for ( auto &x : _domain.bisect() ) {
    Whisker new_whisker( *this );
    new_whisker._domain = x;
    ret.push_back( new_whisker );
  }
  return ret;
}

void Whiskers::reset_counts( void )
{
  if ( !_leaf.empty() ) {
    _leaf.front().reset_count();
  }

  for ( auto &x : _children ) {
    x.reset_counts();
  }
}

bool Whisker::operator==( const Whisker & other ) const
{
  return (_generation == other._generation) && (_window == other._window) && (_rate == other._rate) && (_domain == other._domain); /* ignore count for now */
}

const Whisker & Whiskers::use_whisker( const Memory & _memory, const bool track ) const
{
  const Whisker * ret( whisker( _memory ) );
  assert( ret );

  ret->use();

  if ( track ) {
    ret->domain().track( _memory );
  }

  return *ret;
}

const Whisker * Whiskers::whisker( const Memory & _memory ) const
{
  if ( !_domain.contains( _memory ) ) {
    return nullptr;
  }

  if ( !_leaf.empty() ) {
    return &_leaf[ 0 ];
  }

  /* need to descend */
  for ( auto &x : _children ) {
    auto ret( x.whisker( _memory ) );
    if ( ret ) {
      return ret;
    }
  }

  assert( false );
}

Whisker::Whisker( const unsigned int s_window, const double s_rate, const MemoryRange & s_domain )
  : _generation( 0 ),
    _window( s_window ),
    _rate( s_rate ),
    _count( 0 ),
    _domain( s_domain )
{
}

Whisker::Whisker( const Whisker & other )
  : _generation( other._generation ),
    _window( other._window ),
    _rate( other._rate ),
    _count( other._count ),
    _domain( other._domain )
{
}

vector< Whisker > Whisker::next_generation( void ) const
{
  vector< Whisker > ret_windows;

  /* generate all window sizes */
  Whisker copy( *this );
  copy._generation++;
  ret_windows.push_back( copy );

  for ( unsigned int i = 1; i <= MAX_WINDOW ; i *= 2 ) {
    Whisker new_whisker( *this );
    new_whisker._generation++;

    if ( _window + i <= MAX_WINDOW ) {
      new_whisker._window = _window + i;
      ret_windows.push_back( new_whisker );
    }

    if ( _window > i ) {
      new_whisker._window = _window - i;
      ret_windows.push_back( new_whisker );
    }
  }

  /* generate all rates */
  vector< Whisker > ret;
  for ( auto &x : ret_windows ) {
    Whisker rate_copy( x );
    ret.push_back( rate_copy );

    for ( double rate_incr = 0.1; rate_incr <= MAX_RATE; rate_incr *= 2 ) {
      Whisker new_whisker( x );

      if ( x._rate + rate_incr <= MAX_RATE ) {
	new_whisker._rate = x._rate + rate_incr;
	ret.push_back( new_whisker );
      }

      if ( x._rate > rate_incr ) {
	new_whisker._rate = x._rate - rate_incr;
	ret.push_back( new_whisker );
      }
    }
  }

  return ret;
}

const Whisker * Whiskers::most_used( const unsigned int max_generation ) const
{
  if ( !_leaf.empty() ) {
    if ( (_leaf.front().generation() <= max_generation)
	 && (_leaf.front().count() > 0) ) {
      return &_leaf[ 0 ];
    }
    return nullptr;
  }

  /* recurse */
  unsigned int count_max = 0;
  const Whisker * ret( nullptr );

  for ( auto &x : _children ) {
    const Whisker * candidate( x.most_used( max_generation ) );
    if ( candidate
	 && (candidate->generation() <= max_generation)
	 && (candidate->count() >= count_max) ) {
      ret = candidate;
      count_max = candidate->count();
    }
  }

  return ret;
}

void Whiskers::promote( const unsigned int generation )
{
  if ( !_leaf.empty() ) {
    assert( _leaf.size() == 1 );
    assert( _children.empty() );
    _leaf.front().promote( generation );
  }

  for ( auto &x : _children ) {
    x.promote( generation );
  }
}

void Whisker::promote( const unsigned int generation )
{
  _generation = max( _generation, generation );
}

bool Whiskers::replace( const Whisker & w )
{
  if ( !_domain.contains( w.domain().range_median() ) ) {
    return false;
  }

  if ( !_leaf.empty() ) {
    assert( w.domain() == _leaf.front().domain() );
    _leaf.front() = w;
    return true;
  }

  for ( auto &x : _children ) {
    if ( x.replace( w ) ) {
      return true;
    }
  }

  assert( false );
}

bool Whiskers::replace( const Whisker & src, const Whiskers & dst )
{
  if ( !_domain.contains( src.domain().range_median() ) ) {
    return false;
  }
 
  if ( !_leaf.empty() ) {
    assert( src.domain() == _leaf.front().domain() );
    /* convert from leaf to interior node */
    *this = dst;
    return true;
  }

  for ( auto &x : _children ) {
    if ( x.replace( src, dst ) ) {
      return true;
    }
  }

  assert( false );
}

string Whisker::str( void ) const
{
  char tmp[ 128 ];
  snprintf( tmp, 128, "{%s} gen=%u ct=%u => (win=%u, rate=%.2f)",
	    _domain.str().c_str(), _generation, _count, _window, _rate );
  return tmp;
}

string Whiskers::str( void ) const
{

  if ( !_leaf.empty() ) {
    assert( _children.empty() );
    char tmp[ 128 ];
    snprintf( tmp, 128, "[%s]", _leaf.front().str().c_str() );
    return tmp;
  }

  string ret;

  for ( auto &x : _children ) {
    ret += x.str();
  }

  return ret;
}

unsigned int Whiskers::num_children( void ) const
{
  if ( !_leaf.empty() ) {
    assert( _leaf.size() == 1 );
    return 1;
  }

  return _children.size();
}

