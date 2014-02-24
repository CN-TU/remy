#include <cstdio>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "whiskertree.hh"
#include "network.cc"
#include "sendergangofgangs.hh"
#include "rat.hh"
#include "graph.hh"

using namespace std;

int main( int argc, char *argv[] )
{
  WhiskerTree whiskers;
  unsigned int num_senders = 2;
  double link_ppt = 1.0;
  double delay = 100.0;
  double mean_on_duration = 5000.0;
  double mean_off_duration = 5000.0;

  for ( int i = 1; i < argc; i++ ) {
    string arg( argv[ i ] );
    if ( arg.substr( 0, 3 ) == "if=" ) {
      string filename( arg.substr( 3 ) );
      int fd = open( filename.c_str(), O_RDONLY );
      if ( fd < 0 ) {
	perror( "open" );
	exit( 1 );
      }

      RemyBuffers::WhiskerTree tree;
      if ( !tree.ParseFromFileDescriptor( fd ) ) {
	fprintf( stderr, "Could not parse %s.\n", filename.c_str() );
	exit( 1 );
      }
      whiskers = WhiskerTree( tree );

      if ( close( fd ) < 0 ) {
	perror( "close" );
	exit( 1 );
      }

      if ( tree.has_config() ) {
	printf( "Prior assumptions:\n%s\n\n", tree.config().DebugString().c_str() );
      }

      if ( tree.has_optimizer() ) {
	printf( "Remy optimization settings:\n%s\n\n", tree.optimizer().DebugString().c_str() );
      }
    } else if ( arg.substr( 0, 5 ) == "nsrc=" ) {
      num_senders = atoi( arg.substr( 5 ).c_str() );
      fprintf( stderr, "Setting num_senders to %d\n", num_senders );
    } else if ( arg.substr( 0, 5 ) == "link=" ) {
      link_ppt = atof( arg.substr( 5 ).c_str() );
      fprintf( stderr, "Setting link packets per ms to %f\n", link_ppt );
    } else if ( arg.substr( 0, 4 ) == "rtt=" ) {
      delay = atof( arg.substr( 4 ).c_str() );
      fprintf( stderr, "Setting delay to %f ms\n", delay );
    } else if ( arg.substr( 0, 3 ) == "on=" ) {
      mean_on_duration = atof( arg.substr( 3 ).c_str() );
      fprintf( stderr, "Setting mean_on_duration to %f ms\n", mean_on_duration );
    } else if ( arg.substr( 0, 4 ) == "off=" ) {
      mean_off_duration = atof( arg.substr( 4 ).c_str() );
      fprintf( stderr, "Setting mean_off_duration to %f ms\n", mean_off_duration );
    }
  }

  NetConfig configuration = NetConfig().set_link_ppt( link_ppt ).set_delay( delay ).set_num_senders( num_senders ).set_on_duration( mean_on_duration ).set_off_duration( mean_off_duration );

  PRNG prng( 50 );
  Network<Rat, Rat> network( Rat( whiskers, false ), prng, configuration );

  Graph graph( num_senders + 1, 1024, 600, "Ratatouille", 0, link_ppt * delay * 1.2 );

  graph.set_color( 0, 1, 0, 0, 0.75 );
  graph.set_color( 1, 1, 0.38, 0, 0.75 );
  graph.set_color( 2, 0, 0, 1, 0.75 );

  float t = 0.0;

  while ( 1 ) {
    network.run_simulation_until( t * 1000.0 );

    const vector< int > packets_in_flight = network.senders().packets_in_flight();

    float ideal_pif_per_sender = 0;
    const unsigned int active_senders = network.senders().count_active_senders();

    if ( active_senders ) {
      ideal_pif_per_sender = link_ppt * delay / active_senders;
    }

    graph.add_data_point( 0, t, ideal_pif_per_sender );

    for ( unsigned int i = 0; i < packets_in_flight.size(); i++ ) {
      graph.add_data_point( i + 1, t, packets_in_flight[ i ] );
    }

    graph.set_window( t, 10 );

    if ( graph.blocking_draw( t, 10, 0, link_ppt * delay * 1.2 ) ) {
      break;
    }

    t += .01;
  }

  for ( auto &x : network.senders().throughputs_delays() ) {
    printf( "sender: [tp=%f, del=%f]\n", x.first, x.second );
  }

  return EXIT_SUCCESS;
}
