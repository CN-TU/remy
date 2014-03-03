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
#include "aimd-templates.cc"
#include "graph.hh"
#include "fader-templates.cc"

using namespace std;

int main( int argc, char *argv[] )
{
  WhiskerTree whiskers;
  unsigned int num_senders = 2;
  double link_ppt = 1.0;
  double delay = 100.0;
  double mean_on_duration = 5000.0;
  double mean_off_duration = 5000.0;
  string fader_filename;

  for ( int i = 1; i < argc; i++ ) {
    string arg( argv[ i ] );
    if ( arg.substr( 0, 6 ) == "fader=" ) {
      fader_filename = arg.substr( 6 );
    } else if ( arg.substr( 0, 3 ) == "if=" ) {
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

  Fader fader( fader_filename );

  NetConfig configuration = NetConfig().set_link_ppt( link_ppt ).set_delay( delay ).set_num_senders( num_senders ).set_on_duration( mean_on_duration ).set_off_duration( mean_off_duration );

  PRNG prng( 50 );
  Network<SenderGang<Rat, ExternalSwitchedSender<Rat>>,
	  SenderGang<Aimd, ExternalSwitchedSender<Aimd>>> network( Rat( whiskers, false ), Aimd(), prng, configuration );

  float upper_limit = link_ppt * delay * 1.2;

  Graph graph( 2 * num_senders + 1, 1024, 600, "Ratatouille", 0, upper_limit );

  graph.set_color( 0, 0, 0, 0, 1.0 );
  graph.set_color( 1, 1, 0.38, 0, 0.8 );
  graph.set_color( 2, 0, 0.2, 1, 0.8 );
  graph.set_color( 3, 1, 0, 0, 0.8 );
  graph.set_color( 4, 0.5, 0, 0.5, 0.8 );
  graph.set_color( 5, 0, 0.5, 0.5, 0.8 );
  graph.set_color( 6, 0.5, 0.5, 0.5, 0.8 );
  graph.set_color( 7, 0.2, 0.2, 0.5, 0.8 );
  graph.set_color( 8, 0.2, 0.5, 0.2, 0.8 );

  float t = 0.0;

  while ( 1 ) {
    fader.update( network );

    network.run_simulation_until( t * 1000.0 );

    const vector< unsigned int > packets_in_flight = network.packets_in_flight();

    float ideal_pif_per_sender = 0;
    const unsigned int active_senders = network.senders().count_active_senders();

    if ( active_senders ) {
      ideal_pif_per_sender = link_ppt * delay / active_senders;
    }

    graph.add_data_point( 0, t, ideal_pif_per_sender );
    if ( ideal_pif_per_sender > upper_limit ) {
      upper_limit = ideal_pif_per_sender * 1.1;
    }

    upper_limit = max( upper_limit, ideal_pif_per_sender );

    for ( unsigned int i = 0; i < packets_in_flight.size(); i++ ) {
      graph.add_data_point( i + 1, t, packets_in_flight[ i ] );

      if ( packets_in_flight[ i ] > upper_limit ) {
	upper_limit = packets_in_flight[ i ] * 1.1;
      }
    }

    graph.set_window( t, 10 );

    if ( graph.blocking_draw( t, 10, 0, upper_limit ) ) {
      break;
    }

    t += .01;
  }

  for ( auto &x : network.senders().throughputs_delays() ) {
    printf( "sender: [tp=%f, del=%f]\n", x.first, x.second );
  }

  return EXIT_SUCCESS;
}
