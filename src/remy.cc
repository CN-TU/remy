#include <cstdio>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "ratbreeder.hh"
#include "dna.pb.h"
#include "configrange.hh"
using namespace std;

int main( int argc, char *argv[] )
{
  WhiskerTree whiskers;
  string output_filename;
  RemyBuffers::ConfigRange input_config;
  string config_filename;

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
    } else if ( arg.substr( 0, 3 ) == "of=" ) {
      output_filename = string( arg.substr( 3 ) );
    } else if ( arg.substr(0, 3 ) == "cf=" ) {
      config_filename = string( arg.substr( 3 ) );
      int cfd = open( config_filename.c_str(), O_RDONLY );
      if ( cfd < 0 ) {
        perror( "open config file error");
        exit( 1 );
      }
      if ( !input_config.ParseFromFileDescriptor( cfd ) ) {
        fprintf( stderr, "Could not parse input config from file %s. \n", config_filename.c_str() );
        exit ( 1 );
      }
      if ( close( cfd ) < 0 ) {
        perror( "close" );
        exit( 1 );
      }
    } 
  }

  if ( config_filename.empty() ) {
    fprintf( stderr, "Provide an input config protobuf. \n");
    exit ( 1 );
  }


  ConfigRange configuration_range;
  
  configuration_range.link_ppt = Range( input_config.link_packets_per_ms().low(), input_config.link_packets_per_ms().high(), input_config.link_packets_per_ms().incr() );

  configuration_range.rtt = Range( input_config.rtt().low(), input_config.rtt().high(), input_config.rtt().incr() );

  configuration_range.num_senders = Range( input_config.num_senders().low(), input_config.num_senders().high(), input_config.num_senders().incr() );
  configuration_range.mean_on_duration = Range( input_config.mean_on_duration().low(), input_config.mean_on_duration().high(), input_config.mean_on_duration().incr() );
  configuration_range.mean_off_duration = Range( input_config.mean_off_duration().low(), input_config.mean_off_duration().high(), input_config.mean_off_duration().incr() );
  RatBreeder breeder( configuration_range );
  unsigned int run = 0;

  printf( "#######################\n" );
  printf( "Optimizing for link packets_per_ms in [%f, %f]\n",
	  configuration_range.link_ppt.low,
	  configuration_range.link_ppt.high );
  printf( "Optimizing for rtt_ms in [%f, %f]\n",
	  configuration_range.rtt.low,
	  configuration_range.rtt.high );
  printf( "Optimizing for num_senders in [%f, %f]\n",
	  configuration_range.num_senders.low, configuration_range.num_senders.high );
  printf( "Optimizing for mean_on_duration in [%f, %f], mean_off_duration in [ %f, %f]\n",
	  configuration_range.mean_on_duration.low, configuration_range.mean_on_duration.high, configuration_range.mean_off_duration.low, configuration_range.mean_off_duration.high );

  printf( "Initial rules (use if=FILENAME to read from disk): %s\n", whiskers.str().c_str() );
  printf( "#######################\n" );

  if ( !output_filename.empty() ) {
    printf( "Writing to \"%s.N\".\n", output_filename.c_str() );
  } else {
    printf( "Not saving output. Use the of=FILENAME argument to save the results.\n" );
  }

  while ( 1 ) {
    auto outcome = breeder.improve( whiskers );
    printf( "run = %u, score = %f\n", run, outcome.score );

    printf( "whiskers: %s\n", whiskers.str().c_str() );

    for ( auto &run : outcome.throughputs_delays ) {
      printf( "===\nconfig: %s\n", run.first.str().c_str() );
      for ( auto &x : run.second ) {
	printf( "sender: [tp=%f, del=%f]\n", x.first / run.first.link_ppt, x.second / run.first.delay );
      }
    }

    if ( !output_filename.empty() ) {
      char of[ 128 ];
      snprintf( of, 128, "%s.%d", output_filename.c_str(), run );
      fprintf( stderr, "Writing to \"%s\"... ", of );
      int fd = open( of, O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IWUSR );
      if ( fd < 0 ) {
	perror( "open" );
	exit( 1 );
      }

      auto remycc = whiskers.DNA();
      remycc.mutable_config()->CopyFrom( configuration_range.DNA() );
      remycc.mutable_optimizer()->CopyFrom( Whisker::get_optimizer().DNA() );

      if ( not remycc.SerializeToFileDescriptor( fd ) ) {
	fprintf( stderr, "Could not serialize RemyCC.\n" );
	exit( 1 );
      }

      if ( close( fd ) < 0 ) {
	perror( "close" );
	exit( 1 );
      }

      fprintf( stderr, "done.\n" );
    }

    fflush( NULL );
    run++;
  }

  return 0;
}
