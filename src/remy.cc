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
  RatBreederOptions options;
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

    } else if ( arg.substr( 0, 4 ) == "opt=" ) {
      options.improver_options.optimize_window_increment = false;
      options.improver_options.optimize_window_multiple = false;
      options.improver_options.optimize_intersend = false;
      for ( char & c : arg.substr( 4 ) ) {
        if ( c == 'b' ) {
          options.improver_options.optimize_window_increment = true;
        } else if ( c == 'm' ) {
          options.improver_options.optimize_window_multiple = true;
        } else if ( c == 'r' ) {
          options.improver_options.optimize_intersend = true;
        } else {
          fprintf( stderr, "Invalid optimize option: %c\n", c );
          exit( 1 );
        }
      }

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
    fprintf( stderr, "An input configuration protobuf must be provided via the cf= option. \n");
    fprintf( stderr, "You can generate one using './configuration'. \n");
    exit ( 1 );
  }


  options.config_range.link_ppt = Range( input_config.link_packets_per_ms() );
  options.config_range.rtt = Range( input_config.rtt() );
  options.config_range.num_senders = Range( input_config.num_senders() );
  options.config_range.mean_on_duration = Range( input_config.mean_on_duration() );
  options.config_range.mean_off_duration = Range( input_config.mean_off_duration() );
  options.config_range.buffer_size = Range( input_config.buffer_size() );
  options.config_range.simulation_ticks = input_config.simulation_ticks();

  RatBreeder breeder( options );

  unsigned int run = 0;

  printf( "#######################\n" );
  printf( "Evaluator simulations will run for %d ticks\n",
    options.config_range.simulation_ticks );
  printf( "Optimizing for link packets_per_ms in [%f, %f]\n",
	  options.config_range.link_ppt.low,
	  options.config_range.link_ppt.high );
  printf( "Optimizing for rtt_ms in [%f, %f]\n",
	  options.config_range.rtt.low,
	  options.config_range.rtt.high );
  printf( "Optimizing for num_senders in [%f, %f]\n",
	  options.config_range.num_senders.low, options.config_range.num_senders.high );
  printf( "Optimizing for mean_on_duration in [%f, %f], mean_off_duration in [ %f, %f]\n",
	  options.config_range.mean_on_duration.low, options.config_range.mean_on_duration.high, options.config_range.mean_off_duration.low, options.config_range.mean_off_duration.high );
  printf( "Optimizing window increment: %d, window multiple: %d, intersend: %d\n",
          options.improver_options.optimize_window_increment, options.improver_options.optimize_window_multiple,
          options.improver_options.optimize_intersend);
  if ( options.config_range.buffer_size.low != numeric_limits<unsigned int>::max() ) {
    printf( "Optimizing for buffer_size in [%f, %f]\n",
            options.config_range.buffer_size.low,
            options.config_range.buffer_size.high );
  } else {
    printf( "Optimizing for infinitely sized buffers. \n");
  }

  printf( "Initial rules (use if=FILENAME to read from disk): %s\n", whiskers.str().c_str() );
  printf( "#######################\n" );

  if ( !output_filename.empty() ) {
    printf( "Writing to \"%s.N\".\n", output_filename.c_str() );
  } else {
    printf( "Not saving output. Use the of=FILENAME argument to save the results.\n" );
  }

  RemyBuffers::ConfigVector training_configs;
  bool written = false;

  while ( 1 ) {
    auto outcome = breeder.improve( whiskers );
    printf( "run = %u, score = %f\n", run, outcome.score );

    printf( "whiskers: %s\n", whiskers.str().c_str() );

    for ( auto &run : outcome.throughputs_delays ) {
      if ( !(written) ) {
        for ( auto &run : outcome.throughputs_delays) {
          // record the config to the protobuf
          RemyBuffers::NetConfig* net_config = training_configs.add_config();
          *net_config = run.first.DNA();
          written = true;
      
        }
      }
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
      remycc.mutable_config()->CopyFrom( options.config_range.DNA() );
      remycc.mutable_optimizer()->CopyFrom( Whisker::get_optimizer().DNA() );
      remycc.mutable_configvector()->CopyFrom( training_configs );
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
