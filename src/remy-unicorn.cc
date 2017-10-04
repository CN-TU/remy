#include <cstdio>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "unicornbreeder.hh"
#include "rainbow.hh"
#include "configrange.hh"
#include <thread>

#include <signal.h>
#include <stdlib.h>
#include <limits>

#include <unistd.h>

struct stat st = {};
bool cooperative = true;

using namespace std;

void print_range( const Range & range, const string & name )
{
  printf( "Optimizing for %s over [%f : %f : %f]\n", name.c_str(),
    range.low, range.incr, range.high );
}

void signal_handler(int s) {
  printf("Signal Handler: Caught signal %d\n", s);
  Rainbow& unicorn_farm = Rainbow::getInstance(cooperative);
  unicorn_farm.save_session();
  exit(EXIT_SUCCESS);
}

void signal_handler_just_save(int s) {
  printf("Signal Handler: Caught signal %d\n", s);
  Rainbow& unicorn_farm = Rainbow::getInstance(cooperative);
  unicorn_farm.save_session();
}

void unicorn_thread(const size_t thread_id, const BreederOptionsUnicorn options, const size_t iterations_per_thread) {
  printf("Creating thread no %zd\n", thread_id);
  UnicornBreeder breeder(options, thread_id);
  auto outcome = breeder.run(iterations_per_thread);
  printf("thread = %u, score = %f\n", (unsigned int) thread_id, outcome.score);
}

int main( int argc, char *argv[] )
{
  // WhiskerTree whiskers;
  BreederOptionsUnicorn options;
  // FIXME: Unused. Might be useful to compare to original Remy...
  // WhiskerImproverOptions whisker_options;
  RemyBuffers::ConfigRangeUnicorn input_config;
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

      // RemyBuffers::WhiskerTree tree;
      // if ( !tree.ParseFromFileDescriptor( fd ) ) {
      //   fprintf( stderr, "Could not parse %s.\n", filename.c_str() );
      //   exit( 1 );
      // }
      // whiskers = WhiskerTree( tree );

      if ( close( fd ) < 0 ) {
        perror( "close" );
        exit( 1 );
      }

    // } else if ( arg.substr( 0, 3 ) == "of=" ) {
    //   output_filename = string( arg.substr( 3 ) );

    // } else if ( arg.substr( 0, 4 ) == "opt=" ) {
    //   whisker_options.optimize_window_increment = false;
    //   whisker_options.optimize_window_multiple = false;
    //   whisker_options.optimize_intersend = false;
    //   for ( char & c : arg.substr( 4 ) ) {
    //     if ( c == 'b' ) {
    //       whisker_options.optimize_window_increment = true;
    //     } else if ( c == 'm' ) {
    //       whisker_options.optimize_window_multiple = true;
    //     } else if ( c == 'r' ) {
    //       whisker_options.optimize_intersend = true;
    //     } else {
    //       fprintf( stderr, "Invalid optimize option: %c\n", c );
    //       exit( 1 );
    //     }
    //   }

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

  options.config_range = ConfigRangeUnicorn( input_config );

  // unsigned int run = 0;

  printf( "#######################\n" );
  // printf( "Evaluator simulations will run for %d ticks\n",
      // options.config_range.simulation_ticks );
  print_range( options.config_range.simulation_ticks, "simulation_ticks" );
  // printf( "Optimizing window increment: %d, window multiple: %d, intersend: %d\n",
  //         whisker_options.optimize_window_increment, whisker_options.optimize_window_multiple,
  //         whisker_options.optimize_intersend);
  print_range( options.config_range.link_ppt, "link packets_per_ms" );
  print_range( options.config_range.rtt, "rtt_ms" );
  print_range( options.config_range.num_senders, "num_senders" );
  print_range( options.config_range.mean_on_duration, "mean_on_duration" );
  print_range( options.config_range.mean_off_duration, "mean_off_duration" );
  print_range( options.config_range.stochastic_loss_rate, "stochastic_loss_rate" );
  if ( options.config_range.buffer_size.low != numeric_limits<unsigned int>::max() ) {
    print_range( options.config_range.buffer_size, "buffer_size" );
  } else {
    printf( "Optimizing for infinitely sized buffers. \n");
  }
  printf( "Number of threads is %u\n", options.config_range.num_threads );
  printf( "Training %s\n", options.config_range.cooperative ? "cooperatively" : "independently" );
  printf("Delay delta is %f\n", options.config_range.delay_delta);
  cooperative = options.config_range.cooperative;
  printf( "#######################\n" );

  // Allocates storage
  char* number_of_threads_string = (char*) malloc(2048 * sizeof(char));
  // Prints "Hello world!" on hello_world
  sprintf(number_of_threads_string, "num_threads=%u", options.config_range.num_threads*(uint32_t) options.config_range.num_senders.high);
  putenv(number_of_threads_string);

  const size_t iterations_per_thread = std::numeric_limits<size_t>::max();
  vector<thread> thread_array(options.config_range.num_threads);

  if (stat("stats", &st) == -1) {
    mkdir("stats", 0700);
  }
  for (size_t i=0; i<options.config_range.num_threads; i++) {
    thread_array[i] = thread(unicorn_thread, i, options, iterations_per_thread);
  }

  printf("Created threads\n");
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGUSR1, signal_handler_just_save);
  printf("Registered handlers\n");

  // Never gonna happen
  for (size_t i=0; i<options.config_range.num_threads; i++) {
    thread_array[i].join();
    printf("All threads finished!\n");
  }

  return 0;
}
