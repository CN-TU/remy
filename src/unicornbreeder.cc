#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <cstdio>
#include <sys/stat.h>
#include <fcntl.h>

#include "unicornbreeder.hh"

using namespace std;

UnicornEvaluator::Outcome UnicornBreeder::run(const size_t iterations)
{
  char file_name[64 * sizeof(char)];
  sprintf(file_name, "stats/thread%lu", _thread_id);
  FILE* f = fopen(file_name, "w");
  if (f == NULL) {
    puts("Error opening file!\n");
    exit(1);
  }
  for (size_t i=0; i<iterations; i++) {
    const UnicornEvaluator eval( _options.config_range, _thread_id );

    auto outcome_and_logging = eval.score_with_logging();

    auto outcome = get<0>(outcome_and_logging);
    auto logging = get<1>(outcome_and_logging);

    string csv_stuff = "";

    csv_stuff += "time,sender_number,packets_received,sending_duration,total_delay\n";
    for (vector<SimulationRunData>::const_iterator it = logging.data().begin(); it != logging.data().end(); ++it) {
      for (vector<SimulationRunDataPoint>::const_iterator inner_it = (*it).data().begin(); inner_it != (*it).data().end(); ++inner_it) {
        size_t sender_index = 0;
        for (vector<SenderDataPoint>::const_iterator sender_it = (*inner_it).sender_data().begin(); sender_it != (*inner_it).sender_data().end(); ++sender_it) {
          csv_stuff += to_string((*inner_it).seconds()) + "," + to_string(sender_index) + "," + to_string((*sender_it).utility_data.packets_received()) + "," + to_string((*sender_it).utility_data.sending_duration()) + "," + to_string((*sender_it).utility_data.total_delay()) + "\n";
          sender_index += 1;
        }
      }
    }

    const double final_score = outcome.score;
    fprintf(f, "%lu,%lu,%f\n", i, (unsigned long)time(NULL), final_score);
    fflush(f);
    printf("Finished iteration %lu in thread %lu! Score is %f.\n", i, _thread_id, final_score);

    char* file_name = getenv("checkpoints");
    string output_filename = "logging/" + string(strchr(file_name, '/')+1);

    if ( !output_filename.empty() ) {
      char of[ 128 ];
      snprintf( of, 128, "%s.%d", output_filename.c_str(), (int) i );
      fprintf( stderr, "Writing to \"%s\"... ", of );
      int fd = open( of, O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IWUSR );
      if ( fd < 0 ) {
        perror( "open" );
        exit( 1 );
      }

      auto log_thing = logging.DNA();
      if ( not log_thing.SerializeToFileDescriptor( fd ) ) {
        fprintf( stderr, "Could not serialize RemyCC.\n" );
        exit( 1 );
      }

      if ( close( fd ) < 0 ) {
        perror( "close" );
        exit( 1 );
      }

      fprintf( stderr, "done.\n" );
    }

    string output_filename_csv = "csv/" + string(strchr(file_name, '/')+1);

    if ( !output_filename_csv.empty() ) {
      char of[ 128 ];
      snprintf( of, 128, "%s.%d.csv", output_filename_csv.c_str(), (int) i );
      fprintf( stderr, "Writing to \"%s\"... ", of );
      FILE* fd = fopen( of, "w" );
      if ( fd == NULL ) {
        perror( "open" );
        exit( 1 );
      }

      fprintf(fd, "%s", csv_stuff.c_str());
      fflush(fd);

      fclose( fd );

      fprintf( stderr, "done.\n" );
    }
  }

  fclose(f);

  const UnicornEvaluator eval2( _options.config_range, _thread_id );
  const auto new_score = eval2.score();

  return new_score;
}