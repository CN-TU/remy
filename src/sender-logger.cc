#include <iostream>
#include <fstream>
#include <string>

#include "simulationresults.hh"
#include "network.cc"
#include "sendergang.hh"
#include "fish.hh"
#include "fish-templates.cc"
#include "rat.hh"
#include "rat-templates.cc"

template <typename ActionTree>
void print_tree(ActionTree & tree)
{
  if ( tree.has_config() ) {
    printf( "Prior assumptions:\n%s\n\n", tree.config().DebugString().c_str() );
  }

  if ( tree.has_optimizer() ) {
    printf( "Remy optimization settings:\n%s\n\n", tree.optimizer().DebugString().c_str() );
  }
}

template <typename Sender, typename ActionTree>
SimulationResults< ActionTree > run_simulation_for_results(ActionTree & tree, NetConfig & config,
    Sender & example_sender, unsigned int ticks_to_run, double interval) {

  unsigned int prng_seed = global_PRNG()();
  PRNG run_prng( prng_seed );
  SimulationResults< ActionTree > results( tree );

  results.set_prng_seed( prng_seed );
  results.set_tick_count( ticks_to_run );

  SimulationRunData & run_data = results.add_run_data( config, interval );
  Network<SenderGang<Sender, ExternalSwitchedSender<Sender>>,
    SenderGang<Sender, ExternalSwitchedSender<Sender>>> network1( example_sender, run_prng, config );
  network1.mutable_senders().mutable_gang1().mutable_sender(0).switch_on(network1.tickno());
  network1.run_simulation_with_logging( ticks_to_run, run_data, interval );

  return results;
}

template <typename ActionTree>
void serialize_to_file( SimulationResults<ActionTree> & results, string & datafilename )
{
  if ( datafilename.empty() ) return;

  ofstream datafile;
  datafile.open( datafilename );
  if ( datafile.is_open() ) {
    if ( !results.DNA().SerializeToOstream( &datafile ) ) {
      cerr << "Could not serialize to file " << datafilename << endl;
    } else {
      cerr << "Wrote simulation data to file " << datafilename << endl;
    }
  } else {
    cerr << "Could not open file " << datafilename << endl;
  }
  datafile.close();
}

int main( int argc, char *argv[] )
{
  WhiskerTree whiskers;
  FinTree fins;
  bool is_poisson = false;
  double link_ppt = 1.0;
  double delay = 100.0;
  unsigned int simulation_ticks = 1000000;
  double log_interval_ticks = 1000;
  double buffer_size = numeric_limits<unsigned int>::max();
  string datafilename;

  for ( int i = 1; i < argc; i++ ) {
    string arg( argv[ i ] );
    if ( arg.substr( 0, 7) == "sender=" ) {
        string sender_type( arg.substr( 7 ) );
        if ( sender_type == "poisson" ) {
          is_poisson = true;
          cerr << "Running poisson sender" << endl;
        }
    } else if ( arg.substr( 0, 3 ) == "if=" ) {
      string filename( arg.substr( 3 ) );
      ifstream file;
      file.open( filename );
      if ( !file.is_open() ) {
        cerr << "Could not open file " << filename << endl;
        exit( 1 );
      }

      if ( is_poisson ) {
        RemyBuffers::FinTree tree;
        if ( !tree.ParseFromIstream( &file ) ) {
          cerr << "Could not parse " << filename << endl;
          exit( 1 );
        }
        fins = FinTree( tree );
        print_tree< RemyBuffers::FinTree >(tree);
      } else {
        RemyBuffers::WhiskerTree tree;
        if ( !tree.ParseFromIstream( &file ) ) {
          cerr << "Could not parse " << filename << endl;
          exit( 1 );
        }
        whiskers = WhiskerTree( tree );
        print_tree< RemyBuffers::WhiskerTree >(tree);
      }

      file.close();

    } else if ( arg.substr( 0, 5 ) == "link=" ) {
      link_ppt = atof( arg.substr( 5 ).c_str() );
      cerr << "Setting link packets per ms to " << link_ppt << " ms" << endl;
    } else if ( arg.substr( 0, 4 ) == "rtt=" ) {
      delay = atof( arg.substr( 4 ).c_str() );
      cerr << "Setting delay to " << delay << endl;
    } else if ( arg.substr( 0, 5 ) == "time=" ) {
      simulation_ticks = stof( arg.substr( 5 ) ) * 1000;
      cerr << "Setting simulation length to " << simulation_ticks << " ms" << endl;
    } else if ( arg.substr( 0, 9 ) == "interval=" ) {
      log_interval_ticks = stof( arg.substr( 9 ) ) * 1000;
      cerr << "Setting logging interval to " << log_interval_ticks << " ms" << endl;
    } else if ( arg.substr( 0, 4 ) == "buf=" ) {
      if (arg.substr( 4 ) == "inf") {
        buffer_size = numeric_limits<unsigned int>::max();
      } else {
        buffer_size = stoi( arg.substr( 4 ) );
      }
    } else if ( arg.substr( 0, 3 ) == "of=" ) {
      datafilename = arg.substr( 3 );
      cerr << "Will write simulation data to " << datafilename << endl;
    }
  }

  if ( datafilename.empty() ) {
    cerr << "Warning: No data file supplied. Use of= option to get this to log data to a file." << endl;
  }

  NetConfig config;
  config.link_ppt = link_ppt;
  config.num_senders = 2;
  config.delay = delay;
  config.buffer_size = buffer_size;


  if ( is_poisson ) {
    SimulationResults<FinTree> results;
    Fish example_sender = Fish( fins, global_PRNG()(), true );
    results = run_simulation_for_results<Fish, FinTree>( fins, config, example_sender, simulation_ticks, log_interval_ticks );
    serialize_to_file( results, datafilename );
  } else {
    SimulationResults<WhiskerTree> results;
    Rat example_sender = Rat( whiskers, true );
    results = run_simulation_for_results<Rat, WhiskerTree>( whiskers, config, example_sender, simulation_ticks, log_interval_ticks );
    serialize_to_file( results, datafilename );
  }


  return 0;
}
