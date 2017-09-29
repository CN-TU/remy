#include <fcntl.h>

#include "configrange.hh"
#include "unicornevaluator.hh"
#include "network.cc"
#include "random.hh"
#include "unicorn-templates.cc"

#include <algorithm>
#include <cstdlib>
#include <cmath>

UnicornEvaluator::UnicornEvaluator( const ConfigRangeUnicorn & range )
  : _prng_seed( global_PRNG()() ), /* freeze the PRNG seed for the life of this UnicornEvaluator */
    // _tick_count( range.simulation_ticks ),
    _configs(),
    _config_range(range)
{
  // printf("Creating UnicornEvaluator with random seed %u\n", _prng_seed);
  srand(_prng_seed);
  // add configs from every point in the cube of configs
  for (double simulation_ticks = range.simulation_ticks.low; simulation_ticks <= range.simulation_ticks.high; simulation_ticks += range.simulation_ticks.incr) {
    for (double link_ppt = range.link_ppt.low; link_ppt <= range.link_ppt.high; link_ppt += range.link_ppt.incr) {
      for (double rtt = range.rtt.high; rtt <= range.rtt.high; rtt += range.rtt.incr) {
        for (unsigned int senders = range.num_senders.low; senders <= range.num_senders.high; senders += range.num_senders.incr) {
          for (double on = range.mean_on_duration.low; on <= range.mean_on_duration.high; on += range.mean_on_duration.incr) {
            for (double off = range.mean_off_duration.low; off <= range.mean_off_duration.high; off += range.mean_off_duration.incr) {
              for ( double buffer_size = range.buffer_size.low; buffer_size <= range.buffer_size.high; buffer_size += range.buffer_size.incr) {
                for ( double loss_rate = range.stochastic_loss_rate.low; loss_rate <= range.stochastic_loss_rate.high; loss_rate += range.stochastic_loss_rate.incr) {
                  // _configs.push_back( NetConfig().set_link_ppt( link_ppt ).set_delay( rtt ).set_num_senders( senders ).set_on_duration( on ).set_off_duration(off).set_buffer_size( buffer_size ).set_stochastic_loss_rate( loss_rate ).set_simulation_ticks(round(rand() % ((int)simulation_ticks) +1)) );
                  _configs.push_back( NetConfig().set_link_ppt( link_ppt ).set_delay( rtt ).set_num_senders( senders ).set_on_duration( on ).set_off_duration(off).set_buffer_size( buffer_size ).set_stochastic_loss_rate( loss_rate ).set_simulation_ticks( simulation_ticks )) ;
                  if ( range.stochastic_loss_rate.isOne() ) { break; }
                }
                if ( range.buffer_size.isOne() ) { break; }
              }
              if ( range.mean_off_duration.isOne() ) { break; }
            }
            if ( range.mean_on_duration.isOne() ) { break; }
          }
          if ( range.num_senders.isOne() ) { break; }
        }
        if ( range.rtt.isOne() ) { break; }
      }
      if ( range.link_ppt.isOne() ) { break; }
    }
    if ( range.simulation_ticks.isOne() ) { break; }
  }
}

// template <typename T>
// ProblemBuffers::Problem UnicornEvaluator< T >::_ProblemSettings_DNA( void ) const
// {
//   ProblemBuffers::Problem ret;

//   ProblemBuffers::ProblemSettings settings;
//   settings.set_prng_seed( _prng_seed );
//   settings.set_tick_count( _tick_count );

//   ret.mutable_settings()->CopyFrom( settings );

//   for ( auto &x : _configs ) {
//     RemyBuffers::NetConfig *config = ret.add_configs();
//     *config = x.DNA();
//   }

//   return ret;
// }

// template <>
// ProblemBuffers::Problem UnicornEvaluator< WhiskerTree >::DNA( const WhiskerTree & whiskers ) const
// {
//   ProblemBuffers::Problem ret = _ProblemSettings_DNA();
//   ret.mutable_whiskers()->CopyFrom( whiskers.DNA() );

//   return ret;
// }

UnicornEvaluator::Outcome UnicornEvaluator::score(
             const unsigned int prng_seed,
             const vector<NetConfig> & configs,
             const ConfigRangeUnicorn& config_range)
{
  PRNG run_prng( prng_seed );

  /* run tests */
  UnicornEvaluator::Outcome the_outcome;

  vector<NetConfig> shuffled_configs(configs);
  std::shuffle(shuffled_configs.begin(), shuffled_configs.end(), run_prng);

  for ( auto &x : shuffled_configs ) {
    printf("Running for %.f ticks\n", x.simulation_ticks);
    /* run once */
    Network<SenderGang<Unicorn, ByteSwitchedSender<Unicorn>>,
      SenderGang<Unicorn, ByteSwitchedSender<Unicorn>>> network1( Unicorn(config_range.cooperative), run_prng, x );
    network1.run_simulation( x.simulation_ticks );

    the_outcome.score += network1.senders().utility();
    the_outcome.throughputs_delays.emplace_back( x, network1.senders().throughputs_delays() );
  }

  return the_outcome;
}

// template <>
// typename UnicornEvaluator< WhiskerTree >::Outcome UnicornEvaluator< WhiskerTree >::parse_problem_and_evaluate( const ProblemBuffers::Problem & problem )
// {
//   vector<NetConfig> configs;
//   for ( const auto &x : problem.configs() ) {
//     configs.emplace_back( x );
//   }

//   WhiskerTree run_whiskers = WhiskerTree( problem.whiskers() );

//   return UnicornEvaluator< WhiskerTree >::score( run_whiskers, problem.settings().prng_seed(),
// 			   configs, false, problem.settings().tick_count() );
// }

// template <typename T>
// AnswerBuffers::Outcome UnicornEvaluator< T >::Outcome::DNA( void ) const
// {
//   AnswerBuffers::Outcome ret;

//   for ( const auto & run : throughputs_delays ) {
//     AnswerBuffers::ThroughputsDelays *tp_del = ret.add_throughputs_delays();
//     tp_del->mutable_config()->CopyFrom( run.first.DNA() );

//     for ( const auto & x : run.second ) {
//       AnswerBuffers::SenderResults *results = tp_del->add_results();
//       results->set_throughput( x.first );
//       results->set_delay( x.second );
//     }
//   }

//   ret.set_score( score );

//   return ret;
// }

// template <typename T>
// UnicornEvaluator< T >::Outcome::Outcome( const AnswerBuffers::Outcome & dna )
//   : score( dna.score() ), throughputs_delays(), used_actions() {
//   for ( const auto &x : dna.throughputs_delays() ) {
//     vector< pair< double, double > > tp_del;
//     for ( const auto &result : x.results() ) {
//       tp_del.emplace_back( result.throughput(), result.delay() );
//     }

//     throughputs_delays.emplace_back( NetConfig( x.config() ), tp_del );
//   }
// }

UnicornEvaluator::Outcome UnicornEvaluator::score() const
{
  return score(_prng_seed, _configs, _config_range);
}
