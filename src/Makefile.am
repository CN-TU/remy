AM_CPPFLAGS = $(CXX11_FLAGS) -I../protobufs
AM_CXXFLAGS = $(PICKY_CXXFLAGS) -g $(PYTHON_CFLAGS)
LDADD = ../protobufs/libremyprotos.a -lm $(protobuf_LIBS) $(PYTHON_LIBS)

#bin_PROGRAMS = remy remy-unicorn remy-poisson sender-runner sender-logger scoring-example configuration configuration-unicorn inspect-config inspect-simulationsdata
bin_PROGRAMS = remy-unicorn configuration-unicorn

common_source = delay.hh evaluator.cc evaluator.hh                 \
	exponential.hh link.hh link-templates.cc stochastic-loss.hh    \
	memory.cc memory.hh memoryrange.cc memoryrange.hh              \
	network.cc network.hh packet.hh poisson.hh                     \
	random.cc random.hh rat.cc rat.hh rat-templates.cc             \
	receiver.cc receiver.hh sendergang.cc sendergang.hh            \
	senderdatapoint.hh                                             \
	sendergangofgangs.cc sendergangofgangs.hh                      \
	utility.hh whisker.cc whisker.hh whiskertree.cc whiskertree.hh \
	aimd-templates.cc aimd.cc aimd.hh                              \
	configrange.hh configrange.cc                                  \
	simulationresults.hh simulationresults.cc                      \
    action.hh fin.hh fin.cc fintree.cc fintree.hh  	               \
    fish.hh fish.cc fish-templates.cc unicorn-templates.cc

noinst_LIBRARIES = libremycore.a
libremycore_a_SOURCES = $(common_source)

remy_SOURCES = $(common_source) remy.cc ratbreeder.cc ratbreeder.hh \
	breeder.cc breeder.hh

remy_unicorn_SOURCES = $(common_source) remy-unicorn.cc unicornbreeder.cc unicornbreeder.hh \
	breeder.cc breeder.hh rainbow.cc rainbow.hh unicornevaluator.cc unicornevaluator.hh \
	unicorn.hh unicorn.cc

remy_poisson_SOURCES = $(common_source) remy-poisson.cc breeder.cc breeder.hh fishbreeder.cc fishbreeder.hh

sender_runner_SOURCES = $(common_source) sender-runner.cc

sender_logger_SOURCES = $(common_source) sender-logger.cc

scoring_example_SOURCES = $(common_source) scoring-example.cc

configuration_SOURCES = input-configrange.cc

configuration_unicorn_SOURCES = input-configrange-unicorn.cc

inspect_config_SOURCES = inspect-configrange.cc

inspect_simulationsdata_SOURCES = inspect-simulationsdata.cc

