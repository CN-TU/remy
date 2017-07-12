#ifndef UNICORN_FARM_HH
#define UNICORN_FARM_HH

#include <vector>
#include "whisker.hh"

class UnicornFarm
{
private:
	PyObject* pModule;
	PyObject* pActionFunc;
	PyObject* pActionFunc;

	int create_thread();
	std::vector<int> get_action(int thread_id, std::vector<DataType> state);
	void put_reward(int thread_id, int reward, bool terminal);

public:
	static UnicornFarm& getInstance();
};

#endif

// TODO: Call Python base program, don't do anything. It's possible to create Unicorns from the UnicornFarm
// UnicornBreeders don't interact with the UnicornFarm, they only contain the Remy emulator's simulation.
// However, each UnicornBreeder owns a number of Unicorns, that he has to destroy when the simulation ends. 
// A new simulation is started, which creates new Unicorns in the UnicornBreeder. 
// So 1 UnicornFarm, a fixed number of UnicornBreeders and a variable number of Unicorns,
// their number depends on the number of UnicornBreeders and the number of Unicorns required in each UnicornBreeder. 
