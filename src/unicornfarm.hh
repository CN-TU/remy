#ifndef UNICORN_FARM_HH
#define UNICORN_FARM_HH

#include <vector>
#include <mutex>
#include <Python.h>

typedef struct action_struct {
	long unsigned int window_increment;
	double window_multiple;
	double intersend;
} action_struct;

class UnicornFarm
{
private:
	// FIXME: Replace global_lock by Python's GIL
	std::mutex global_lock;
	PyObject* pModule;
	PyObject* pActionFunc;
	PyObject* pRewardFunc;
	PyObject* pCreateFunc;
	PyObject* pDeleteFunc;
	PyObject* pFinishFunc;
	UnicornFarm();

public:
	UnicornFarm(UnicornFarm const&) = delete;
	void operator=(UnicornFarm const&) = delete;
	
	static UnicornFarm& getInstance();
	long unsigned int create_thread();
	action_struct get_action(const long unsigned int thread_id, const std::vector<double> state);
	void put_reward(const long unsigned int thread_id, const int reward);
	void finish(const long unsigned int thread_id, const std::vector<double> state);
	void delete_thread(const long unsigned int thread_id);
};

#endif
