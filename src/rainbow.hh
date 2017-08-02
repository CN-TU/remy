#ifndef RAINBOW_HH
#define RAINBOW_HH

#include <vector>
#include <mutex>
#include <Python.h>

typedef struct action_struct {
	double window_increment;
	double window_multiple;
	double intersend;
} action_struct;

class Rainbow
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
	PyObject* pSaveFunc;
	Rainbow();

public:
	Rainbow(Rainbow const&) = delete;
	void operator=(Rainbow const&) = delete;
	
	static Rainbow& getInstance();
	long unsigned int create_thread();
	action_struct get_action(const long unsigned int thread_id, const std::vector<double> state, const action_struct* action_to_put_struct);
	void put_reward(const long unsigned int thread_id, const double reward);
	void finish(const long unsigned int thread_id);
	void delete_thread(const long unsigned int thread_id);
	void save_session();
	void print_errors();
};

#endif
