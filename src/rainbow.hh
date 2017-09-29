#ifndef RAINBOW_HH
#define RAINBOW_HH

#include <vector>
#include <mutex>
#include <Python.h>

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
	Rainbow(const bool& cooperative);

public:
	bool _training;
	Rainbow(Rainbow const&) = delete;
	void operator=(Rainbow const&) = delete;

	static Rainbow& getInstance(const bool& cooperative);
	long unsigned int create_thread();
	double get_action(const long unsigned int thread_id, const std::vector<double> state);
	void put_reward(const long unsigned int thread_id, const double reward_throughput, const double reward_delay, const double duration);
	void finish(const long unsigned int thread_id, size_t actions_to_remove, const double time_difference, const double window);
	void delete_thread(const long unsigned int thread_id);
	void save_session();
};

#endif
