#ifndef UNICORN_FARM_HH
#define UNICORN_FARM_HH

#include <vector>
#include "whisker.hh"
#include <mutex>

typedef struct action_struct {
	int window_increment,
	double window_multiple,
	double intersend
} action_struct;

class UnicornFarm
{
private:
	std::mutex global_lock;
	PyObject* pModule;
	PyObject* pActionFunc;
	PyObject* pRewardFunc;
	PyObject* pCreateFunc;
	PyObject* pDeleteFunc;
	UnicornFarm();

public:
	UnicornFarm(UnicornFarm const&) = delete;
	void operator=(UnicornFarm const&) = delete;
	static UnicornFarm& getInstance();
	int create_thread();
	action_struct get_action(const int thread_id, const std::vector<DataType> state);
	void put_reward(const int thread_id, const int reward);
	void finish(const int thread_id, const std::vector<DataType> state);
	void delete_thread(const int thread_id);
};

#endif
