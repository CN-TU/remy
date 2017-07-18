#include <Python.h>
#include "unicornfarm.hh"
#include <thread>

UnicornFarm::getInstance() {
	std::unique_lock<std::mutex> lock(global_lock);

	static UnicornFarm instance;
	return instance;
}

UnicornFarm::UnicornFarm() {
	Py_Initialize();
	PyObject* pModuleName = PyString_FromString("../async_deep_reinforce/a3c.py");
	PyObject* pActionFuncName = PyString_FromString("call_process_action");
	PyObject* pRewardFuncName = PyString_FromString("call_process_reward");
	PyObject* pCreateFuncName = PyString_FromString("create_training_thread");
	PyObject* pDeleteFuncName = PyString_FromString("delete_training_thread");
	PyObject* pFinishFuncName = PyString_FromString("call_process_finished");	

	pModule = PyImport_Import(pModuleName);
	Py_DECREF(pModuleName);

	pActionFunc = PyObject_GetAttrString(pModule, pActionFuncName);
	Py_DECREF(pActionFuncName);

	pRewardFunc = PyObject_GetAttrString(pModule, pRewardFuncName);
	Py_DECREF(pRewardFuncName);

	pCreateFunc = PyObject_GetAttrString(pModule, pCreateFuncName);
	Py_DECREF(pCreateFuncName);

	pDeleteFunc = PyObject_GetAttrString(pModule, pDeleteFuncName);
	Py_DECREF(pDeleteFuncName);

	pFinishFunc = PyObject_GetAttrString(pModule, pFinishFuncName);
	Py_DECREF(pFinishFunc);
}

std::vector<int> UnicornFarm::get_action(const int thread_id, const std::vector<DataType> state) {
	std::unique_lock<std::mutex> lock(global_lock);

	PyObject* pArgs = PyTuple_New(state.size());
	for (auto i; i<state.size(); i++) {
		PyTuple_SetItem(pArgs, i, PyFloat_FromDouble(state[i]));
	}
	PyObject* pActionArrayValue = PyObject_CallObject(pActionFunc, pArgs);
	action_struct action = {
		PyTuple_GetItem(pActionArrayValue, 0),
		PyTuple_GetItem(pActionArrayValue, 1),
		PyTuple_GetItem(pActionArrayValue, 2)
	};
	Py_DECREF(pActionArrayValue);
	Py_DECREF(pArgs);
	return actions;
}

void UnicornFarm::put_reward(const int thread_id, const int reward) {
	std::unique_lock<std::mutex> lock(global_lock);

	PyObject* pRewardArgs = Py_BuildValue("(ii)", (long) thread_id);
	PyObject* pReturnValue = PyObject_CallObject(pRewardFunc, pRewardArgs);
	Py_DECREF(pRewardArgs);
	Py_DECREF(pReturnValue);
}

int UnicornFarm::create_thread() {
	std::unique_lock<std::mutex> lock(global_lock);

	PyObject* pThreadId = PyObject_CallObject(pCreateFunc, NULL);
	int thread_id = (int) PyLong_asLong(pThreadId);
	Py_DECREF(pthread_id);
	return pThreadId;
}

void UnicornFarm::delete_thread(const int thread_id) {
	std::unique_lock<std::mutex> lock(global_lock);

	PyObject* pThreadIdTuple = Py_BuildValue("(i)", (long) thread_id);
	PyObject* pReturnValue = PyObject_CallObject(pThreadIdTuple);
	Py_DECREF(pReturnValue);
	Py_DECREF(pThreadIdTuple);
}

void UnicornFarm::finish(const int thread_id, const std::vector<DataType> state) {
	std::unique_lock<std::mutex> lock(global_lock);

	PyObject* pArgs = PyTuple_New(state.size());
	for (auto i; i<state.size(); i++) {
		PyTuple_SetItem(pArgs, i, PyFloat_FromDouble(state[i]));
	}
	PyObject* pReturnValue = PyObject_CallObject(pFinishFunc, pArgs);
	Py_DECREF(pReturnValue);
}