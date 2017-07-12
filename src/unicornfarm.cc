#include <Python.h>
#include "unicornfarm.hh"

UnicornFarm::getInstance() {
	static UnicornFarm instance;
	return instance;
}

UnicornFarm::UnicornFarm() {
	Py_Initialize();
	PyObject* pModuleName = PyString_FromString("../async_deep_reinforce/a3c.py");
	PyObject* pActionFuncName = PyString_FromString("call_process_action");
	PyObject* pRewardFuncName = PyString_FromString("call_process_reward");

	pModule = PyImport_Import(pModuleName);
	Py_DECREF(pModuleName);

	pActionFunc = PyObject_GetAttrString(pModule, pActionFuncName);
	Py_DECREF(pActionFuncName);

	pActionFunc = PyObject_GetAttrString(pModule, pActionFuncName);
	Py_DECREF(pRewardFuncName);
}

// TODO: Probably add a global lock here
std::vector<int> get_action(int thread_id, std::vector<DataType> state) {
	pArgs = PyTuple_New(argc - 3);
	PyObject* pActionArrayValue = PyObject_CallObject(pActionFunc);
}

// TODO: Probably add a global lock here
UnicornFarm::put_reward(int thread_id, int reward, bool terminal) {

}

// UnicornFarm::getUnicornFarm() {
// 	return UnicornFarm;
// }