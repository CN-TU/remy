#include "unicornfarm.hh"
#include <thread>
#include <stddef.h>
#include <unistd.h>
#include <cstdio>
#include <cerrno>
// #include <Python.h>

UnicornFarm& UnicornFarm::getInstance() {
	// Apparently in C++11 all that code is thread_safe...
	// PyGILState_STATE gstate; gstate = PyGILState_Ensure();

	static UnicornFarm instance;
	return instance;
}

UnicornFarm::UnicornFarm() : 
	// global_lock(),
	pModule(NULL),
	pActionFunc(NULL),
	pRewardFunc(NULL),
	pCreateFunc(NULL),
	pDeleteFunc(NULL),
	pFinishFunc(NULL),
	pSaveFunc(NULL)

	{
	
	Py_Initialize();
	// PyObject* pModuleName = PyUnicode_FromString("../async_deep_reinforce/a3c");
	// PyObject* pActionFuncName = PyUnicode_FromString("call_process_action");
	// PyObject* pRewardFuncName = PyUnicode_FromString("call_process_reward");
	// PyObject* pCreateFuncName = PyUnicode_FromString("create_training_thread");
	// PyObject* pDeleteFuncName = PyUnicode_FromString("delete_training_thread");
	// PyObject* pFinishFuncName = PyUnicode_FromString("call_process_finished");

	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	char cwd[1024];
	if (getcwd(cwd, sizeof(cwd)) != NULL)
		fprintf(stdout, "Current working dir: %s\n", cwd);
	else
		perror("getcwd() error");
	const char python_directory[] = "/async_deep_reinforce";
	char* search_path = new char[strlen(cwd)+strlen(python_directory)+1];
	sprintf(search_path, "%s%s", cwd, python_directory);
	printf("%s\n", search_path);

	const char pModuleName[] = "a3c";
	const char pActionFuncName[] = "call_process_action";
	const char pRewardFuncName[] = "call_process_reward";
	const char pCreateFuncName[] = "create_training_thread";
	const char pDeleteFuncName[] = "delete_training_thread";
	const char pFinishFuncName[] = "call_process_finished";
	const char pSaveFuncName[] = "save_session";

	// PyObject* pLoadModule = PyImport_ImportModule(pLoadModuleName);
	// printf("pLoadModule %zd\n", (size_t) pLoadModule);
	// if (pLoadModule == NULL) {
	// 	PyErr_Print();
	// }
	PyObject* path = PySys_GetObject("path");
	if (path == NULL) {
		PyErr_Print();
	}
	PyObject* pSearchPath = PyUnicode_FromString(search_path);
	if (pSearchPath == NULL) {
		PyErr_Print();
	}
	delete search_path;
	PyList_Append(path, pSearchPath);
	Py_DECREF(pSearchPath);

	pModule = PyImport_ImportModule(pModuleName);
	printf("pModule %zd\n", (size_t) pModule);
	if (pModule == NULL) {
		PyErr_Print();
	}
	// Py_DECREF(pModuleName);

	pActionFunc = PyObject_GetAttrString(pModule, pActionFuncName);
	// Py_DECREF(pActionFuncName);

	pRewardFunc = PyObject_GetAttrString(pModule, pRewardFuncName);
	// Py_DECREF(pRewardFuncName);

	pCreateFunc = PyObject_GetAttrString(pModule, pCreateFuncName);
	// Py_DECREF(pCreateFuncName);

	pDeleteFunc = PyObject_GetAttrString(pModule, pDeleteFuncName);
	// Py_DECREF(pDeleteFuncName);

	pFinishFunc = PyObject_GetAttrString(pModule, pFinishFuncName);
	// Py_DECREF(pFinishFuncName);

	pSaveFunc = PyObject_GetAttrString(pModule, pSaveFuncName);

	PyGILState_Release(gstate);
}

action_struct UnicornFarm::get_action(const long unsigned int thread_id, const std::vector<double> state) {
	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	PyObject* pState = PyTuple_New(state.size());
	for (size_t i=0; i<state.size(); i++) {
		PyTuple_SetItem(pState, i, PyFloat_FromDouble(state[i]));
	}
	PyObject* pArgs = Py_BuildValue("(i0)", (long) thread_id, pState);
	PyObject* pActionArrayValue = PyObject_CallObject(pActionFunc, pArgs);
	action_struct action = {
		(unsigned long) PyLong_AsLong(PyTuple_GetItem(pActionArrayValue, 0)),
		PyFloat_AsDouble(PyTuple_GetItem(pActionArrayValue, 1)),
		PyFloat_AsDouble(PyTuple_GetItem(pActionArrayValue, 2))
	};
	Py_DECREF(pActionArrayValue);
	Py_DECREF(pArgs);	
	Py_DECREF(pState);

	PyGILState_Release(gstate);
	return action;
}

void UnicornFarm::put_reward(const long unsigned int thread_id, const int reward) {
	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	PyObject* pRewardArgs = Py_BuildValue("(ii)", (long) thread_id, (long) reward);
	PyObject* pReturnValue = PyObject_CallObject(pRewardFunc, pRewardArgs);
	Py_DECREF(pRewardArgs);
	Py_DECREF(pReturnValue);

	PyGILState_Release(gstate);
}

long unsigned int UnicornFarm::create_thread() {
	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	PyObject* pThreadId = PyObject_CallObject(pCreateFunc, NULL);
	long unsigned int thread_id = (int) PyLong_AsLong(pThreadId);
	Py_DECREF(pThreadId);

	PyGILState_Release(gstate);
	return thread_id;
}

void UnicornFarm::delete_thread(const long unsigned int thread_id) {
	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	PyObject* pThreadIdTuple = Py_BuildValue("(i)", (long) thread_id);
	PyObject* pReturnValue = PyObject_CallObject(pDeleteFunc, pThreadIdTuple);
	Py_DECREF(pReturnValue);
	Py_DECREF(pThreadIdTuple);

	PyGILState_Release(gstate);
}

void UnicornFarm::finish(const long unsigned int thread_id, const std::vector<double> state) {
	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	PyObject* pState = PyTuple_New(state.size());
	for (size_t i=0; i<state.size(); i++) {
		PyTuple_SetItem(pState, i, PyFloat_FromDouble(state[i]));
	}
	PyObject* pArgs = Py_BuildValue("(i0)", (long) thread_id, pState);
	PyObject* pReturnValue = PyObject_CallObject(pFinishFunc, pArgs);
	Py_DECREF(pArgs);	
	Py_DECREF(pState);
	Py_DECREF(pReturnValue);

	PyGILState_Release(gstate);
}

void UnicornFarm::save_session() {
	PyGILState_STATE gstate; 
	gstate = PyGILState_Ensure();

	PyObject* pReturnValue = PyObject_CallObject(pSaveFunc, NULL);
	Py_DECREF(pReturnValue);

	PyGILState_Release(gstate);
}
