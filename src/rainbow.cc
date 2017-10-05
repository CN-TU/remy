#include "rainbow.hh"
#include <thread>
#include <stddef.h>
#include <unistd.h>
#include <cstdio>
#include <cerrno>
#include <cmath>
#include <cstdlib>

using namespace std;

Rainbow& Rainbow::getInstance() {
	// FIXME: I guess one should check for a NULL pointer
	const char* cooperative_string = getenv("cooperative");
	const bool cooperative = (bool) strtoul(cooperative_string, NULL, 10L);
	static Rainbow instance(cooperative);
	return instance;
}

Rainbow& Rainbow::getInstance(const bool& cooperative) {
	static Rainbow instance(cooperative);
	return instance;
}

Rainbow::Rainbow(const bool& cooperative) :
	global_lock(),
	pModule(NULL),
	pActionFunc(NULL),
	pRewardFunc(NULL),
	pCreateFunc(NULL),
	pDeleteFunc(NULL),
	pFinishFunc(NULL),
	pSaveFunc(NULL),
	_training(true) {

	puts("Initializing Python interpreter");
	Py_Initialize();

	// char cwd[1024 * sizeof(char)];
	// if (getcwd(cwd, sizeof(cwd)) != NULL)
	// 	fprintf(stdout, "Current working dir: %s\n", cwd);
	// else
	// 	perror("getcwd() error");
	const char python_directory[] = "/repos/remy/async_deep_reinforce";
	const char* home_directory = getenv("HOME");
	char* search_path = new char[strlen(home_directory)+strlen(python_directory)+1];
	sprintf(search_path, "%s%s", home_directory, python_directory);
	// // FIXME: Hardcoded path to python stuff is NOT GOOD.
	// const char python_directory[] = "~/repos/remy/async_deep_reinforce";
	// const char* search_path = python_directory;
	printf("Current search path: %s\n", search_path);

	const char pModuleName[] = "a3c";
	const char pActionFuncName[] = "call_process_action";
	const char pRewardFuncName[] = "call_process_reward";
	const char pCreateFuncName[] = "create_training_thread";
	const char pDeleteFuncName[] = "delete_training_thread";
	const char pFinishFuncName[] = "call_process_finished";
	const char pSaveFuncName[] = "save_session";

	PyObject* path = PySys_GetObject("path");
	if (path == NULL) {
		PyErr_Print();
	}
	PyObject* pSearchPath = PyUnicode_FromString(search_path);
	if (pSearchPath == NULL) {
		PyErr_Print();
	}
	// FIXME: search_path doesn't get deleted anymore... but shouldn't actually matter...
	// delete search_path;
	PyList_Append(path, pSearchPath);
	Py_DECREF(pSearchPath);

	// size_t dummy_size = 1;
	// // FIXME: pArgString never gets freed.
	// wchar_t* pArgString = Py_DecodeLocale("", &dummy_size);

	wchar_t* pArgCooperativeString = NULL;
	if (cooperative) {
		const char* raw_string = "cooperative";
		size_t dummy_size = strlen(raw_string)+1;
		pArgCooperativeString = Py_DecodeLocale(raw_string, &dummy_size);
	} else {
		const char* raw_string = "independent";
		size_t dummy_size = strlen(raw_string)+1;
		pArgCooperativeString = Py_DecodeLocale(raw_string, &dummy_size);
	}
	// PySys_SetArgv(1, &pArgString);
	PySys_SetArgv(1, &pArgCooperativeString);
	pModule = PyImport_ImportModule(pModuleName);
	if (pModule == NULL) {
		PyErr_Print();
	}
	puts("Yeah, loaded the a3c module");

	pActionFunc = PyObject_GetAttrString(pModule, pActionFuncName);
	pRewardFunc = PyObject_GetAttrString(pModule, pRewardFuncName);
	pCreateFunc = PyObject_GetAttrString(pModule, pCreateFuncName);
	pDeleteFunc = PyObject_GetAttrString(pModule, pDeleteFuncName);
	pFinishFunc = PyObject_GetAttrString(pModule, pFinishFuncName);
	pSaveFunc = PyObject_GetAttrString(pModule, pSaveFuncName);
}

double Rainbow::get_action(const long unsigned int thread_id, const vector<double> state, const double& tickno, const double& window) {
	lock_guard<mutex> guard(global_lock);

	PyObject* pState = PyTuple_New(state.size());
	for (size_t i=0; i<state.size(); i++) {
		PyTuple_SetItem(pState, i, PyFloat_FromDouble(state[i]));
	}

	PyObject* pArgs = Py_BuildValue("(iOff)", (long) thread_id, pState, tickno, window);

	PyObject* pActionReturnValue = PyObject_CallObject(pActionFunc, pArgs);
	if (pActionReturnValue == NULL) {
		PyErr_Print();
	}

	double action = PyFloat_AsDouble(pActionReturnValue);
	// printf("action: %f\n", action);
	Py_DECREF(pActionReturnValue);
	Py_DECREF(pArgs);
	Py_DECREF(pState);

	return action;
}

void Rainbow::put_reward(const long unsigned int thread_id, const double reward_throughput, const double reward_delay, const double duration) {
	lock_guard<mutex> guard(global_lock);

	PyObject* pRewardArgs = Py_BuildValue("(ifff)", (long) thread_id, reward_throughput, reward_delay, duration);
	PyObject* pReturnValue = PyObject_CallObject(pRewardFunc, pRewardArgs);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pRewardArgs);
	Py_DECREF(pReturnValue);
}

long unsigned int Rainbow::create_thread(const double& delay_delta) {
	lock_guard<mutex> guard(global_lock);

	PyObject* pArgs = Py_BuildValue("(Of)", _training ? Py_True : Py_False, delay_delta);
	if (pArgs == NULL) {
		PyErr_Print();
	}

	PyObject* pThreadId = PyObject_CallObject(pCreateFunc, pArgs);
	if (pThreadId == NULL) {
		puts("Oh no, NULL value for create_thread");
		PyErr_Print();
	}
	long unsigned int thread_id = (int) PyLong_AsLong(pThreadId);
	Py_DECREF(pArgs);
	Py_DECREF(pThreadId);

	printf("%lu: Created training thread\n", thread_id);
	return thread_id;
}

void Rainbow::delete_thread(const long unsigned int thread_id) {
	lock_guard<mutex> guard(global_lock);

	PyObject* pThreadIdTuple = Py_BuildValue("(i)", (long) thread_id);
	PyObject* pReturnValue = PyObject_CallObject(pDeleteFunc, pThreadIdTuple);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pReturnValue);
	Py_DECREF(pThreadIdTuple);
}

void Rainbow::finish(const long unsigned int thread_id, size_t actions_to_remove, const double time_difference, const double window) {
	lock_guard<mutex> guard(global_lock);

	PyObject* pArgs = Py_BuildValue("(iiff)", (long) thread_id, (long) actions_to_remove, time_difference, window);
	if (pArgs == NULL) {
		PyErr_Print();
	}
	PyObject* pReturnValue = PyObject_CallObject(pFinishFunc, pArgs);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pArgs);
	Py_DECREF(pReturnValue);
}

void Rainbow::save_session() {
	lock_guard<mutex> guard(global_lock);
	puts("Saving session");

	PyErr_Print();

	PyObject* pReturnValue = PyObject_CallObject(pSaveFunc, NULL);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pReturnValue);

	puts("Saved session");
}
