#include "unicornfarm.hh"
#include <thread>
#include <stddef.h>
#include <unistd.h>
#include <cstdio>
#include <cerrno>
#include <cmath>
// #include <Python.h>

UnicornFarm& UnicornFarm::getInstance() {
	// Apparently in C++11 all that code is thread_safe...
	// PyGILState_STATE gstate; gstate = PyGILState_Ensure();

	static UnicornFarm instance;
	return instance;
}

UnicornFarm::UnicornFarm() : 
	global_lock(),
	pModule(NULL),
	pActionFunc(NULL),
	pRewardFunc(NULL),
	pCreateFunc(NULL),
	pDeleteFunc(NULL),
	pFinishFunc(NULL),
	pSaveFunc(NULL)

	{
	
	puts("Initializing Python interpreter");
	Py_Initialize();
	PyEval_InitThreads();
	// PyObject* pModuleName = PyUnicode_FromString("../async_deep_reinforce/a3c");
	// PyObject* pActionFuncName = PyUnicode_FromString("call_process_action");
	// PyObject* pRewardFuncName = PyUnicode_FromString("call_process_reward");
	// PyObject* pCreateFuncName = PyUnicode_FromString("create_training_thread");
	// PyObject* pDeleteFuncName = PyUnicode_FromString("delete_training_thread");
	// PyObject* pFinishFuncName = PyUnicode_FromString("call_process_finished");

	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

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
	// const char pTestModuleName[] = "python_embedding_test";
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

	// PyObject* pTestModule = PyImport_ImportModule(pTestModuleName);
	// if (pTestModule == NULL) {
	// 	PyErr_Print();
	// }
	// Py_DECREF(pTestModule);
	size_t dummy_size = 1;
	wchar_t* pEmptyString = Py_DecodeLocale("", &dummy_size);
	// const char argv[][1] = {""};
	PySys_SetArgv(1, &pEmptyString);
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

	// PyGILState_Release(gstate);
}

action_struct UnicornFarm::get_action(const long unsigned int thread_id, const std::vector<double> state) {
	std::lock_guard<std::mutex> guard(global_lock);
	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

	PyObject* pState = PyTuple_New(state.size());
	for (size_t i=0; i<state.size(); i++) {
		PyTuple_SetItem(pState, i, PyFloat_FromDouble(state[i]));
	}
	PyObject* pArgs = Py_BuildValue("(iO)", (long) thread_id, pState);

	PyObject* pActionArrayValue = PyObject_CallObject(pActionFunc, pArgs);
	if (pActionArrayValue == NULL) {
		PyErr_Print();
	}
	action_struct action = {
		PyFloat_AsDouble(PyTuple_GetItem(pActionArrayValue, 0)),
		PyFloat_AsDouble(PyTuple_GetItem(pActionArrayValue, 1)),
		PyFloat_AsDouble(PyTuple_GetItem(pActionArrayValue, 2))
	};
	// printf("%f, %f, %f\n", action.window_increment, action.window_multiple, action.intersend);
	Py_DECREF(pActionArrayValue);
	Py_DECREF(pArgs);	
	Py_DECREF(pState);

	// PyGILState_Release(gstate);
	return action;
}

void UnicornFarm::put_reward(const long unsigned int thread_id, const double reward) {
	std::lock_guard<std::mutex> guard(global_lock);
	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

	PyObject* pRewardArgs = Py_BuildValue("(if)", (long) thread_id, reward);
	PyObject* pReturnValue = PyObject_CallObject(pRewardFunc, pRewardArgs);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pRewardArgs);
	Py_DECREF(pReturnValue);

	// PyGILState_Release(gstate);
}

long unsigned int UnicornFarm::create_thread() {
	std::lock_guard<std::mutex> guard(global_lock);
	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

	// puts("Creating training thread");
	// printf("Do I hold the GIL? %d\n", PyGILState_Check());
	PyObject* pThreadId = PyObject_CallObject(pCreateFunc, NULL);
	if (pThreadId == NULL) {
		puts("Oh oh, NULL value for create_thread");
		PyErr_Print();
	}
	long unsigned int thread_id = (int) PyLong_AsLong(pThreadId);
	Py_DECREF(pThreadId);

	puts("Created training thread");
	// PyGILState_Release(gstate);
	return thread_id;
}

void UnicornFarm::delete_thread(const long unsigned int thread_id) {
	std::lock_guard<std::mutex> guard(global_lock);
	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

	PyObject* pThreadIdTuple = Py_BuildValue("(i)", (long) thread_id);
	PyObject* pReturnValue = PyObject_CallObject(pDeleteFunc, pThreadIdTuple);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}	
	Py_DECREF(pReturnValue);
	Py_DECREF(pThreadIdTuple);

	// PyGILState_Release(gstate);
}

void UnicornFarm::finish(const long unsigned int thread_id, const std::vector<double> state, const bool remove_last) {
	std::lock_guard<std::mutex> guard(global_lock);
	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

	PyObject* pState = PyTuple_New(state.size());
	for (size_t i=0; i<state.size(); i++) {
		PyTuple_SetItem(pState, i, PyFloat_FromDouble(state[i]));
	}
	PyObject* pArgs = Py_BuildValue("(iOp)", (long) thread_id, pState, remove_last);
	
	PyObject* pReturnValue = PyObject_CallObject(pFinishFunc, pArgs);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pArgs);	
	Py_DECREF(pState);
	Py_DECREF(pReturnValue);

	// PyGILState_Release(gstate);
}

void UnicornFarm::save_session() {
	std::lock_guard<std::mutex> guard(global_lock);
	printf("Saving session\n");
	// PyGILState_STATE gstate; 
	// gstate = PyGILState_Ensure();

	PyErr_Print();

	PyObject* pReturnValue = PyObject_CallObject(pSaveFunc, NULL);
	if (pReturnValue == NULL) {
		PyErr_Print();
	}
	Py_DECREF(pReturnValue);

	// PyGILState_Release(gstate);
	printf("Saved session\n");
}

void UnicornFarm::print_errors() {
	std::lock_guard<std::mutex> guard(global_lock);

	PyErr_Print();
}
