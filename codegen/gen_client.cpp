// Generated code.

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
                
#include <cstring>
#include <string>
#include <unordered_map>

#include "gen_api.h"

extern int rpc_start_request(const unsigned int request);
extern int rpc_write(const void *data, const size_t size);
extern int rpc_read(void *data, const size_t size);
extern int rpc_wait_for_response(const unsigned int request_id);
extern int rpc_end_request(void *return_value, const unsigned int request_id);

/**
* Initialize NVML, but don't initialize any GPUs yet.
*
* \note nvmlInit_v3 introduces a "flags" argument, that allows passing boolean values
*       modifying the behaviour of nvmlInit().
* \note In NVML 5.319 new nvmlInit_v2 has replaced nvmlInit"_v1" (default in NVML 4.304 and older) that
*       did initialize all GPU devices in the system.
*
* This allows NVML to communicate with a GPU
* when other GPUs in the system are unstable or in a bad state.  When using this API, GPUs are
* discovered and initialized in nvmlDeviceGetHandleBy* functions instead.
*
* \note To contrast nvmlInit_v2 with nvmlInit"_v1", NVML 4.304 nvmlInit"_v1" will fail when any detected GPU is in
*       a bad or unstable state.
*
* For all products.
*
* This method, should be called once before invoking any other methods in the library.
* A reference count of the number of initializations is maintained.  Shutdown only occurs
* when the reference count reaches zero.
*
* @return
*         - \ref NVML_SUCCESS                   if NVML has been properly initialized
*         - \ref NVML_ERROR_DRIVER_NOT_LOADED   if NVIDIA driver is not running
*         - \ref NVML_ERROR_NO_PERMISSION       if NVML does not have permission to talk to the driver
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlInit_v2() {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlInit_v2);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* nvmlInitWithFlags is a variant of nvmlInit(), that allows passing a set of boolean values
*       modifying the behaviour of nvmlInit().
*       Other than the "flags" parameter it is completely similar to \ref nvmlInit_v2.
*
* For all products.
*
* @param flags                                 behaviour modifier flags
*
* @return
*         - \ref NVML_SUCCESS                   if NVML has been properly initialized
*         - \ref NVML_ERROR_DRIVER_NOT_LOADED   if NVIDIA driver is not running
*         - \ref NVML_ERROR_NO_PERMISSION       if NVML does not have permission to talk to the driver
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlInitWithFlags(unsigned int flags) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlInitWithFlags);
    if (request_id < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Shut down NVML by releasing all GPU resources previously allocated with \ref nvmlInit_v2().
*
* For all products.
*
* This method should be called after NVML work is done, once for each call to \ref nvmlInit_v2()
* A reference count of the number of initializations is maintained.  Shutdown only occurs
* when the reference count reaches zero.  For backwards compatibility, no error is reported if
* nvmlShutdown() is called more times than nvmlInit().
*
* @return
*         - \ref NVML_SUCCESS                 if NVML has been properly shut down
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlShutdown() {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlShutdown);
    if (request_id < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the version of the system's graphics driver.
*
* For all products.
*
* The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
* (including the NULL terminator).  See \ref nvmlConstants::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
*
* @param version                              Reference in which to return the version identifier
* @param length                               The maximum allowed length of the string returned in \a version
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*/
nvmlReturn_t nvmlSystemGetDriverVersion(char* version, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetDriverVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(version, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the version of the NVML library.
*
* For all products.
*
* The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
* (including the NULL terminator).  See \ref nvmlConstants::NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE.
*
* @param version                              Reference in which to return the version identifier
* @param length                               The maximum allowed length of the string returned in \a version
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*/
nvmlReturn_t nvmlSystemGetNVMLVersion(char* version, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetNVMLVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(version, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the version of the CUDA driver.
*
* For all products.
*
* The CUDA driver version returned will be retrieved from the currently installed version of CUDA.
* If the cuda library is not found, this function will return a known supported version number.
*
* @param cudaDriverVersion                    Reference in which to return the version identifier
*
* @return
*         - \ref NVML_SUCCESS                 if \a cudaDriverVersion has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a cudaDriverVersion is NULL
*/
nvmlReturn_t nvmlSystemGetCudaDriverVersion(int* cudaDriverVersion) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion);
    if (request_id < 0 ||
        rpc_write(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the version of the CUDA driver from the shared library.
*
* For all products.
*
* The returned CUDA driver version by calling cuDriverGetVersion()
*
* @param cudaDriverVersion                    Reference in which to return the version identifier
*
* @return
*         - \ref NVML_SUCCESS                  if \a cudaDriverVersion has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a cudaDriverVersion is NULL
*         - \ref NVML_ERROR_LIBRARY_NOT_FOUND  if \a libcuda.so.1 or libcuda.dll is not found
*         - \ref NVML_ERROR_FUNCTION_NOT_FOUND if \a cuDriverGetVersion() is not found in the shared library
*/
nvmlReturn_t nvmlSystemGetCudaDriverVersion_v2(int* cudaDriverVersion) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetCudaDriverVersion_v2);
    if (request_id < 0 ||
        rpc_write(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cudaDriverVersion, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets name of the process with provided process id
*
* For all products.
*
* Returned process name is cropped to provided length.
* name string is encoded in ANSI.
*
* @param pid                                  The identifier of the process
* @param name                                 Reference in which to return the process name
* @param length                               The maximum allowed length of the string returned in \a name
*
* @return
*         - \ref NVML_SUCCESS                 if \a name has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a name is NULL or \a length is 0.
*         - \ref NVML_ERROR_NOT_FOUND         if process doesn't exists
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlSystemGetProcessName(unsigned int pid, char* name, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetProcessName);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(name, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the number of units in the system.
*
* For S-class products.
*
* @param unitCount                            Reference in which to return the number of units
*
* @return
*         - \ref NVML_SUCCESS                 if \a unitCount has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unitCount is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlUnitGetCount(unsigned int* unitCount) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetCount);
    if (request_id < 0 ||
        rpc_write(unitCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unitCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Acquire the handle for a particular unit, based on its index.
*
* For S-class products.
*
* Valid indices are derived from the \a unitCount returned by \ref nvmlUnitGetCount().
*   For example, if \a unitCount is 2 the valid indices are 0 and 1, corresponding to UNIT 0 and UNIT 1.
*
* The order in which NVML enumerates units has no guarantees of consistency between reboots.
*
* @param index                                The index of the target unit, >= 0 and < \a unitCount
* @param unit                                 Reference in which to return the unit handle
*
* @return
*         - \ref NVML_SUCCESS                 if \a unit has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a index is invalid or \a unit is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlUnitGetHandleByIndex(unsigned int index, nvmlUnit_t* unit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the static information associated with a unit.
*
* For S-class products.
*
* See \ref nvmlUnitInfo_t for details on available unit info.
*
* @param unit                                 The identifier of the target unit
* @param info                                 Reference in which to return the unit information
*
* @return
*         - \ref NVML_SUCCESS                 if \a info has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a info is NULL
*/
nvmlReturn_t nvmlUnitGetUnitInfo(nvmlUnit_t unit, nvmlUnitInfo_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetUnitInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlUnitInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the LED state associated with this unit.
*
* For S-class products.
*
* See \ref nvmlLedState_t for details on allowed states.
*
* @param unit                                 The identifier of the target unit
* @param state                                Reference in which to return the current LED state
*
* @return
*         - \ref NVML_SUCCESS                 if \a state has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a state is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlUnitSetLedState()
*/
nvmlReturn_t nvmlUnitGetLedState(nvmlUnit_t unit, nvmlLedState_t* state) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetLedState);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(state, sizeof(nvmlLedState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the PSU stats for the unit.
*
* For S-class products.
*
* See \ref nvmlPSUInfo_t for details on available PSU info.
*
* @param unit                                 The identifier of the target unit
* @param psu                                  Reference in which to return the PSU information
*
* @return
*         - \ref NVML_SUCCESS                 if \a psu has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a psu is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlUnitGetPsuInfo(nvmlUnit_t unit, nvmlPSUInfo_t* psu) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetPsuInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(psu, sizeof(nvmlPSUInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the temperature readings for the unit, in degrees C.
*
* For S-class products.
*
* Depending on the product, readings may be available for intake (type=0),
* exhaust (type=1) and board (type=2).
*
* @param unit                                 The identifier of the target unit
* @param type                                 The type of reading to take
* @param temp                                 Reference in which to return the intake temperature
*
* @return
*         - \ref NVML_SUCCESS                 if \a temp has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit or \a type is invalid or \a temp is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlUnitGetTemperature(nvmlUnit_t unit, unsigned int type, unsigned int* temp) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetTemperature);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&type, sizeof(unsigned int)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the fan speed readings for the unit.
*
* For S-class products.
*
* See \ref nvmlUnitFanSpeeds_t for details on available fan speed info.
*
* @param unit                                 The identifier of the target unit
* @param fanSpeeds                            Reference in which to return the fan speed information
*
* @return
*         - \ref NVML_SUCCESS                 if \a fanSpeeds has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid or \a fanSpeeds is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlUnitGetFanSpeedInfo(nvmlUnit_t unit, nvmlUnitFanSpeeds_t* fanSpeeds) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetFanSpeedInfo);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fanSpeeds, sizeof(nvmlUnitFanSpeeds_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the set of GPU devices that are attached to the specified unit.
*
* For S-class products.
*
* The \a deviceCount argument is expected to be set to the size of the input \a devices array.
*
* @param unit                                 The identifier of the target unit
* @param deviceCount                          Reference in which to provide the \a devices array size, and
*                                             to return the number of attached GPU devices
* @param devices                              Reference in which to return the references to the attached GPU devices
*
* @return
*         - \ref NVML_SUCCESS                 if \a deviceCount and \a devices have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a deviceCount indicates that the \a devices array is too small
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit is invalid, either of \a deviceCount or \a devices is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlUnitGetDevices(nvmlUnit_t unit, unsigned int* deviceCount, nvmlDevice_t* devices) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitGetDevices);
    if (request_id < 0 ||
        rpc_write(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(devices, *deviceCount * sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_read(devices, *deviceCount * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the IDs and firmware versions for any Host Interface Cards (HICs) in the system.
*
* For S-class products.
*
* The \a hwbcCount argument is expected to be set to the size of the input \a hwbcEntries array.
* The HIC must be connected to an S-class system for it to be reported by this function.
*
* @param hwbcCount                            Size of hwbcEntries array
* @param hwbcEntries                          Array holding information about hwbc
*
* @return
*         - \ref NVML_SUCCESS                 if \a hwbcCount and \a hwbcEntries have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if either \a hwbcCount or \a hwbcEntries is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a hwbcCount indicates that the \a hwbcEntries array is too small
*/
nvmlReturn_t nvmlSystemGetHicVersion(unsigned int* hwbcCount, nvmlHwbcEntry_t* hwbcEntries) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetHicVersion);
    if (request_id < 0 ||
        rpc_write(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_write(hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(hwbcCount, sizeof(unsigned int)) < 0 ||
        rpc_read(hwbcEntries, *hwbcCount * sizeof(nvmlHwbcEntry_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the number of compute devices in the system. A compute device is a single GPU.
*
* For all products.
*
* Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system
*       even if nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for such device.
*       Update your code to handle this error, or use NVML 4.304 or older nvml header file.
*       For backward binary compatibility reasons _v1 version of the API is still present in the shared
*       library.
*       Old _v1 version of nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
*
* @param deviceCount                          Reference in which to return the number of accessible devices
*
* @return
*         - \ref NVML_SUCCESS                 if \a deviceCount has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a deviceCount is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int* deviceCount) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCount_v2);
    if (request_id < 0 ||
        rpc_write(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get attributes (engine counts etc.) for the given NVML device handle.
*
* @note This API currently only supports MIG device handles.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               NVML device handle
* @param attributes                           Device attributes
*
* @return
*        - \ref NVML_SUCCESS                  if \a device attributes were successfully retrieved
*        - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device handle is invalid
*        - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*        - \ref NVML_ERROR_NOT_SUPPORTED      if this query is not supported by the device
*        - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetAttributes_v2(nvmlDevice_t device, nvmlDeviceAttributes_t* attributes) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAttributes_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(attributes, sizeof(nvmlDeviceAttributes_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Acquire the handle for a particular device, based on its index.
*
* For all products.
*
* Valid indices are derived from the \a accessibleDevices count returned by
*   \ref nvmlDeviceGetCount_v2(). For example, if \a accessibleDevices is 2 the valid indices
*   are 0 and 1, corresponding to GPU 0 and GPU 1.
*
* The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it
*   is recommended that devices be looked up by their PCI ids or UUID. See
*   \ref nvmlDeviceGetHandleByUUID() and \ref nvmlDeviceGetHandleByPciBusId_v2().
*
* Note: The NVML index may not correlate with other APIs, such as the CUDA device index.
*
* Starting from NVML 5, this API causes NVML to initialize the target GPU
* NVML may initialize additional GPUs if:
*  - The target GPU is an SLI slave
*
* Note: New nvmlDeviceGetCount_v2 (default in NVML 5.319) returns count of all devices in the system
*       even if nvmlDeviceGetHandleByIndex_v2 returns NVML_ERROR_NO_PERMISSION for such device.
*       Update your code to handle this error, or use NVML 4.304 or older nvml header file.
*       For backward binary compatibility reasons _v1 version of the API is still present in the shared
*       library.
*       Old _v1 version of nvmlDeviceGetCount doesn't count devices that NVML has no permission to talk to.
*
*       This means that nvmlDeviceGetHandleByIndex_v2 and _v1 can return different devices for the same index.
*       If you don't touch macros that map old (_v1) versions to _v2 versions at the top of the file you don't
*       need to worry about that.
*
* @param index                                The index of the target GPU, >= 0 and < \a accessibleDevices
* @param device                               Reference in which to return the device handle
*
* @return
*         - \ref NVML_SUCCESS                  if \a device has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a index is invalid or \a device is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
*         - \ref NVML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
*         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*
* @see nvmlDeviceGetIndex
* @see nvmlDeviceGetCount
*/
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t* device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByIndex_v2);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Acquire the handle for a particular device, based on its board serial number.
*
* For Fermi &tm; or newer fully supported devices.
*
* This number corresponds to the value printed directly on the board, and to the value returned by
*   \ref nvmlDeviceGetSerial().
*
* @deprecated Since more than one GPU can exist on a single board this function is deprecated in favor
*             of \ref nvmlDeviceGetHandleByUUID.
*             For dual GPU boards this function will return NVML_ERROR_INVALID_ARGUMENT.
*
* Starting from NVML 5, this API causes NVML to initialize the target GPU
* NVML may initialize additional GPUs as it searches for the target GPU
*
* @param serial                               The board serial number of the target GPU
* @param device                               Reference in which to return the device handle
*
* @return
*         - \ref NVML_SUCCESS                  if \a device has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a serial is invalid, \a device is NULL or more than one
*                                              device has the same serial (dual GPU boards)
*         - \ref NVML_ERROR_NOT_FOUND          if \a serial does not match a valid device on the system
*         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
*         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
*         - \ref NVML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*
* @see nvmlDeviceGetSerial
* @see nvmlDeviceGetHandleByUUID
*/
nvmlReturn_t nvmlDeviceGetHandleBySerial(const char* serial, nvmlDevice_t* device) {
    nvmlReturn_t return_value;
    int serial_size = std::strlen(serial) + 1;
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleBySerial);
    if (request_id < 0 ||
        rpc_write(&serial_size, sizeof(int)) < 0 ||
        rpc_write(serial, serial_size * sizeof(char)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Acquire the handle for a particular device, based on its globally unique immutable UUID associated with each device.
*
* For all products.
*
* @param uuid                                 The UUID of the target GPU or MIG instance
* @param device                               Reference in which to return the device handle or MIG device handle
*
* Starting from NVML 5, this API causes NVML to initialize the target GPU
* NVML may initialize additional GPUs as it searches for the target GPU
*
* @return
*         - \ref NVML_SUCCESS                  if \a device has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a uuid is invalid or \a device is null
*         - \ref NVML_ERROR_NOT_FOUND          if \a uuid does not match a valid device on the system
*         - \ref NVML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
*         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
*         - \ref NVML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*
* @see nvmlDeviceGetUUID
*/
nvmlReturn_t nvmlDeviceGetHandleByUUID(const char* uuid, nvmlDevice_t* device) {
    nvmlReturn_t return_value;
    int uuid_size = std::strlen(uuid) + 1;
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByUUID);
    if (request_id < 0 ||
        rpc_write(&uuid_size, sizeof(int)) < 0 ||
        rpc_write(uuid, uuid_size * sizeof(char)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Acquire the handle for a particular device, based on its PCI bus id.
*
* For all products.
*
* This value corresponds to the nvmlPciInfo_t::busId returned by \ref nvmlDeviceGetPciInfo_v3().
*
* Starting from NVML 5, this API causes NVML to initialize the target GPU
* NVML may initialize additional GPUs if:
*  - The target GPU is an SLI slave
*
* \note NVML 4.304 and older version of nvmlDeviceGetHandleByPciBusId"_v1" returns NVML_ERROR_NOT_FOUND
*       instead of NVML_ERROR_NO_PERMISSION.
*
* @param pciBusId                             The PCI bus id of the target GPU
* @param device                               Reference in which to return the device handle
*
* @return
*         - \ref NVML_SUCCESS                  if \a device has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a pciBusId is invalid or \a device is NULL
*         - \ref NVML_ERROR_NOT_FOUND          if \a pciBusId does not match a valid device on the system
*         - \ref NVML_ERROR_INSUFFICIENT_POWER if the attached device has improperly attached external power cables
*         - \ref NVML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
*         - \ref NVML_ERROR_IRQ_ISSUE          if NVIDIA kernel detected an interrupt issue with the attached GPUs
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char* pciBusId, nvmlDevice_t* device) {
    nvmlReturn_t return_value;
    int pciBusId_size = std::strlen(pciBusId) + 1;
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHandleByPciBusId_v2);
    if (request_id < 0 ||
        rpc_write(&pciBusId_size, sizeof(int)) < 0 ||
        rpc_write(pciBusId, pciBusId_size * sizeof(char)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the name of this device.
*
* For all products.
*
* The name is an alphanumeric string that denotes a particular product, e.g. Tesla &tm; C2070. It will not
* exceed 96 characters in length (including the NULL terminator).  See \ref
* nvmlConstants::NVML_DEVICE_NAME_V2_BUFFER_SIZE.
*
* When used with MIG device handles the API returns MIG device names which can be used to identify devices
* based on their attributes.
*
* @param device                               The identifier of the target device
* @param name                                 Reference in which to return the product name
* @param length                               The maximum allowed length of the string returned in \a name
*
* @return
*         - \ref NVML_SUCCESS                 if \a name has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a name is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char* name, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetName);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(name, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(name, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the brand of this device.
*
* For all products.
*
* The type is a member of \ref nvmlBrandType_t defined above.
*
* @param device                               The identifier of the target device
* @param type                                 Reference in which to return the product brand type
*
* @return
*         - \ref NVML_SUCCESS                 if \a name has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a type is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetBrand(nvmlDevice_t device, nvmlBrandType_t* type) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBrand);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(nvmlBrandType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the NVML index of this device.
*
* For all products.
*
* Valid indices are derived from the \a accessibleDevices count returned by
*   \ref nvmlDeviceGetCount_v2(). For example, if \a accessibleDevices is 2 the valid indices
*   are 0 and 1, corresponding to GPU 0 and GPU 1.
*
* The order in which NVML enumerates devices has no guarantees of consistency between reboots. For that reason it
*   is recommended that devices be looked up by their PCI ids or GPU UUID. See
*   \ref nvmlDeviceGetHandleByPciBusId_v2() and \ref nvmlDeviceGetHandleByUUID().
*
* When used with MIG device handles this API returns indices that can be
* passed to \ref nvmlDeviceGetMigDeviceHandleByIndex to retrieve an identical handle.
* MIG device indices are unique within a device.
*
* Note: The NVML index may not correlate with other APIs, such as the CUDA device index.
*
* @param device                               The identifier of the target device
* @param index                                Reference in which to return the NVML index of the device
*
* @return
*         - \ref NVML_SUCCESS                 if \a index has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a index is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetHandleByIndex()
* @see nvmlDeviceGetCount()
*/
nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int* index) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(index, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(index, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the globally unique board serial number associated with this device's board.
*
* For all products with an inforom.
*
* The serial number is an alphanumeric string that will not exceed 30 characters (including the NULL terminator).
* This number matches the serial number tag that is physically attached to the board.  See \ref
* nvmlConstants::NVML_DEVICE_SERIAL_BUFFER_SIZE.
*
* @param device                               The identifier of the target device
* @param serial                               Reference in which to return the board/module serial number
* @param length                               The maximum allowed length of the string returned in \a serial
*
* @return
*         - \ref NVML_SUCCESS                 if \a serial has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a serial is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetSerial(nvmlDevice_t device, char* serial, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSerial);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(serial, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(serial, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryAffinity(nvmlDevice_t device, unsigned int nodeSetSize, unsigned long* nodeSet, nvmlAffinityScope_t scope) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryAffinity);
    if (request_id < 0 ||
        rpc_write(&nodeSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(nodeSet, nodeSetSize * sizeof(unsigned long)) < 0 ||
        rpc_write(&scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(nodeSet, nodeSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetCpuAffinityWithinScope(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet, nvmlAffinityScope_t scope) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinityWithinScope);
    if (request_id < 0 ||
        rpc_write(&cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_write(&scope, sizeof(nvmlAffinityScope_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity for the device
* For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
*     result[0] = 0x3, result[1] = 0x3
* This is equivalent to calling \ref nvmlDeviceGetCpuAffinityWithinScope with \ref NVML_AFFINITY_SCOPE_NODE.
*
* For Kepler &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               The identifier of the target device
* @param cpuSetSize                           The size of the cpuSet array that is safe to access
* @param cpuSet                               Array reference in which to return a bitmask of CPUs, 64 CPUs per
*                                                 unsigned long on 64-bit machines, 32 on 32-bit machines
*
* @return
*         - \ref NVML_SUCCESS                 if \a cpuAffinity has been filled
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, cpuSetSize == 0, or cpuSet is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetCpuAffinity(nvmlDevice_t device, unsigned int cpuSetSize, unsigned long* cpuSet) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&cpuSetSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(cpuSet, cpuSetSize * sizeof(unsigned long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Sets the ideal affinity for the calling thread and device using the guidelines
* given in nvmlDeviceGetCpuAffinity().  Note, this is a change as of version 8.0.
* Older versions set the affinity for a calling process and all children.
* Currently supports up to 1024 processors.
*
* For Kepler &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if the calling process has been successfully bound
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetCpuAffinity(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Clear all affinity bindings for the calling thread.  Note, this is a change as of version
* 8.0 as older versions cleared the affinity for a calling process and all children.
*
* For Kepler &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if the calling process has been successfully unbound
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceClearCpuAffinity(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceClearCpuAffinity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/** @} */
nvmlReturn_t nvmlDeviceGetTopologyCommonAncestor(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuTopologyLevel_t* pathInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyCommonAncestor);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pathInfo, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the set of GPUs that are nearest to a given device at a specific interconnectivity level
* For all products.
* Supported on Linux only.
*
* @param device                               The identifier of the first device
* @param level                                The \ref nvmlGpuTopologyLevel_t level to search for other GPUs
* @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray
*                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
*                                             number of device handles.
* @param deviceArray                          An array of device handles for GPUs found at \a level
*
* @return
*         - \ref NVML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a level, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
*         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
*/
nvmlReturn_t nvmlDeviceGetTopologyNearestGpus(nvmlDevice_t device, nvmlGpuTopologyLevel_t level, unsigned int* count, nvmlDevice_t* deviceArray) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTopologyNearestGpus);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&level, sizeof(nvmlGpuTopologyLevel_t)) < 0 ||
        rpc_write(deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, *count * sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the set of GPUs that have a CPU affinity with the given CPU number
* For all products.
* Supported on Linux only.
*
* @param cpuNumber                            The CPU number
* @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray
*                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
*                                             number of device handles.
* @param deviceArray                          An array of device handles for GPUs found with affinity to \a cpuNumber
*
* @return
*         - \ref NVML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a cpuNumber, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
*         - \ref NVML_ERROR_UNKNOWN           an error has occurred in underlying topology discovery
*/
nvmlReturn_t nvmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int* count, nvmlDevice_t* deviceArray) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSystemGetTopologyGpuSet);
    if (request_id < 0 ||
        rpc_write(&cpuNumber, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(deviceArray, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(deviceArray, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the status for a given p2p capability index between a given pair of GPU
*
* @param device1                              The first device
* @param device2                              The second device
* @param p2pIndex                             p2p Capability Index being looked for between \a device1 and \a device2
* @param p2pStatus                            Reference in which to return the status of the \a p2pIndex
*                                             between \a device1 and \a device2
* @return
*         - \ref NVML_SUCCESS         if \a p2pStatus has been populated
*         - \ref NVML_ERROR_INVALID_ARGUMENT     if \a device1 or \a device2 or \a p2pIndex is invalid or \a p2pStatus is NULL
*         - \ref NVML_ERROR_UNKNOWN              on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetP2PStatus(nvmlDevice_t device1, nvmlDevice_t device2, nvmlGpuP2PCapsIndex_t p2pIndex, nvmlGpuP2PStatus_t* p2pStatus) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetP2PStatus);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&p2pIndex, sizeof(nvmlGpuP2PCapsIndex_t)) < 0 ||
        rpc_write(p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(p2pStatus, sizeof(nvmlGpuP2PStatus_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string,
* that augments the immutable, board serial identifier.
*
* For all products.
*
* The UUID is a globally unique identifier. It is the only available identifier for pre-Fermi-architecture products.
* It does NOT correspond to any identifier printed on the board.  It will not exceed 96 characters in length
* (including the NULL terminator).  See \ref nvmlConstants::NVML_DEVICE_UUID_V2_BUFFER_SIZE.
*
* When used with MIG device handles the API returns globally unique UUIDs which can be used to identify MIG
* devices across both GPU and MIG devices. UUIDs are immutable for the lifetime of a MIG device.
*
* @param device                               The identifier of the target device
* @param uuid                                 Reference in which to return the GPU UUID
* @param length                               The maximum allowed length of the string returned in \a uuid
*
* @return
*         - \ref NVML_SUCCESS                 if \a uuid has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a uuid is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char* uuid, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetUUID);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(uuid, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the MDEV UUID of a vGPU instance.
*
* The MDEV UUID is a globally unique identifier of the mdev device assigned to the VM, and is returned as a 5-part hexadecimal string,
* not exceeding 80 characters in length (including the NULL terminator).
* MDEV UUID is displayed only on KVM platform.
* See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param mdevUuid                 Pointer to caller-supplied buffer to hold MDEV UUID
* @param size                     Size of buffer in bytes
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_NOT_SUPPORTED     on any hypervisor other than KVM
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a mdevUuid is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetMdevUUID(nvmlVgpuInstance_t vgpuInstance, char* mdevUuid, unsigned int size) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMdevUUID);
    if (request_id < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(mdevUuid, size * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mdevUuid, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves minor number for the device. The minor number for the device is such that the Nvidia device node file for
* each GPU will have the form /dev/nvidia[minor number].
*
* For all products.
* Supported only for Linux
*
* @param device                                The identifier of the target device
* @param minorNumber                           Reference in which to return the minor number for the device
* @return
*         - \ref NVML_SUCCESS                 if the minor number is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minorNumber is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinorNumber);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minorNumber, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the the device board part number which is programmed into the board's InfoROM
*
* For all products.
*
* @param device                                Identifier of the target device
* @param partNumber                            Reference to the buffer to return
* @param length                                Length of the buffer reference
*
* @return
*         - \ref NVML_SUCCESS                  if \a partNumber has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_NOT_SUPPORTED      if the needed VBIOS fields have not been filled
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a serial is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetBoardPartNumber(nvmlDevice_t device, char* partNumber, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardPartNumber);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(partNumber, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(partNumber, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the version information for the device's infoROM object.
*
* For all products with an inforom.
*
* Fermi and higher parts have non-volatile on-board memory for persisting device info, such as aggregate
* ECC counts. The version of the data structures in this memory may change from time to time. It will not
* exceed 16 characters in length (including the NULL terminator).
* See \ref nvmlConstants::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
*
* See \ref nvmlInforomObject_t for details on the available infoROM objects.
*
* @param device                               The identifier of the target device
* @param object                               The target infoROM object
* @param version                              Reference in which to return the infoROM version
* @param length                               The maximum allowed length of the string returned in \a version
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetInforomImageVersion
*/
nvmlReturn_t nvmlDeviceGetInforomVersion(nvmlDevice_t device, nvmlInforomObject_t object, char* version, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&object, sizeof(nvmlInforomObject_t)) < 0 ||
        rpc_write(version, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the global infoROM image version
*
* For all products with an inforom.
*
* Image version just like VBIOS version uniquely describes the exact version of the infoROM flashed on the board
* in contrast to infoROM object version which is only an indicator of supported features.
* Version string will not exceed 16 characters in length (including the NULL terminator).
* See \ref nvmlConstants::NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
*
* @param device                               The identifier of the target device
* @param version                              Reference in which to return the infoROM image version
* @param length                               The maximum allowed length of the string returned in \a version
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a version is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetInforomVersion
*/
nvmlReturn_t nvmlDeviceGetInforomImageVersion(nvmlDevice_t device, char* version, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomImageVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(version, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the checksum of the configuration stored in the device's infoROM.
*
* For all products with an inforom.
*
* Can be used to make sure that two GPUs have the exact same configuration.
* Current checksum takes into account configuration stored in PWR and ECC infoROM objects.
* Checksum can change between driver releases or when user changes configuration (e.g. disable/enable ECC)
*
* @param device                               The identifier of the target device
* @param checksum                             Reference in which to return the infoROM configuration checksum
*
* @return
*         - \ref NVML_SUCCESS                 if \a checksum has been set
*         - \ref NVML_ERROR_CORRUPTED_INFOROM if the device's checksum couldn't be retrieved due to infoROM corruption
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a checksum is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice_t device, unsigned int* checksum) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetInforomConfigurationChecksum);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(checksum, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(checksum, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Reads the infoROM from the flash and verifies the checksums.
*
* For all products with an inforom.
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if infoROM is not corrupted
*         - \ref NVML_ERROR_CORRUPTED_INFOROM if the device's infoROM is corrupted
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceValidateInforom(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceValidateInforom);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the display mode for the device.
*
* For all products.
*
* This method indicates whether a physical display (e.g. monitor) is currently connected to
* any of the device's connectors.
*
* See \ref nvmlEnableState_t for details on allowed modes.
*
* @param device                               The identifier of the target device
* @param display                              Reference in which to return the display mode
*
* @return
*         - \ref NVML_SUCCESS                 if \a display has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a display is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetDisplayMode(nvmlDevice_t device, nvmlEnableState_t* display) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(display, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the display active state for the device.
*
* For all products.
*
* This method indicates whether a display is initialized on the device.
* For example whether X Server is attached to this device and has allocated memory for the screen.
*
* Display can be active even when no monitor is physically attached.
*
* See \ref nvmlEnableState_t for details on allowed modes.
*
* @param device                               The identifier of the target device
* @param isActive                             Reference in which to return the display active state
*
* @return
*         - \ref NVML_SUCCESS                 if \a isActive has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isActive is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetDisplayActive(nvmlDevice_t device, nvmlEnableState_t* isActive) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDisplayActive);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the persistence mode associated with this device.
*
* For all products.
* For Linux only.
*
* When driver persistence mode is enabled the driver software state is not torn down when the last
* client disconnects. By default this feature is disabled.
*
* See \ref nvmlEnableState_t for details on allowed modes.
*
* @param device                               The identifier of the target device
* @param mode                                 Reference in which to return the current driver persistence mode
*
* @return
*         - \ref NVML_SUCCESS                 if \a mode has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetPersistenceMode()
*/
nvmlReturn_t nvmlDeviceGetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPersistenceMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the PCI attributes of this device.
*
* For all products.
*
* See \ref nvmlPciInfo_t for details on the available PCI info.
*
* @param device                               The identifier of the target device
* @param pci                                  Reference in which to return the PCI info
*
* @return
*         - \ref NVML_SUCCESS                 if \a pci has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pci is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t* pci) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPciInfo_v3);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the maximum PCIe link generation possible with this device and system
*
* I.E. for a generation 2 PCIe device attached to a generation 1 PCIe bus the max link generation this function will
* report is generation 1.
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param maxLinkGen                           Reference in which to return the max PCIe link generation
*
* @return
*         - \ref NVML_SUCCESS                 if \a maxLinkGen has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkGen is null
*         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGen) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the maximum PCIe link generation supported by this device
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param maxLinkGenDevice                     Reference in which to return the max PCIe link generation
*
* @return
*         - \ref NVML_SUCCESS                 if \a maxLinkGenDevice has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkGenDevice is null
*         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGpuMaxPcieLinkGeneration(nvmlDevice_t device, unsigned int* maxLinkGenDevice) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkGenDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the maximum PCIe link width possible with this device and system
*
* I.E. for a device with a 16x PCIe bus width attached to a 8x PCIe system bus this function will report
* a max link width of 8.
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param maxLinkWidth                         Reference in which to return the max PCIe link generation
*
* @return
*         - \ref NVML_SUCCESS                 if \a maxLinkWidth has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a maxLinkWidth is null
*         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice_t device, unsigned int* maxLinkWidth) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxPcieLinkWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current PCIe link generation
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param currLinkGen                          Reference in which to return the current PCIe link generation
*
* @return
*         - \ref NVML_SUCCESS                 if \a currLinkGen has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a currLinkGen is null
*         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int* currLinkGen) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkGeneration);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currLinkGen, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current PCIe link width
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param currLinkWidth                        Reference in which to return the current PCIe link generation
*
* @return
*         - \ref NVML_SUCCESS                 if \a currLinkWidth has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a currLinkWidth is null
*         - \ref NVML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int* currLinkWidth) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrPcieLinkWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currLinkWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve PCIe utilization information.
* This function is querying a byte counter over a 20ms interval and thus is the
*   PCIe throughput over that interval.
*
* For Maxwell &tm; or newer fully supported devices.
*
* This method is not supported in virtual machines running virtual GPU (vGPU).
*
* @param device                               The identifier of the target device
* @param counter                              The specific counter that should be queried \ref nvmlPcieUtilCounter_t
* @param value                                Reference in which to return throughput in KB/s
*
* @return
*         - \ref NVML_SUCCESS                 if \a value has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a counter is invalid, or \a value is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPcieThroughput(nvmlDevice_t device, nvmlPcieUtilCounter_t counter, unsigned int* value) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieThroughput);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counter, sizeof(nvmlPcieUtilCounter_t)) < 0 ||
        rpc_write(value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the PCIe replay counter.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param value                                Reference in which to return the counter's value
*
* @return
*         - \ref NVML_SUCCESS                 if \a value has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a value is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPcieReplayCounter(nvmlDevice_t device, unsigned int* value) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieReplayCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(value, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(value, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current clock speeds for the device.
*
* For Fermi &tm; or newer fully supported devices.
*
* See \ref nvmlClockType_t for details on available clock information.
*
* @param device                               The identifier of the target device
* @param type                                 Identify which clock domain to query
* @param clock                                Reference in which to return the clock speed in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a clock has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clock, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the maximum clock speeds for the device.
*
* For Fermi &tm; or newer fully supported devices.
*
* See \ref nvmlClockType_t for details on available clock information.
*
* \note On GPUs from Fermi family current P0 clocks (reported by \ref nvmlDeviceGetClockInfo) can differ from max clocks
*       by few MHz.
*
* @param device                               The identifier of the target device
* @param type                                 Identify which clock domain to query
* @param clock                                Reference in which to return the clock speed in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a clock has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMaxClockInfo(nvmlDevice_t device, nvmlClockType_t type, unsigned int* clock) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxClockInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clock, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clock, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current setting of a clock that applications will use unless an overspec situation occurs.
* Can be changed using \ref nvmlDeviceSetApplicationsClocks.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param clockType                            Identify which clock domain to query
* @param clockMHz                             Reference in which to return the clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a clockMHz has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetApplicationsClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the default applications clock that GPU boots with or
* defaults to after \ref nvmlDeviceResetApplicationsClocks call.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param clockType                            Identify which clock domain to query
* @param clockMHz                             Reference in which to return the default clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a clockMHz has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* \see nvmlDeviceGetApplicationsClock
*/
nvmlReturn_t nvmlDeviceGetDefaultApplicationsClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultApplicationsClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Resets the application clock to the default value
*
* This is the applications clock that will be used after system reboot or driver reload.
* Default value is constant, but the current value an be changed using \ref nvmlDeviceSetApplicationsClocks.
*
* On Pascal and newer hardware, if clocks were previously locked with \ref nvmlDeviceSetApplicationsClocks,
* this call will unlock clocks. This returns clocks their default behavior ofautomatically boosting above
* base clocks as thermal limits allow.
*
* @see nvmlDeviceGetApplicationsClock
* @see nvmlDeviceSetApplicationsClocks
*
* For Fermi &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if new settings were successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceResetApplicationsClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceResetApplicationsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the clock speed for the clock specified by the clock type and clock ID.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param clockType                            Identify which clock domain to query
* @param clockId                              Identify which clock in the domain to query
* @param clockMHz                             Reference in which to return the clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a clockMHz has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetClock(nvmlDevice_t device, nvmlClockType_t clockType, nvmlClockId_t clockId, unsigned int* clockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&clockId, sizeof(nvmlClockId_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the customer defined maximum boost clock speed specified by the given clock type.
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param clockType                            Identify which clock domain to query
* @param clockMHz                             Reference in which to return the clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a clockMHz has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device or the \a clockType on this device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMaxCustomerBoostClock(nvmlDevice_t device, nvmlClockType_t clockType, unsigned int* clockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxCustomerBoostClock);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&clockType, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the list of possible memory clocks that can be used as an argument for \ref nvmlDeviceSetApplicationsClocks.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param count                                Reference in which to provide the \a clocksMHz array size, and
*                                             to return the number of elements
* @param clocksMHz                            Reference in which to return the clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a count and \a clocksMHz have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a count is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to the number of
*                                                required elements)
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetApplicationsClocks
* @see nvmlDeviceGetSupportedGraphicsClocks
*/
nvmlReturn_t nvmlDeviceGetSupportedMemoryClocks(nvmlDevice_t device, unsigned int* count, unsigned int* clocksMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedMemoryClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the list of possible graphics clocks that can be used as an argument for \ref nvmlDeviceSetApplicationsClocks.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param memoryClockMHz                       Memory clock for which to return possible graphics clocks
* @param count                                Reference in which to provide the \a clocksMHz array size, and
*                                             to return the number of elements
* @param clocksMHz                            Reference in which to return the clocks in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if \a count and \a clocksMHz have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_NOT_FOUND         if the specified \a memoryClockMHz is not a supported frequency
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clock is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetApplicationsClocks
* @see nvmlDeviceGetSupportedMemoryClocks
*/
nvmlReturn_t nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice_t device, unsigned int memoryClockMHz, unsigned int* count, unsigned int* clocksMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedGraphicsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memoryClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(clocksMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the current state of Auto Boosted clocks on a device and store it in \a isEnabled
*
* For Kepler &tm; or newer fully supported devices.
*
* Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
* to maximize performance as thermal limits allow.
*
* On Pascal and newer hardware, Auto Aoosted clocks are controlled through application clocks.
* Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
* behavior.
*
* @param device                               The identifier of the target device
* @param isEnabled                            Where to store the current state of Auto Boosted clocks of the target device
* @param defaultIsEnabled                     Where to store the default Auto Boosted clocks behavior of the target device that the device will
*                                                 revert to when no applications are using the GPU
*
* @return
*         - \ref NVML_SUCCESS                 If \a isEnabled has been been set with the Auto Boosted clocks state of \a device
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isEnabled is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
*/
nvmlReturn_t nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t* isEnabled, nvmlEnableState_t* defaultIsEnabled) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(defaultIsEnabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Try to set the current state of Auto Boosted clocks on a device.
*
* For Kepler &tm; or newer fully supported devices.
*
* Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
* to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
* rates are desired.
*
* Non-root users may use this API by default but can be restricted by root from using this API by calling
* \ref nvmlDeviceSetAPIRestriction with apiType=NVML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS.
* Note: Persistence Mode is required to modify current Auto Boost settings, therefore, it must be enabled.
*
* On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
* Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
* behavior.
*
* @param device                               The identifier of the target device
* @param enabled                              What state to try to set Auto Boosted clocks of the target device to
*
* @return
*         - \ref NVML_SUCCESS                 If the Auto Boosted clocks were successfully set to the state specified by \a enabled
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
*/
nvmlReturn_t nvmlDeviceSetAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Try to set the default state of Auto Boosted clocks on a device. This is the default state that Auto Boosted clocks will
* return to when no compute running processes (e.g. CUDA application which have an active context) are running
*
* For Kepler &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
* Requires root/admin permissions.
*
* Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
* to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
* rates are desired.
*
* On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
* Use \ref nvmlDeviceSetApplicationsClocks and \ref nvmlDeviceResetApplicationsClocks to control Auto Boost
* behavior.
*
* @param device                               The identifier of the target device
* @param enabled                              What state to try to set default Auto Boosted clocks of the target device to
* @param flags                                Flags that change the default behavior. Currently Unused.
*
* @return
*         - \ref NVML_SUCCESS                 If the Auto Boosted clock's default state was successfully set to the state specified by \a enabled
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_NO_PERMISSION     If the calling user does not have permission to change Auto Boosted clock's default state.
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
*/
nvmlReturn_t nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice_t device, nvmlEnableState_t enabled, unsigned int flags) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&enabled, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the intended operating speed of the device's fan.
*
* Note: The reported speed is the intended fan speed.  If the fan is physically blocked and unable to spin, the
* output will not match the actual fan speed.
*
* For all discrete products with dedicated fans.
*
* The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
* This value may exceed 100% in certain cases.
*
* @param device                               The identifier of the target device
* @param speed                                Reference in which to return the fan speed percentage
*
* @return
*         - \ref NVML_SUCCESS                 if \a speed has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a speed is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a fan
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetFanSpeed(nvmlDevice_t device, unsigned int* speed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the intended operating speed of the device's specified fan.
*
* Note: The reported speed is the intended fan speed. If the fan is physically blocked and unable to spin, the
* output will not match the actual fan speed.
*
* For all discrete products with dedicated fans.
*
* The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
* This value may exceed 100% in certain cases.
*
* @param device                                The identifier of the target device
* @param fan                                   The index of the target fan, zero indexed.
* @param speed                                 Reference in which to return the fan speed percentage
*
* @return
*        - \ref NVML_SUCCESS                   if \a speed has been set
*        - \ref NVML_ERROR_UNINITIALIZED       if the library has not been successfully initialized
*        - \ref NVML_ERROR_INVALID_ARGUMENT    if \a device is invalid, \a fan is not an acceptable index, or \a speed is NULL
*        - \ref NVML_ERROR_NOT_SUPPORTED       if the device does not have a fan or is newer than Maxwell
*        - \ref NVML_ERROR_GPU_IS_LOST         if the target GPU has fallen off the bus or is otherwise inaccessible
*        - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int* speed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(speed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the intended target speed of the device's specified fan.
*
* Normally, the driver dynamically adjusts the fan based on
* the needs of the GPU.  But when user set fan speed using nvmlDeviceSetFanSpeed_v2,
* the driver will attempt to make the fan achieve the setting in
* nvmlDeviceSetFanSpeed_v2.  The actual current speed of the fan
* is reported in nvmlDeviceGetFanSpeed_v2.
*
* For all discrete products with dedicated fans.
*
* The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
* This value may exceed 100% in certain cases.
*
* @param device                                The identifier of the target device
* @param fan                                   The index of the target fan, zero indexed.
* @param targetSpeed                           Reference in which to return the fan speed percentage
*
* @return
*        - \ref NVML_SUCCESS                   if \a speed has been set
*        - \ref NVML_ERROR_UNINITIALIZED       if the library has not been successfully initialized
*        - \ref NVML_ERROR_INVALID_ARGUMENT    if \a device is invalid, \a fan is not an acceptable index, or \a speed is NULL
*        - \ref NVML_ERROR_NOT_SUPPORTED       if the device does not have a fan or is newer than Maxwell
*        - \ref NVML_ERROR_GPU_IS_LOST         if the target GPU has fallen off the bus or is otherwise inaccessible
*        - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetTargetFanSpeed(nvmlDevice_t device, unsigned int fan, unsigned int* targetSpeed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTargetFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(targetSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Sets the speed of the fan control policy to default.
*
* For all cuda-capable discrete products with fans
*
* @param device                        The identifier of the target device
* @param fan                           The index of the fan, starting at zero
*
* return
*         NVML_SUCCESS                 if speed has been adjusted
*         NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         NVML_ERROR_INVALID_ARGUMENT  if device is invalid
*         NVML_ERROR_NOT_SUPPORTED     if the device does not support this
*                                      (doesn't have fans)
*         NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetDefaultFanSpeed_v2(nvmlDevice_t device, unsigned int fan) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetDefaultFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the min and max fan speed that user can set for the GPU fan.
*
* For all cuda-capable discrete products with fans
*
* @param device                        The identifier of the target device
* @param minSpeed                      The minimum speed allowed to set
* @param maxSpeed                      The maximum speed allowed to set
*
* return
*         NVML_SUCCESS                 if speed has been adjusted
*         NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         NVML_ERROR_INVALID_ARGUMENT  if device is invalid
*         NVML_ERROR_NOT_SUPPORTED     if the device does not support this
*                                      (doesn't have fans)
*         NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMinMaxFanSpeed(nvmlDevice_t device, unsigned int* minSpeed, unsigned int* maxSpeed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxFanSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_write(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minSpeed, sizeof(unsigned int)) < 0 ||
        rpc_read(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets current fan control policy.
*
* For Maxwell &tm; or newer fully supported devices.
*
* For all cuda-capable discrete products with fans
*
* device                               The identifier of the target \a device
* policy                               Reference in which to return the fan control \a policy
*
* return
*         NVML_SUCCESS                 if \a policy has been populated
*         NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a policy is null or the \a fan given doesn't reference
*                                            a fan that exists.
*         NVML_ERROR_NOT_SUPPORTED     if the \a device is older than Maxwell
*         NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetFanControlPolicy_v2(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t* policy) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFanControlPolicy_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Sets current fan control policy.
*
* For Maxwell &tm; or newer fully supported devices.
*
* Requires privileged user.
*
* For all cuda-capable discrete products with fans
*
* device                               The identifier of the target \a device
* policy                               The fan control \a policy to set
*
* return
*         NVML_SUCCESS                 if \a policy has been set
*         NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a policy is null or the \a fan given doesn't reference
*                                            a fan that exists.
*         NVML_ERROR_NOT_SUPPORTED     if the \a device is older than Maxwell
*         NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetFanControlPolicy(nvmlDevice_t device, unsigned int fan, nvmlFanControlPolicy_t policy) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetFanControlPolicy);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(&policy, sizeof(nvmlFanControlPolicy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the number of fans on the device.
*
* For all discrete products with dedicated fans.
*
* @param device                               The identifier of the target device
* @param numFans                              The number of fans
*
* @return
*         - \ref NVML_SUCCESS                 if \a fan number query was successful
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a numFans is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a fan
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNumFans(nvmlDevice_t device, unsigned int* numFans) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumFans);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(numFans, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numFans, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current temperature readings for the device, in degrees C.
*
* For all products.
*
* See \ref nvmlTemperatureSensors_t for details on available temperature sensors.
*
* @param device                               The identifier of the target device
* @param sensorType                           Flag that indicates which sensor reading to retrieve
* @param temp                                 Reference in which to return the temperature reading
*
* @return
*         - \ref NVML_SUCCESS                 if \a temp has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a sensorType is invalid or \a temp is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have the specified sensor
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetTemperature(nvmlDevice_t device, nvmlTemperatureSensors_t sensorType, unsigned int* temp) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperature);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorType, sizeof(nvmlTemperatureSensors_t)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the temperature threshold for the GPU with the specified threshold type in degrees C.
*
* For Kepler &tm; or newer fully supported devices.
*
* See \ref nvmlTemperatureThresholds_t for details on available temperature thresholds.
*
* @param device                               The identifier of the target device
* @param thresholdType                        The type of threshold value queried
* @param temp                                 Reference in which to return the temperature reading
* @return
*         - \ref NVML_SUCCESS                 if \a temp has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, unsigned int* temp) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(temp, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Sets the temperature threshold for the GPU with the specified threshold type in degrees C.
*
* For Maxwell &tm; or newer fully supported devices.
*
* See \ref nvmlTemperatureThresholds_t for details on available temperature thresholds.
*
* @param device                               The identifier of the target device
* @param thresholdType                        The type of threshold value to be set
* @param temp                                 Reference which hold the value to be set
* @return
*         - \ref NVML_SUCCESS                 if \a temp has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetTemperatureThreshold(nvmlDevice_t device, nvmlTemperatureThresholds_t thresholdType, int* temp) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetTemperatureThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&thresholdType, sizeof(nvmlTemperatureThresholds_t)) < 0 ||
        rpc_write(temp, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(temp, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Used to execute a list of thermal system instructions.
*
* @param device                               The identifier of the target device
* @param sensorIndex                          The index of the thermal sensor
* @param pThermalSettings                     Reference in which to return the thermal sensor information
*
* @return
*         - \ref NVML_SUCCESS                 if \a pThermalSettings has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pThermalSettings is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetThermalSettings(nvmlDevice_t device, unsigned int sensorIndex, nvmlGpuThermalSettings_t* pThermalSettings) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetThermalSettings);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&sensorIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pThermalSettings, sizeof(nvmlGpuThermalSettings_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current performance state for the device.
*
* For Fermi &tm; or newer fully supported devices.
*
* See \ref nvmlPstates_t for details on allowed performance states.
*
* @param device                               The identifier of the target device
* @param pState                               Reference in which to return the performance state reading
*
* @return
*         - \ref NVML_SUCCESS                 if \a pState has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pState is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPerformanceState(nvmlDevice_t device, nvmlPstates_t* pState) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPerformanceState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves current clocks throttling reasons.
*
* For all fully supported products.
*
* \note More than one bit can be enabled at the same time. Multiple reasons can be affecting clocks at once.
*
* @param device                                The identifier of the target device
* @param clocksThrottleReasons                 Reference in which to return bitmask of active clocks throttle
*                                                  reasons
*
* @return
*         - \ref NVML_SUCCESS                 if \a clocksThrottleReasons has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a clocksThrottleReasons is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlClocksThrottleReasons
* @see nvmlDeviceGetSupportedClocksThrottleReasons
*/
nvmlReturn_t nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice_t device, unsigned long long* clocksThrottleReasons) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCurrentClocksThrottleReasons);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(clocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves bitmask of supported clocks throttle reasons that can be returned by
* \ref nvmlDeviceGetCurrentClocksThrottleReasons
*
* For all fully supported products.
*
* This method is not supported in virtual machines running virtual GPU (vGPU).
*
* @param device                               The identifier of the target device
* @param supportedClocksThrottleReasons       Reference in which to return bitmask of supported
*                                              clocks throttle reasons
*
* @return
*         - \ref NVML_SUCCESS                 if \a supportedClocksThrottleReasons has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a supportedClocksThrottleReasons is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlClocksThrottleReasons
* @see nvmlDeviceGetCurrentClocksThrottleReasons
*/
nvmlReturn_t nvmlDeviceGetSupportedClocksThrottleReasons(nvmlDevice_t device, unsigned long long* supportedClocksThrottleReasons) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedClocksThrottleReasons);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(supportedClocksThrottleReasons, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Deprecated: Use \ref nvmlDeviceGetPerformanceState. This function exposes an incorrect generalization.
*
* Retrieve the current performance state for the device.
*
* For Fermi &tm; or newer fully supported devices.
*
* See \ref nvmlPstates_t for details on allowed performance states.
*
* @param device                               The identifier of the target device
* @param pState                               Reference in which to return the performance state reading
*
* @return
*         - \ref NVML_SUCCESS                 if \a pState has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pState is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPowerState(nvmlDevice_t device, nvmlPstates_t* pState) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pState, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* This API has been deprecated.
*
* Retrieves the power management mode associated with this device.
*
* For products from the Fermi family.
*     - Requires \a NVML_INFOROM_POWER version 3.0 or higher.
*
* For from the Kepler or newer families.
*     - Does not require \a NVML_INFOROM_POWER object.
*
* This flag indicates whether any power management algorithm is currently active on the device. An
* enabled state does not necessarily mean the device is being actively throttled -- only that
* that the driver will do so if the appropriate conditions are met.
*
* See \ref nvmlEnableState_t for details on allowed modes.
*
* @param device                               The identifier of the target device
* @param mode                                 Reference in which to return the current power management mode
*
* @return
*         - \ref NVML_SUCCESS                 if \a mode has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPowerManagementMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the power management limit associated with this device.
*
* For Fermi &tm; or newer fully supported devices.
*
* The power limit defines the upper boundary for the card's power draw. If
* the card's total power draw reaches this limit the power management algorithm kicks in.
*
* This reading is only available if power management mode is supported.
* See \ref nvmlDeviceGetPowerManagementMode.
*
* @param device                               The identifier of the target device
* @param limit                                Reference in which to return the power management limit in milliwatts
*
* @return
*         - \ref NVML_SUCCESS                 if \a limit has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a limit is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPowerManagementLimit(nvmlDevice_t device, unsigned int* limit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves information about possible values of power management limits on this device.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param minLimit                             Reference in which to return the minimum power management limit in milliwatts
* @param maxLimit                             Reference in which to return the maximum power management limit in milliwatts
*
* @return
*         - \ref NVML_SUCCESS                 if \a minLimit and \a maxLimit have been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minLimit or \a maxLimit is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetPowerManagementLimit
*/
nvmlReturn_t nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice_t device, unsigned int* minLimit, unsigned int* maxLimit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementLimitConstraints);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minLimit, sizeof(unsigned int)) < 0 ||
        rpc_write(maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minLimit, sizeof(unsigned int)) < 0 ||
        rpc_read(maxLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves default power management limit on this device, in milliwatts.
* Default power management limit is a power management limit that the device boots with.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param defaultLimit                         Reference in which to return the default power management limit in milliwatts
*
* @return
*         - \ref NVML_SUCCESS                 if \a defaultLimit has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a defaultLimit is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice_t device, unsigned int* defaultLimit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerManagementDefaultLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(defaultLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves power usage for this GPU in milliwatts and its associated circuitry (e.g. memory)
*
* For Fermi &tm; or newer fully supported devices.
*
* On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
*
* It is only available if power management mode is supported. See \ref nvmlDeviceGetPowerManagementMode.
*
* @param device                               The identifier of the target device
* @param power                                Reference in which to return the power usage information
*
* @return
*         - \ref NVML_SUCCESS                 if \a power has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a power is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support power readings
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPowerUsage(nvmlDevice_t device, unsigned int* power) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerUsage);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(power, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(power, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded
*
* For Volta &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param energy                               Reference in which to return the energy consumption information
*
* @return
*         - \ref NVML_SUCCESS                 if \a energy has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a energy is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support energy readings
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetTotalEnergyConsumption(nvmlDevice_t device, unsigned long long* energy) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEnergyConsumption);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(energy, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(energy, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get the effective power limit that the driver enforces after taking into account all limiters
*
* Note: This can be different from the \ref nvmlDeviceGetPowerManagementLimit if other limits are set elsewhere
* This includes the out of band power limit interface
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                           The device to communicate with
* @param limit                            Reference in which to return the power management limit in milliwatts
*
* @return
*         - \ref NVML_SUCCESS                 if \a limit has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a limit is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetEnforcedPowerLimit(nvmlDevice_t device, unsigned int* limit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEnforcedPowerLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(limit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current GOM and pending GOM (the one that GPU will switch to after reboot).
*
* For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
* Modes \ref NVML_GOM_LOW_DP and \ref NVML_GOM_ALL_ON are supported on fully supported GeForce products.
* Not supported on Quadro &reg; and Tesla &tm; C-class products.
*
* @param device                               The identifier of the target device
* @param current                              Reference in which to return the current GOM
* @param pending                              Reference in which to return the pending GOM
*
* @return
*         - \ref NVML_SUCCESS                 if \a mode has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a current or \a pending is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlGpuOperationMode_t
* @see nvmlDeviceSetGpuOperationMode
*/
nvmlReturn_t nvmlDeviceGetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t* current, nvmlGpuOperationMode_t* pending) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuOperationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_write(pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the amount of used, free, reserved and total memory available on the device, in bytes.
* The reserved amount is supported on version 2 only.
*
* For all products.
*
* Enabling ECC reduces the amount of total available memory, due to the extra required parity bits.
* Under WDDM most device memory is allocated and managed on startup by Windows.
*
* Under Linux and Windows TCC, the reported amount of used memory is equal to the sum of memory allocated
* by all active channels on the device.
*
* See \ref nvmlMemory_v2_t for details on available memory info.
*
* @note In MIG mode, if device handle is provided, the API returns aggregate
*       information, only if the caller has appropriate privileges. Per-instance
*       information can be queried by using specific MIG device handles.
*
* @note nvmlDeviceGetMemoryInfo_v2 adds additional memory information.
*
* @param device                               The identifier of the target device
* @param memory                               Reference in which to return the memory information
*
* @return
*         - \ref NVML_SUCCESS                 if \a memory has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a memory is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t* memory) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memory, sizeof(nvmlMemory_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t* memory) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(memory, sizeof(nvmlMemory_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current compute mode for the device.
*
* For all products.
*
* See \ref nvmlComputeMode_t for details on allowed compute modes.
*
* @param device                               The identifier of the target device
* @param mode                                 Reference in which to return the current compute mode
*
* @return
*         - \ref NVML_SUCCESS                 if \a mode has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetComputeMode()
*/
nvmlReturn_t nvmlDeviceGetComputeMode(nvmlDevice_t device, nvmlComputeMode_t* mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the CUDA compute capability of the device.
*
* For all products.
*
* Returns the major and minor compute capability version numbers of the
* device.  The major and minor versions are equivalent to the
* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR and
* CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR attributes that would be
* returned by CUDA's cuDeviceGetAttribute().
*
* @param device                               The identifier of the target device
* @param major                                Reference in which to return the major CUDA compute capability
* @param minor                                Reference in which to return the minor CUDA compute capability
*
* @return
*         - \ref NVML_SUCCESS                 if \a major and \a minor have been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a major or \a minor are NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCudaComputeCapability);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(major, sizeof(int)) < 0 ||
        rpc_write(minor, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(major, sizeof(int)) < 0 ||
        rpc_read(minor, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current and pending ECC modes for the device.
*
* For Fermi &tm; or newer fully supported devices.
* Only applicable to devices with ECC.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher.
*
* Changing ECC modes requires a reboot. The "pending" ECC mode refers to the target mode following
* the next reboot.
*
* See \ref nvmlEnableState_t for details on allowed modes.
*
* @param device                               The identifier of the target device
* @param current                              Reference in which to return the current ECC mode
* @param pending                              Reference in which to return the pending ECC mode
*
* @return
*         - \ref NVML_SUCCESS                 if \a current and \a pending have been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or either \a current or \a pending is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetEccMode()
*/
nvmlReturn_t nvmlDeviceGetEccMode(nvmlDevice_t device, nvmlEnableState_t* current, nvmlEnableState_t* pending) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_write(pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the default ECC modes for the device.
*
* For Fermi &tm; or newer fully supported devices.
* Only applicable to devices with ECC.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher.
*
* See \ref nvmlEnableState_t for details on allowed modes.
*
* @param device                               The identifier of the target device
* @param defaultMode                          Reference in which to return the default ECC mode
*
* @return
*         - \ref NVML_SUCCESS                 if \a current and \a pending have been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a default is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetEccMode()
*/
nvmlReturn_t nvmlDeviceGetDefaultEccMode(nvmlDevice_t device, nvmlEnableState_t* defaultMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDefaultEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(defaultMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the device boardId from 0-N.
* Devices with the same boardId indicate GPUs connected to the same PLX.  Use in conjunction with
*  \ref nvmlDeviceGetMultiGpuBoard() to decide if they are on the same board as well.
*  The boardId returned is a unique ID for the current configuration.  Uniqueness and ordering across
*  reboots and system configurations is not guaranteed (i.e. if a Tesla K40c returns 0x100 and
*  the two GPUs on a Tesla K10 in the same system returns 0x200 it is not guaranteed they will
*  always return those values but they will always be different from each other).
*
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param boardId                              Reference in which to return the device's board ID
*
* @return
*         - \ref NVML_SUCCESS                 if \a boardId has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a boardId is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetBoardId(nvmlDevice_t device, unsigned int* boardId) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBoardId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(boardId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(boardId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves whether the device is on a Multi-GPU Board
* Devices that are on multi-GPU boards will set \a multiGpuBool to a non-zero value.
*
* For Fermi &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param multiGpuBool                         Reference in which to return a zero or non-zero value
*                                                 to indicate whether the device is on a multi GPU board
*
* @return
*         - \ref NVML_SUCCESS                 if \a multiGpuBool has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a multiGpuBool is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMultiGpuBoard(nvmlDevice_t device, unsigned int* multiGpuBool) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMultiGpuBoard);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(multiGpuBool, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the total ECC error counts for the device.
*
* For Fermi &tm; or newer fully supported devices.
* Only applicable to devices with ECC.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher.
* Requires ECC Mode to be enabled.
*
* The total error count is the sum of errors across each of the separate memory systems, i.e. the total set of
* errors across the entire device.
*
* See \ref nvmlMemoryErrorType_t for a description of available error types.\n
* See \ref nvmlEccCounterType_t for a description of available counter types.
*
* @param device                               The identifier of the target device
* @param errorType                            Flag that specifies the type of the errors.
* @param counterType                          Flag that specifies the counter-type of the errors.
* @param eccCounts                            Reference in which to return the specified ECC errors
*
* @return
*         - \ref NVML_SUCCESS                 if \a eccCounts has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceClearEccErrorCounts()
*/
nvmlReturn_t nvmlDeviceGetTotalEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, unsigned long long* eccCounts) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetTotalEccErrors);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccCounts, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the detailed ECC error counts for the device.
*
* @deprecated   This API supports only a fixed set of ECC error locations
*               On different GPU architectures different locations are supported
*               See \ref nvmlDeviceGetMemoryErrorCounter
*
* For Fermi &tm; or newer fully supported devices.
* Only applicable to devices with ECC.
* Requires \a NVML_INFOROM_ECC version 2.0 or higher to report aggregate location-based ECC counts.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher to report all other ECC counts.
* Requires ECC Mode to be enabled.
*
* Detailed errors provide separate ECC counts for specific parts of the memory system.
*
* Reports zero for unsupported ECC error counters when a subset of ECC error counters are supported.
*
* See \ref nvmlMemoryErrorType_t for a description of available bit types.\n
* See \ref nvmlEccCounterType_t for a description of available counter types.\n
* See \ref nvmlEccErrorCounts_t for a description of provided detailed ECC counts.
*
* @param device                               The identifier of the target device
* @param errorType                            Flag that specifies the type of the errors.
* @param counterType                          Flag that specifies the counter-type of the errors.
* @param eccCounts                            Reference in which to return the specified ECC errors
*
* @return
*         - \ref NVML_SUCCESS                 if \a eccCounts has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceClearEccErrorCounts()
*/
nvmlReturn_t nvmlDeviceGetDetailedEccErrors(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlEccErrorCounts_t* eccCounts) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDetailedEccErrors);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccCounts, sizeof(nvmlEccErrorCounts_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the requested memory error counter for the device.
*
* For Fermi &tm; or newer fully supported devices.
* Requires \a NVML_INFOROM_ECC version 2.0 or higher to report aggregate location-based memory error counts.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher to report all other memory error counts.
*
* Only applicable to devices with ECC.
*
* Requires ECC Mode to be enabled.
*
* @note On MIG-enabled GPUs, per instance information can be queried using specific
*       MIG device handles. Per instance information is currently only supported for
*       non-DRAM uncorrectable volatile errors. Querying volatile errors using device
*       handles is currently not supported.
*
* See \ref nvmlMemoryErrorType_t for a description of available memory error types.\n
* See \ref nvmlEccCounterType_t for a description of available counter types.\n
* See \ref nvmlMemoryLocation_t for a description of available counter locations.\n
*
* @param device                               The identifier of the target device
* @param errorType                            Flag that specifies the type of error.
* @param counterType                          Flag that specifies the counter-type of the errors.
* @param locationType                         Specifies the location of the counter.
* @param count                                Reference in which to return the ECC counter
*
* @return
*         - \ref NVML_SUCCESS                 if \a count has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a bitTyp,e \a counterType or \a locationType is
*                                             invalid, or \a count is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support ECC error reporting in the specified memory
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device, nvmlMemoryErrorType_t errorType, nvmlEccCounterType_t counterType, nvmlMemoryLocation_t locationType, unsigned long long* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryErrorCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&errorType, sizeof(nvmlMemoryErrorType_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_write(&locationType, sizeof(nvmlMemoryLocation_t)) < 0 ||
        rpc_write(count, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current utilization rates for the device's major subsystems.
*
* For Fermi &tm; or newer fully supported devices.
*
* See \ref nvmlUtilization_t for details on available utilization rates.
*
* \note During driver initialization when ECC is enabled one can see high GPU and Memory Utilization readings.
*       This is caused by ECC Memory Scrubbing mechanism that is performed during driver initialization.
*
* @note On MIG-enabled GPUs, querying device utilization rates is not currently supported.
*
* @param device                               The identifier of the target device
* @param utilization                          Reference in which to return the utilization information
*
* @return
*         - \ref NVML_SUCCESS                 if \a utilization has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a utilization is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetUtilizationRates(nvmlDevice_t device, nvmlUtilization_t* utilization) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetUtilizationRates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(nvmlUtilization_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current utilization and sampling size in microseconds for the Encoder
*
* For Kepler &tm; or newer fully supported devices.
*
* @note On MIG-enabled GPUs, querying encoder utilization is not currently supported.
*
* @param device                               The identifier of the target device
* @param utilization                          Reference to an unsigned int for encoder utilization info
* @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
*
* @return
*         - \ref NVML_SUCCESS                 if \a utilization has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetEncoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current capacity of the device's encoder, as a percentage of maximum encoder capacity with valid values in the range 0-100.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param encoderQueryType                  Type of encoder to query
* @param encoderCapacity                   Reference to an unsigned int for the encoder capacity
*
* @return
*         - \ref NVML_SUCCESS                  if \a encoderCapacity is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a encoderCapacity is NULL, or \a device or \a encoderQueryType
*                                              are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED      if device does not support the encoder specified in \a encodeQueryType
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetEncoderCapacity(nvmlDevice_t device, nvmlEncoderType_t encoderQueryType, unsigned int* encoderCapacity) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&encoderQueryType, sizeof(nvmlEncoderType_t)) < 0 ||
        rpc_write(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current encoder statistics for a given device.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param sessionCount                      Reference to an unsigned int for count of active encoder sessions
* @param averageFps                        Reference to an unsigned int for trailing average FPS of all active sessions
* @param averageLatency                    Reference to an unsigned int for encode latency in microseconds
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionCount, \a averageFps and \a averageLatency is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount, or \a device or \a averageFps,
*                                              or \a averageLatency is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetEncoderStats(nvmlDevice_t device, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves information about active encoder sessions on a target device.
*
* An array of active encoder sessions is returned in the caller-supplied buffer pointed at by \a sessionInfos. The
* array elememt count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accommodate the active session array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlEncoderSessionInfo_t array required in \a sessionCount.
* To query the number of active encoder sessions, call this function with *sessionCount = 0.  The code will return
* NVML_SUCCESS with number of active encoder sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfos                      Reference in which to return the session information
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionInfos is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount is NULL.
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_SUPPORTED      if this query is not supported by \a device
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetEncoderSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfos) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionInfos, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfos, *sessionCount * sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current utilization and sampling size in microseconds for the Decoder
*
* For Kepler &tm; or newer fully supported devices.
*
* @note On MIG-enabled GPUs, querying decoder utilization is not currently supported.
*
* @param device                               The identifier of the target device
* @param utilization                          Reference to an unsigned int for decoder utilization info
* @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
*
* @return
*         - \ref NVML_SUCCESS                 if \a utilization has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetDecoderUtilization(nvmlDevice_t device, unsigned int* utilization, unsigned int* samplingPeriodUs) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDecoderUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(unsigned int)) < 0 ||
        rpc_write(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(unsigned int)) < 0 ||
        rpc_read(samplingPeriodUs, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the active frame buffer capture sessions statistics for a given device.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param fbcStats                          Reference to nvmlFBCStats_t structure containing NvFBC stats
*
* @return
*         - \ref NVML_SUCCESS                  if \a fbcStats is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a fbcStats is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetFBCStats(nvmlDevice_t device, nvmlFBCStats_t* fbcStats) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves information about active frame buffer capture sessions on a target device.
*
* An array of active FBC sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accomodate the active session array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlFBCSessionInfo_t array required in \a sessionCount.
* To query the number of active FBC sessions, call this function with *sessionCount = 0.  The code will return
* NVML_SUCCESS with number of active FBC sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @note hResolution, vResolution, averageFPS and averageLatency data for a FBC session returned in \a sessionInfo may
*       be zero if there are no new frames captured since the session started.
*
* @param device                            The identifier of the target device
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfo                       Reference in which to return the session information
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount is NULL.
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetFBCSessions(nvmlDevice_t device, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFBCSessions);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current and pending driver model for the device.
*
* For Fermi &tm; or newer fully supported devices.
* For windows only.
*
* On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
* to the device it must run in WDDM mode. TCC mode is preferred if a display is not attached.
*
* See \ref nvmlDriverModel_t for details on available driver models.
*
* @param device                               The identifier of the target device
* @param current                              Reference in which to return the current driver model
* @param pending                              Reference in which to return the pending driver model
*
* @return
*         - \ref NVML_SUCCESS                 if either \a current and/or \a pending have been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or both \a current and \a pending are NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform is not windows
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceSetDriverModel()
*/
nvmlReturn_t nvmlDeviceGetDriverModel(nvmlDevice_t device, nvmlDriverModel_t* current, nvmlDriverModel_t* pending) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDriverModel);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(current, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_read(pending, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get VBIOS version of the device.
*
* For all products.
*
* The VBIOS version may change from time to time. It will not exceed 32 characters in length
* (including the NULL terminator).  See \ref nvmlConstants::NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE.
*
* @param device                               The identifier of the target device
* @param version                              Reference to which to return the VBIOS version
* @param length                               The maximum allowed length of the string returned in \a version
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a version is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVbiosVersion(nvmlDevice_t device, char* version, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVbiosVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(version, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get Bridge Chip Information for all the bridge chips on the board.
*
* For all fully supported products.
* Only applicable to multi-GPU products.
*
* @param device                                The identifier of the target device
* @param bridgeHierarchy                       Reference to the returned bridge chip Hierarchy
*
* @return
*         - \ref NVML_SUCCESS                 if bridge chip exists
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a bridgeInfo is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if bridge chip not supported on the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
*/
nvmlReturn_t nvmlDeviceGetBridgeChipInfo(nvmlDevice_t device, nvmlBridgeChipHierarchy_t* bridgeHierarchy) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBridgeChipInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bridgeHierarchy, sizeof(nvmlBridgeChipHierarchy_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get information about processes with a compute context on a device
*
* For Fermi &tm; or newer fully supported devices.
*
* This function returns information only about compute running processes (e.g. CUDA application which have
* active context). Any graphics applications (e.g. using OpenGL, DirectX) won't be listed by this function.
*
* To query the current number of running compute processes, call this function with *infoCount = 0. The
* return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
* \a infos is allowed to be NULL.
*
* The usedGpuMemory field returned is all of the memory used by the application.
*
* Keep in mind that information returned by this call is dynamic and the number of elements might change in
* time. Allocate more space for \a infos table in case new compute processes are spawned.
*
* @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
*       the caller has appropriate privileges. Per-instance information can be queried by using
*       specific MIG device handles.
*       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
*
* @param device                               The device handle or MIG device handle
* @param infoCount                            Reference in which to provide the \a infos array size, and
*                                             to return the number of returned elements
* @param infos                                Reference in which to return the process information
*
* @return
*         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
*                                             \a infoCount will contain minimal amount of space necessary for
*                                             the call to complete
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see \ref nvmlSystemGetProcessName
*/
nvmlReturn_t nvmlDeviceGetComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get information about processes with a graphics context on a device
*
* For Kepler &tm; or newer fully supported devices.
*
* This function returns information only about graphics based processes
* (eg. applications using OpenGL, DirectX)
*
* To query the current number of running graphics processes, call this function with *infoCount = 0. The
* return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
* \a infos is allowed to be NULL.
*
* The usedGpuMemory field returned is all of the memory used by the application.
*
* Keep in mind that information returned by this call is dynamic and the number of elements might change in
* time. Allocate more space for \a infos table in case new graphics processes are spawned.
*
* @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
*       the caller has appropriate privileges. Per-instance information can be queried by using
*       specific MIG device handles.
*       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
*
* @param device                               The device handle or MIG device handle
* @param infoCount                            Reference in which to provide the \a infos array size, and
*                                             to return the number of returned elements
* @param infos                                Reference in which to return the process information
*
* @return
*         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
*                                             \a infoCount will contain minimal amount of space necessary for
*                                             the call to complete
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see \ref nvmlSystemGetProcessName
*/
nvmlReturn_t nvmlDeviceGetGraphicsRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGraphicsRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get information about processes with a MPS compute context on a device
*
* For Volta &tm; or newer fully supported devices.
*
* This function returns information only about compute running processes (e.g. CUDA application which have
* active context) utilizing MPS. Any graphics applications (e.g. using OpenGL, DirectX) won't be listed by
* this function.
*
* To query the current number of running compute processes, call this function with *infoCount = 0. The
* return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if none are running. For this call
* \a infos is allowed to be NULL.
*
* The usedGpuMemory field returned is all of the memory used by the application.
*
* Keep in mind that information returned by this call is dynamic and the number of elements might change in
* time. Allocate more space for \a infos table in case new compute processes are spawned.
*
* @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
*       the caller has appropriate privileges. Per-instance information can be queried by using
*       specific MIG device handles.
*       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
*
* @param device                               The device handle or MIG device handle
* @param infoCount                            Reference in which to provide the \a infos array size, and
*                                             to return the number of returned elements
* @param infos                                Reference in which to return the process information
*
* @return
*         - \ref NVML_SUCCESS                 if \a infoCount and \a infos have been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
*                                             \a infoCount will contain minimal amount of space necessary for
*                                             the call to complete
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see \ref nvmlSystemGetProcessName
*/
nvmlReturn_t nvmlDeviceGetMPSComputeRunningProcesses_v3(nvmlDevice_t device, unsigned int* infoCount, nvmlProcessInfo_t* infos) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3);
    if (request_id < 0 ||
        rpc_write(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(infoCount, sizeof(unsigned int)) < 0 ||
        rpc_read(infos, *infoCount * sizeof(nvmlProcessInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Check if the GPU devices are on the same physical board.
*
* For all fully supported products.
*
* @param device1                               The first GPU device
* @param device2                               The second GPU device
* @param onSameBoard                           Reference in which to return the status.
*                                              Non-zero indicates that the GPUs are on the same board.
*
* @return
*         - \ref NVML_SUCCESS                 if \a onSameBoard has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a dev1 or \a dev2 are invalid or \a onSameBoard is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the either GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceOnSameBoard(nvmlDevice_t device1, nvmlDevice_t device2, int* onSameBoard) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceOnSameBoard);
    if (request_id < 0 ||
        rpc_write(&device1, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&device2, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(onSameBoard, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(onSameBoard, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the root/admin permissions on the target API. See \a nvmlRestrictedAPI_t for the list of supported APIs.
* If an API is restricted only root users can call that API. See \a nvmlDeviceSetAPIRestriction to change current permissions.
*
* For all fully supported products.
*
* @param device                               The identifier of the target device
* @param apiType                              Target API type for this operation
* @param isRestricted                         Reference in which to return the current restriction
*                                             NVML_FEATURE_ENABLED indicates that the API is root-only
*                                             NVML_FEATURE_DISABLED indicates that the API is accessible to all users
*
* @return
*         - \ref NVML_SUCCESS                 if \a isRestricted has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a apiType incorrect or \a isRestricted is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device or the device does not support
*                                                 the feature that is being queried (E.G. Enabling/disabling Auto Boosted clocks is
*                                                 not supported by the device)
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlRestrictedAPI_t
*/
nvmlReturn_t nvmlDeviceGetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t* isRestricted) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAPIRestriction);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_write(isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets recent samples for the GPU.
*
* For Kepler &tm; or newer fully supported devices.
*
* Based on type, this method can be used to fetch the power, utilization or clock samples maintained in the buffer by
* the driver.
*
* Power, Utilization and Clock samples are returned as type "unsigned int" for the union nvmlValue_t.
*
* To get the size of samples that user needs to allocate, the method is invoked with samples set to NULL.
* The returned samplesCount will provide the number of samples that can be queried. The user needs to
* allocate the buffer with size as samplesCount * sizeof(nvmlSample_t).
*
* lastSeenTimeStamp represents CPU timestamp in microseconds. Set it to 0 to fetch all the samples maintained by the
* underlying buffer. Set lastSeenTimeStamp to one of the timeStamps retrieved from the date of the previous query
* to get more recent samples.
*
* This method fetches the number of entries which can be accommodated in the provided samples array, and the
* reference samplesCount is updated to indicate how many samples were actually retrieved. The advantage of using this
* method for samples in contrast to polling via existing methods is to get get higher frequency data at lower polling cost.
*
* @note On MIG-enabled GPUs, querying the following sample types, NVML_GPU_UTILIZATION_SAMPLES, NVML_MEMORY_UTILIZATION_SAMPLES
*       NVML_ENC_UTILIZATION_SAMPLES and NVML_DEC_UTILIZATION_SAMPLES, is not currently supported.
*
* @param device                        The identifier for the target device
* @param type                          Type of sampling event
* @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
* @param sampleValType                 Output parameter to represent the type of sample value as described in nvmlSampleVal_t
* @param sampleCount                   Reference to provide the number of elements which can be queried in samples array
* @param samples                       Reference in which samples are returned
* @return
*         - \ref NVML_SUCCESS                 if samples are successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a samplesCount is NULL or
*                                             reference to \a sampleCount is 0 for non null \a samples
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetSamples(nvmlDevice_t device, nvmlSamplingType_t type, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* sampleCount, nvmlSample_t* samples) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSamples);
    if (request_id < 0 ||
        rpc_write(sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlSamplingType_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(samples, *sampleCount * sizeof(nvmlSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sampleCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(samples, *sampleCount * sizeof(nvmlSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets Total, Available and Used size of BAR1 memory.
*
* BAR1 is used to map the FB (device memory) so that it can be directly accessed by the CPU or by 3rd party
* devices (peer-to-peer on the PCIE bus).
*
* @note In MIG mode, if device handle is provided, the API returns aggregate
*       information, only if the caller has appropriate privileges. Per-instance
*       information can be queried by using specific MIG device handles.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param bar1Memory                           Reference in which BAR1 memory
*                                             information is returned.
*
* @return
*         - \ref NVML_SUCCESS                 if BAR1 memory is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a bar1Memory is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
*/
nvmlReturn_t nvmlDeviceGetBAR1MemoryInfo(nvmlDevice_t device, nvmlBAR1Memory_t* bar1Memory) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBAR1MemoryInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bar1Memory, sizeof(nvmlBAR1Memory_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the duration of time during which the device was throttled (lower than requested clocks) due to power
* or thermal constraints.
*
* The method is important to users who are tying to understand if their GPUs throttle at any point during their applications. The
* difference in violation times at two different reference times gives the indication of GPU throttling event.
*
* Violation for thermal capping is not supported at this time.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param perfPolicyType                       Represents Performance policy which can trigger GPU throttling
* @param violTime                             Reference to which violation time related information is returned
*
*
* @return
*         - \ref NVML_SUCCESS                 if violation time is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a perfPolicyType is invalid, or \a violTime is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetViolationStatus(nvmlDevice_t device, nvmlPerfPolicyType_t perfPolicyType, nvmlViolationTime_t* violTime) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetViolationStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&perfPolicyType, sizeof(nvmlPerfPolicyType_t)) < 0 ||
        rpc_write(violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(violTime, sizeof(nvmlViolationTime_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the device's interrupt number
*
* @param device                               The identifier of the target device
* @param irqNum                               The interrupt number associated with the specified device
*
* @return
*         - \ref NVML_SUCCESS                 if irq number is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a irqNum is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetIrqNum(nvmlDevice_t device, unsigned int* irqNum) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetIrqNum);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(irqNum, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(irqNum, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the device's core count
*
* @param device                               The identifier of the target device
* @param numCores                             The number of cores for the specified device
*
* @return
*         - \ref NVML_SUCCESS                 if Gpu core count is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a numCores is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetNumGpuCores(nvmlDevice_t device, unsigned int* numCores) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNumGpuCores);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(numCores, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numCores, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the devices power source
*
* @param device                               The identifier of the target device
* @param powerSource                          The power source of the device
*
* @return
*         - \ref NVML_SUCCESS                 if the current power source was successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a powerSource is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetPowerSource(nvmlDevice_t device, nvmlPowerSource_t* powerSource) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPowerSource);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(powerSource, sizeof(nvmlPowerSource_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the device's memory bus width
*
* @param device                               The identifier of the target device
* @param busWidth                             The devices's memory bus width
*
* @return
*         - \ref NVML_SUCCESS                 if the memory bus width is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a busWidth is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetMemoryBusWidth(nvmlDevice_t device, unsigned int* busWidth) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemoryBusWidth);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(busWidth, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(busWidth, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the device's PCIE Max Link speed in MBPS
*
* @param device                               The identifier of the target device
* @param maxSpeed                             The devices's PCIE Max Link speed in MBPS
*
* @return
*         - \ref NVML_SUCCESS                 if Pcie Max Link Speed is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a maxSpeed is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int* maxSpeed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieLinkMaxSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(maxSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the device's PCIe Link speed in Mbps
*
* @param device                               The identifier of the target device
* @param pcieSpeed                            The devices's PCIe Max Link speed in Mbps
*
* @return
*         - \ref NVML_SUCCESS                 if \a pcieSpeed has been retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pcieSpeed is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support PCIe speed getting
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPcieSpeed(nvmlDevice_t device, unsigned int* pcieSpeed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPcieSpeed);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pcieSpeed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Gets the device's Adaptive Clock status
*
* @param device                               The identifier of the target device
* @param adaptiveClockStatus                  The current adaptive clocking status
*
* @return
*         - \ref NVML_SUCCESS                 if the current adaptive clocking status is successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, or \a adaptiveClockStatus is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*
*/
nvmlReturn_t nvmlDeviceGetAdaptiveClockInfoStatus(nvmlDevice_t device, unsigned int* adaptiveClockStatus) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAdaptiveClockInfoStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(adaptiveClockStatus, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries the state of per process accounting mode.
*
* For Kepler &tm; or newer fully supported devices.
*
* See \ref nvmlDeviceGetAccountingStats for more details.
* See \ref nvmlDeviceSetAccountingMode
*
* @param device                               The identifier of the target device
* @param mode                                 Reference in which to return the current accounting mode
*
* @return
*         - \ref NVML_SUCCESS                 if the mode has been successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode are NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetAccountingMode(nvmlDevice_t device, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries process's accounting stats.
*
* For Kepler &tm; or newer fully supported devices.
*
* Accounting stats capture GPU utilization and other statistics across the lifetime of a process.
* Accounting stats can be queried during life time of the process and after its termination.
* The time field in \ref nvmlAccountingStats_t is reported as 0 during the lifetime of the process and
* updated to actual running time after its termination.
* Accounting stats are kept in a circular buffer, newly created processes overwrite information about old
* processes.
*
* See \ref nvmlAccountingStats_t for description of each returned metric.
* List of processes that can be queried can be retrieved from \ref nvmlDeviceGetAccountingPids.
*
* @note Accounting Mode needs to be on. See \ref nvmlDeviceGetAccountingMode.
* @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
*         queried since they don't contribute to GPU utilization.
* @note In case of pid collision stats of only the latest process (that terminated last) will be reported
*
* @warning On Kepler devices per process statistics are accurate only if there's one process running on a GPU.
*
* @param device                               The identifier of the target device
* @param pid                                  Process Id of the target process to query stats for
* @param stats                                Reference in which to return the process's accounting stats
*
* @return
*         - \ref NVML_SUCCESS                 if stats have been successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a stats are NULL
*         - \ref NVML_ERROR_NOT_FOUND         if process stats were not found
*         - \ref NVML_ERROR_NOT_SUPPORTED     if \a device doesn't support this feature or accounting mode is disabled
*                                              or on vGPU host.
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetAccountingBufferSize
*/
nvmlReturn_t nvmlDeviceGetAccountingStats(nvmlDevice_t device, unsigned int pid, nvmlAccountingStats_t* stats) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingStats);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries list of processes that can be queried for accounting stats. The list of processes returned
* can be in running or terminated state.
*
* For Kepler &tm; or newer fully supported devices.
*
* To just query the number of processes ready to be queried, call this function with *count = 0 and
* pids=NULL. The return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if list is empty.
*
* For more details see \ref nvmlDeviceGetAccountingStats.
*
* @note In case of PID collision some processes might not be accessible before the circular buffer is full.
*
* @param device                               The identifier of the target device
* @param count                                Reference in which to provide the \a pids array size, and
*                                               to return the number of elements ready to be queried
* @param pids                                 Reference in which to return list of process ids
*
* @return
*         - \ref NVML_SUCCESS                 if pids were successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a count is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if \a device doesn't support this feature or accounting mode is disabled
*                                              or on vGPU host.
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to
*                                                 expected value)
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetAccountingBufferSize
*/
nvmlReturn_t nvmlDeviceGetAccountingPids(nvmlDevice_t device, unsigned int* count, unsigned int* pids) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingPids);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the number of processes that the circular buffer with accounting pids can hold.
*
* For Kepler &tm; or newer fully supported devices.
*
* This is the maximum number of processes that accounting information will be stored for before information
* about oldest processes will get overwritten by information about new processes.
*
* @param device                               The identifier of the target device
* @param bufferSize                           Reference in which to provide the size (in number of elements)
*                                               of the circular buffer for accounting stats.
*
* @return
*         - \ref NVML_SUCCESS                 if buffer size was successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a bufferSize is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetAccountingStats
* @see nvmlDeviceGetAccountingPids
*/
nvmlReturn_t nvmlDeviceGetAccountingBufferSize(nvmlDevice_t device, unsigned int* bufferSize) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetAccountingBufferSize);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the list of retired pages by source, including pages that are pending retirement
* The address information provided from this API is the hardware address of the page that was retired.  Note
* that this does not match the virtual address used in CUDA, but will match the address information in XID 63
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param cause                             Filter page addresses by cause of retirement
* @param pageCount                         Reference in which to provide the \a addresses buffer size, and
*                                          to return the number of retired pages that match \a cause
*                                          Set to 0 to query the size without allocating an \a addresses buffer
* @param addresses                         Buffer to write the page addresses into
*
* @return
*         - \ref NVML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
*                                             matching page addresses.  \a pageCount is set to the needed size.
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or
*                                             \a addresses is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetRetiredPages(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages);
    if (request_id < 0 ||
        rpc_write(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the list of retired pages by source, including pages that are pending retirement
* The address information provided from this API is the hardware address of the page that was retired.  Note
* that this does not match the virtual address used in CUDA, but will match the address information in XID 63
*
* \note nvmlDeviceGetRetiredPages_v2 adds an additional timestamps paramter to return the time of each page's
*       retirement.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param cause                             Filter page addresses by cause of retirement
* @param pageCount                         Reference in which to provide the \a addresses buffer size, and
*                                          to return the number of retired pages that match \a cause
*                                          Set to 0 to query the size without allocating an \a addresses buffer
* @param addresses                         Buffer to write the page addresses into
* @param timestamps                        Buffer to write the timestamps of page retirement, additional for _v2
*
* @return
*         - \ref NVML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
*                                             matching page addresses.  \a pageCount is set to the needed size.
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or
*                                             \a addresses is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetRetiredPages_v2(nvmlDevice_t device, nvmlPageRetirementCause_t cause, unsigned int* pageCount, unsigned long long* addresses, unsigned long long* timestamps) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPages_v2);
    if (request_id < 0 ||
        rpc_write(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&cause, sizeof(nvmlPageRetirementCause_t)) < 0 ||
        rpc_write(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_write(timestamps, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pageCount, sizeof(unsigned int)) < 0 ||
        rpc_read(addresses, *pageCount * sizeof(unsigned long long)) < 0 ||
        rpc_read(timestamps, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Check if any pages are pending retirement and need a reboot to fully retire.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param isPending                         Reference in which to return the pending status
*
* @return
*         - \ref NVML_SUCCESS                 if \a isPending was populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a isPending is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetRetiredPagesPendingStatus(nvmlDevice_t device, nvmlEnableState_t* isPending) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRetiredPagesPendingStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isPending, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get number of remapped rows. The number of rows reported will be based on
* the cause of the remapping. isPending indicates whether or not there are
* pending remappings. A reset will be required to actually remap the row.
* failureOccurred will be set if a row remapping ever failed in the past. A
* pending remapping won't affect future work on the GPU since
* error-containment and dynamic page blacklisting will take care of that.
*
* @note On MIG-enabled GPUs with active instances, querying the number of
* remapped rows is not supported
*
* For Ampere &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param corrRows                             Reference for number of rows remapped due to correctable errors
* @param uncRows                              Reference for number of rows remapped due to uncorrectable errors
* @param isPending                            Reference for whether or not remappings are pending
* @param failureOccurred                      Reference that is set when a remapping has failed in the past
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a corrRows, \a uncRows, \a isPending or \a failureOccurred is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If MIG is enabled or if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           Unexpected error
*/
nvmlReturn_t nvmlDeviceGetRemappedRows(nvmlDevice_t device, unsigned int* corrRows, unsigned int* uncRows, unsigned int* isPending, unsigned int* failureOccurred) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRemappedRows);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(corrRows, sizeof(unsigned int)) < 0 ||
        rpc_write(uncRows, sizeof(unsigned int)) < 0 ||
        rpc_write(isPending, sizeof(unsigned int)) < 0 ||
        rpc_write(failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(corrRows, sizeof(unsigned int)) < 0 ||
        rpc_read(uncRows, sizeof(unsigned int)) < 0 ||
        rpc_read(isPending, sizeof(unsigned int)) < 0 ||
        rpc_read(failureOccurred, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get the row remapper histogram. Returns the remap availability for each bank
* on the GPU.
*
* @param device                               Device handle
* @param values                               Histogram values
*
* @return
*        - \ref NVML_SUCCESS                  On success
*        - \ref NVML_ERROR_UNKNOWN            On any unexpected error
*/
nvmlReturn_t nvmlDeviceGetRowRemapperHistogram(nvmlDevice_t device, nvmlRowRemapperHistogramValues_t* values) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetRowRemapperHistogram);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, sizeof(nvmlRowRemapperHistogramValues_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get architecture for device
*
* @param device                               The identifier of the target device
* @param arch                                 Reference where architecture is returned, if call successful.
*                                             Set to NVML_DEVICE_ARCH_* upon success
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device or \a arch (output refererence) are invalid
*/
nvmlReturn_t nvmlDeviceGetArchitecture(nvmlDevice_t device, nvmlDeviceArchitecture_t* arch) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetArchitecture);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(arch, sizeof(nvmlDeviceArchitecture_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the LED state for the unit. The LED can be either green (0) or amber (1).
*
* For S-class products.
* Requires root/admin permissions.
*
* This operation takes effect immediately.
*
*
* <b>Current S-Class products don't provide unique LEDs for each unit. As such, both front
* and back LEDs will be toggled in unison regardless of which unit is specified with this command.</b>
*
* See \ref nvmlLedColor_t for available colors.
*
* @param unit                                 The identifier of the target unit
* @param color                                The target LED color
*
* @return
*         - \ref NVML_SUCCESS                 if the LED color has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a unit or \a color is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this is not an S-class product
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlUnitGetLedState()
*/
nvmlReturn_t nvmlUnitSetLedState(nvmlUnit_t unit, nvmlLedColor_t color) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlUnitSetLedState);
    if (request_id < 0 ||
        rpc_write(&unit, sizeof(nvmlUnit_t)) < 0 ||
        rpc_write(&color, sizeof(nvmlLedColor_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the persistence mode for the device.
*
* For all products.
* For Linux only.
* Requires root/admin permissions.
*
* The persistence mode determines whether the GPU driver software is torn down after the last client
* exits.
*
* This operation takes effect immediately. It is not persistent across reboots. After each reboot the
* persistence mode is reset to "Disabled".
*
* See \ref nvmlEnableState_t for available modes.
*
* After calling this API with mode set to NVML_FEATURE_DISABLED on a device that has its own NUMA
* memory, the given device handle will no longer be valid, and to continue to interact with this
* device, a new handle should be obtained from one of the nvmlDeviceGetHandleBy*() APIs. This
* limitation is currently only applicable to devices that have a coherent NVLink connection to
* system memory.
*
* @param device                               The identifier of the target device
* @param mode                                 The target persistence mode
*
* @return
*         - \ref NVML_SUCCESS                 if the persistence mode was set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetPersistenceMode()
*/
nvmlReturn_t nvmlDeviceSetPersistenceMode(nvmlDevice_t device, nvmlEnableState_t mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetPersistenceMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the compute mode for the device.
*
* For all products.
* Requires root/admin permissions.
*
* The compute mode determines whether a GPU can be used for compute operations and whether it can
* be shared across contexts.
*
* This operation takes effect immediately. Under Linux it is not persistent across reboots and
* always resets to "Default". Under windows it is persistent.
*
* Under windows compute mode may only be set to DEFAULT when running in WDDM
*
* @note On MIG-enabled GPUs, compute mode would be set to DEFAULT and changing it is not supported.
*
* See \ref nvmlComputeMode_t for details on available compute modes.
*
* @param device                               The identifier of the target device
* @param mode                                 The target compute mode
*
* @return
*         - \ref NVML_SUCCESS                 if the compute mode was set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetComputeMode()
*/
nvmlReturn_t nvmlDeviceSetComputeMode(nvmlDevice_t device, nvmlComputeMode_t mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetComputeMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlComputeMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the ECC mode for the device.
*
* For Kepler &tm; or newer fully supported devices.
* Only applicable to devices with ECC.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher.
* Requires root/admin permissions.
*
* The ECC mode determines whether the GPU enables its ECC support.
*
* This operation takes effect after the next reboot.
*
* See \ref nvmlEnableState_t for details on available modes.
*
* @param device                               The identifier of the target device
* @param ecc                                  The target ECC mode
*
* @return
*         - \ref NVML_SUCCESS                 if the ECC mode was set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a ecc is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetEccMode()
*/
nvmlReturn_t nvmlDeviceSetEccMode(nvmlDevice_t device, nvmlEnableState_t ecc) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetEccMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&ecc, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Clear the ECC error and other memory error counts for the device.
*
* For Kepler &tm; or newer fully supported devices.
* Only applicable to devices with ECC.
* Requires \a NVML_INFOROM_ECC version 2.0 or higher to clear aggregate location-based ECC counts.
* Requires \a NVML_INFOROM_ECC version 1.0 or higher to clear all other ECC counts.
* Requires root/admin permissions.
* Requires ECC Mode to be enabled.
*
* Sets all of the specified ECC counters to 0, including both detailed and total counts.
*
* This operation takes effect immediately.
*
* See \ref nvmlMemoryErrorType_t for details on available counter types.
*
* @param device                               The identifier of the target device
* @param counterType                          Flag that indicates which type of errors should be cleared.
*
* @return
*         - \ref NVML_SUCCESS                 if the error counts were cleared
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a counterType is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see
*      - nvmlDeviceGetDetailedEccErrors()
*      - nvmlDeviceGetTotalEccErrors()
*/
nvmlReturn_t nvmlDeviceClearEccErrorCounts(nvmlDevice_t device, nvmlEccCounterType_t counterType) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceClearEccErrorCounts);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&counterType, sizeof(nvmlEccCounterType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the driver model for the device.
*
* For Fermi &tm; or newer fully supported devices.
* For windows only.
* Requires root/admin permissions.
*
* On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
* to the device it must run in WDDM mode.
*
* It is possible to force the change to WDM (TCC) while the display is still attached with a force flag (nvmlFlagForce).
* This should only be done if the host is subsequently powered down and the display is detached from the device
* before the next reboot.
*
* This operation takes effect after the next reboot.
*
* Windows driver model may only be set to WDDM when running in DEFAULT compute mode.
*
* Change driver model to WDDM is not supported when GPU doesn't support graphics acceleration or
* will not support it after reboot. See \ref nvmlDeviceSetGpuOperationMode.
*
* See \ref nvmlDriverModel_t for details on available driver models.
* See \ref nvmlFlagDefault and \ref nvmlFlagForce
*
* @param device                               The identifier of the target device
* @param driverModel                          The target driver model
* @param flags                                Flags that change the default behavior
*
* @return
*         - \ref NVML_SUCCESS                 if the driver model has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a driverModel is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform is not windows or the device does not support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetDriverModel()
*/
nvmlReturn_t nvmlDeviceSetDriverModel(nvmlDevice_t device, nvmlDriverModel_t driverModel, unsigned int flags) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetDriverModel);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&driverModel, sizeof(nvmlDriverModel_t)) < 0 ||
        rpc_write(&flags, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set clocks that device will lock to.
*
* Sets the clocks that the device will be running at to the value in the range of minGpuClockMHz to maxGpuClockMHz.
* Setting this will supercede application clock values and take effect regardless if a cuda app is running.
* See /ref nvmlDeviceSetApplicationsClocks
*
* Can be used as a setting to request constant performance.
*
* This can be called with a pair of integer clock frequencies in MHz, or a pair of /ref nvmlClockLimitId_t values.
* See the table below for valid combinations of these values.
*
* minGpuClock | maxGpuClock | Effect
* ------------+-------------+--------------------------------------------------
*     tdp     |     tdp     | Lock clock to TDP
*  unlimited  |     tdp     | Upper bound is TDP but clock may drift below this
*     tdp     |  unlimited  | Lower bound is TDP but clock may boost above this
*  unlimited  |  unlimited  | Unlocked (== nvmlDeviceResetGpuLockedClocks)
*
* If one arg takes one of these values, the other must be one of these values as
* well. Mixed numeric and symbolic calls return NVML_ERROR_INVALID_ARGUMENT.
*
* Requires root/admin permissions.
*
* After system reboot or driver reload applications clocks go back to their default value.
* See \ref nvmlDeviceResetGpuLockedClocks.
*
* For Volta &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param minGpuClockMHz                       Requested minimum gpu clock in MHz
* @param maxGpuClockMHz                       Requested maximum gpu clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if new settings were successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minGpuClockMHz and \a maxGpuClockMHz
*                                                 is not a valid clock combination
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetGpuLockedClocks(nvmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&minGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&maxGpuClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Resets the gpu clock to the default value
*
* This is the gpu clock that will be used after system reboot or driver reload.
* Default values are idle clocks, but the current values can be changed using \ref nvmlDeviceSetApplicationsClocks.
*
* @see nvmlDeviceSetGpuLockedClocks
*
* For Volta &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if new settings were successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceResetGpuLockedClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceResetGpuLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set memory clocks that device will lock to.
*
* Sets the device's memory clocks to the value in the range of minMemClockMHz to maxMemClockMHz.
* Setting this will supersede application clock values and take effect regardless of whether a cuda app is running.
* See /ref nvmlDeviceSetApplicationsClocks
*
* Can be used as a setting to request constant performance.
*
* Requires root/admin permissions.
*
* After system reboot or driver reload applications clocks go back to their default value.
* See \ref nvmlDeviceResetMemoryLockedClocks.
*
* For Ampere &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param minMemClockMHz                       Requested minimum memory clock in MHz
* @param maxMemClockMHz                       Requested maximum memory clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if new settings were successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a minGpuClockMHz and \a maxGpuClockMHz
*                                                 is not a valid clock combination
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetMemoryLockedClocks(nvmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemoryLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&minMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&maxMemClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Resets the memory clock to the default value
*
* This is the memory clock that will be used after system reboot or driver reload.
* Default values are idle clocks, but the current values can be changed using \ref nvmlDeviceSetApplicationsClocks.
*
* @see nvmlDeviceSetMemoryLockedClocks
*
* For Ampere &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if new settings were successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceResetMemoryLockedClocks(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceResetMemoryLockedClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set clocks that applications will lock to.
*
* Sets the clocks that compute and graphics applications will be running at.
* e.g. CUDA driver requests these clocks during context creation which means this property
* defines clocks at which CUDA applications will be running unless some overspec event
* occurs (e.g. over power, over thermal or external HW brake).
*
* Can be used as a setting to request constant performance.
*
* On Pascal and newer hardware, this will automatically disable automatic boosting of clocks.
*
* On K80 and newer Kepler and Maxwell GPUs, users desiring fixed performance should also call
* \ref nvmlDeviceSetAutoBoostedClocksEnabled to prevent clocks from automatically boosting
* above the clock value being set.
*
* For Kepler &tm; or newer non-GeForce fully supported devices and Maxwell or newer GeForce devices.
* Requires root/admin permissions.
*
* See \ref nvmlDeviceGetSupportedMemoryClocks and \ref nvmlDeviceGetSupportedGraphicsClocks
* for details on how to list available clocks combinations.
*
* After system reboot or driver reload applications clocks go back to their default value.
* See \ref nvmlDeviceResetApplicationsClocks.
*
* @param device                               The identifier of the target device
* @param memClockMHz                          Requested memory clock in MHz
* @param graphicsClockMHz                     Requested graphics clock in MHz
*
* @return
*         - \ref NVML_SUCCESS                 if new settings were successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a memClockMHz and \a graphicsClockMHz
*                                                 is not a valid clock combination
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetApplicationsClocks(nvmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetApplicationsClocks);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&memClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(&graphicsClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the frequency monitor fault status for the device.
*
* For Ampere &tm; or newer fully supported devices.
* Requires root user.
*
* See \ref nvmlClkMonStatus_t for details on decoding the status output.
*
* @param device                               The identifier of the target device
* @param status                               Reference in which to return the clkmon fault status
*
* @return
*         - \ref NVML_SUCCESS                 if \a status has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a status is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetClkMonStatus()
*/
nvmlReturn_t nvmlDeviceGetClkMonStatus(nvmlDevice_t device, nvmlClkMonStatus_t* status) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetClkMonStatus);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(status, sizeof(nvmlClkMonStatus_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set new power limit of this device.
*
* For Kepler &tm; or newer fully supported devices.
* Requires root/admin permissions.
*
* See \ref nvmlDeviceGetPowerManagementLimitConstraints to check the allowed ranges of values.
*
* \note Limit is not persistent across reboots or driver unloads.
* Enable persistent mode to prevent driver from unloading when no application is using the device.
*
* @param device                               The identifier of the target device
* @param limit                                Power management limit in milliwatts to set
*
* @return
*         - \ref NVML_SUCCESS                 if \a limit has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a defaultLimit is out of range
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceGetPowerManagementLimitConstraints
* @see nvmlDeviceGetPowerManagementDefaultLimit
*/
nvmlReturn_t nvmlDeviceSetPowerManagementLimit(nvmlDevice_t device, unsigned int limit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetPowerManagementLimit);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&limit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Sets new GOM. See \a nvmlGpuOperationMode_t for details.
*
* For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
* Modes \ref NVML_GOM_LOW_DP and \ref NVML_GOM_ALL_ON are supported on fully supported GeForce products.
* Not supported on Quadro &reg; and Tesla &tm; C-class products.
* Requires root/admin permissions.
*
* Changing GOMs requires a reboot.
* The reboot requirement might be removed in the future.
*
* Compute only GOMs don't support graphics acceleration. Under windows switching to these GOMs when
* pending driver model is WDDM is not supported. See \ref nvmlDeviceSetDriverModel.
*
* @param device                               The identifier of the target device
* @param mode                                 Target GOM
*
* @return
*         - \ref NVML_SUCCESS                 if \a mode has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a mode incorrect
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support GOM or specific mode
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlGpuOperationMode_t
* @see nvmlDeviceGetGpuOperationMode
*/
nvmlReturn_t nvmlDeviceSetGpuOperationMode(nvmlDevice_t device, nvmlGpuOperationMode_t mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpuOperationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlGpuOperationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Changes the root/admin restructions on certain APIs. See \a nvmlRestrictedAPI_t for the list of supported APIs.
* This method can be used by a root/admin user to give non-root/admin access to certain otherwise-restricted APIs.
* The new setting lasts for the lifetime of the NVIDIA driver; it is not persistent. See \a nvmlDeviceGetAPIRestriction
* to query the current restriction settings.
*
* For Kepler &tm; or newer fully supported devices.
* Requires root/admin permissions.
*
* @param device                               The identifier of the target device
* @param apiType                              Target API type for this operation
* @param isRestricted                         The target restriction
*
* @return
*         - \ref NVML_SUCCESS                 if \a isRestricted has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a apiType incorrect
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support changing API restrictions or the device does not support
*                                                 the feature that api restrictions are being set for (E.G. Enabling/disabling auto
*                                                 boosted clocks is not supported by the device)
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlRestrictedAPI_t
*/
nvmlReturn_t nvmlDeviceSetAPIRestriction(nvmlDevice_t device, nvmlRestrictedAPI_t apiType, nvmlEnableState_t isRestricted) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetAPIRestriction);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&apiType, sizeof(nvmlRestrictedAPI_t)) < 0 ||
        rpc_write(&isRestricted, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Enables or disables per process accounting.
*
* For Kepler &tm; or newer fully supported devices.
* Requires root/admin permissions.
*
* @note This setting is not persistent and will default to disabled after driver unloads.
*       Enable persistence mode to be sure the setting doesn't switch off to disabled.
*
* @note Enabling accounting mode has no negative impact on the GPU performance.
*
* @note Disabling accounting clears all accounting pids information.
*
* @note On MIG-enabled GPUs, accounting mode would be set to DISABLED and changing it is not supported.
*
* See \ref nvmlDeviceGetAccountingMode
* See \ref nvmlDeviceGetAccountingStats
* See \ref nvmlDeviceClearAccountingPids
*
* @param device                               The identifier of the target device
* @param mode                                 The target accounting mode
*
* @return
*         - \ref NVML_SUCCESS                 if the new mode has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a mode are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetAccountingMode(nvmlDevice_t device, nvmlEnableState_t mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Clears accounting information about all processes that have already terminated.
*
* For Kepler &tm; or newer fully supported devices.
* Requires root/admin permissions.
*
* See \ref nvmlDeviceGetAccountingMode
* See \ref nvmlDeviceGetAccountingStats
* See \ref nvmlDeviceSetAccountingMode
*
* @param device                               The identifier of the target device
*
* @return
*         - \ref NVML_SUCCESS                 if accounting information has been cleared
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceClearAccountingPids(nvmlDevice_t device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the state of the device's NvLink for the link specified
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param isActive                             \a nvmlEnableState_t where NVML_FEATURE_ENABLED indicates that
*                                             the link is active and NVML_FEATURE_DISABLED indicates it
*                                             is inactive
*
* @return
*         - \ref NVML_SUCCESS                 if \a isActive has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a isActive is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t* isActive) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isActive, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the version of the device's NvLink for the link specified
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param version                              Requested NvLink version
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a version is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkVersion(nvmlDevice_t device, unsigned int link, unsigned int* version) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(version, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the requested capability from the device's NvLink for the link specified
* Please refer to the \a nvmlNvLinkCapability_t structure for the specific caps that can be queried
* The return value should be treated as a boolean.
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param capability                           Specifies the \a nvmlNvLinkCapability_t to be queried
* @param capResult                            A boolean for the queried capability indicating that feature is available
*
* @return
*         - \ref NVML_SUCCESS                 if \a capResult has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a capability is invalid or \a capResult is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link, nvmlNvLinkCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkCapability);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&capability, sizeof(nvmlNvLinkCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the PCI information for the remote node on a NvLink link
* Note: pciSubSystemId is not filled in this function and is indeterminate
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param pci                                  \a nvmlPciInfo_t of the remote node for the specified link
*
* @return
*         - \ref NVML_SUCCESS                 if \a pci has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid or \a pci is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t* pci) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pci, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the specified error counter value
* Please refer to \a nvmlNvLinkErrorCounter_t for error counters that are available
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param counter                              Specifies the NvLink counter to be queried
* @param counterValue                         Returned counter value
*
* @return
*         - \ref NVML_SUCCESS                 if \a counter has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a counter is invalid or \a counterValue is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkErrorCounter(nvmlDevice_t device, unsigned int link, nvmlNvLinkErrorCounter_t counter, unsigned long long* counterValue) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkErrorCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(nvmlNvLinkErrorCounter_t)) < 0 ||
        rpc_write(counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(counterValue, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Resets all error counters to zero
* Please refer to \a nvmlNvLinkErrorCounter_t for the list of error counters that are reset
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
*
* @return
*         - \ref NVML_SUCCESS                 if the reset is successful
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a link is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceResetNvLinkErrorCounters(nvmlDevice_t device, unsigned int link) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkErrorCounters);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Deprecated: Setting utilization counter control is no longer supported.
*
* Set the NVLINK utilization counter control information for the specified counter, 0 or 1.
* Please refer to \a nvmlNvLinkUtilizationControl_t for the structure definition.  Performs a reset
* of the counters if the reset parameter is non-zero.
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param counter                              Specifies the counter that should be set (0 or 1).
* @param link                                 Specifies the NvLink link to be queried
* @param control                              A reference to the \a nvmlNvLinkUtilizationControl_t to set
* @param reset                                Resets the counters on set if non-zero
*
* @return
*         - \ref NVML_SUCCESS                 if the control has been set successfully
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control, unsigned int reset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_write(&reset, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Deprecated: Getting utilization counter control is no longer supported.
*
* Get the NVLINK utilization counter control information for the specified counter, 0 or 1.
* Please refer to \a nvmlNvLinkUtilizationControl_t for the structure definition
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param counter                              Specifies the counter that should be set (0 or 1).
* @param link                                 Specifies the NvLink link to be queried
* @param control                              A reference to the \a nvmlNvLinkUtilizationControl_t to place information
*
* @return
*         - \ref NVML_SUCCESS                 if the control has been set successfully
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkUtilizationControl(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlNvLinkUtilizationControl_t* control) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationControl);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(control, sizeof(nvmlNvLinkUtilizationControl_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Deprecated: Use \ref nvmlDeviceGetFieldValues with NVML_FI_DEV_NVLINK_THROUGHPUT_* as field values instead.
*
* Retrieve the NVLINK utilization counter based on the current control for a specified counter.
* In general it is good practice to use \a nvmlDeviceSetNvLinkUtilizationControl
*  before reading the utilization counters as they have no default state
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param counter                              Specifies the counter that should be read (0 or 1).
* @param rxcounter                            Receive counter return value
* @param txcounter                            Transmit counter return value
*
* @return
*         - \ref NVML_SUCCESS                 if \a rxcounter and \a txcounter have been successfully set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a counter, or \a link is invalid or \a rxcounter or \a txcounter are NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, unsigned long long* rxcounter, unsigned long long* txcounter) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_write(txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(rxcounter, sizeof(unsigned long long)) < 0 ||
        rpc_read(txcounter, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Deprecated: Freezing NVLINK utilization counters is no longer supported.
*
* Freeze the NVLINK utilization counters
* Both the receive and transmit counters are operated on by this function
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be queried
* @param counter                              Specifies the counter that should be frozen (0 or 1).
* @param freeze                               NVML_FEATURE_ENABLED = freeze the receive and transmit counters
*                                             NVML_FEATURE_DISABLED = unfreeze the receive and transmit counters
*
* @return
*         - \ref NVML_SUCCESS                 if counters were successfully frozen or unfrozen
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, \a counter, or \a freeze is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceFreezeNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter, nvmlEnableState_t freeze) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceFreezeNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_write(&freeze, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Deprecated: Resetting NVLINK utilization counters is no longer supported.
*
* Reset the NVLINK utilization counters
* Both the receive and transmit counters are operated on by this function
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                               The identifier of the target device
* @param link                                 Specifies the NvLink link to be reset
* @param counter                              Specifies the counter that should be reset (0 or 1)
*
* @return
*         - \ref NVML_SUCCESS                 if counters were successfully reset
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a link, or \a counter is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceResetNvLinkUtilizationCounter(nvmlDevice_t device, unsigned int link, unsigned int counter) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceResetNvLinkUtilizationCounter);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(&counter, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get the NVLink device type of the remote device connected over the given link.
*
* @param device                                The device handle of the target GPU
* @param link                                  The NVLink link index on the target GPU
* @param pNvLinkDeviceType                     Pointer in which the output remote device type is returned
*
* @return
*         - \ref NVML_SUCCESS                  if \a pNvLinkDeviceType has been set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_NOT_SUPPORTED      if NVLink is not supported
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device or \a link is invalid, or
*                                              \a pNvLinkDeviceType is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is
*                                              otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link, nvmlIntNvLinkDeviceType_t* pNvLinkDeviceType) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetNvLinkRemoteDeviceType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&link, sizeof(unsigned int)) < 0 ||
        rpc_write(pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pNvLinkDeviceType, sizeof(nvmlIntNvLinkDeviceType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Create an empty set of events.
* Event set should be freed by \ref nvmlEventSetFree
*
* For Fermi &tm; or newer fully supported devices.
* @param set                                  Reference in which to return the event handle
*
* @return
*         - \ref NVML_SUCCESS                 if the event has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a set is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlEventSetFree
*/
nvmlReturn_t nvmlEventSetCreate(nvmlEventSet_t* set) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlEventSetCreate);
    if (request_id < 0 ||
        rpc_write(set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Starts recording of events on a specified devices and add the events to specified \ref nvmlEventSet_t
*
* For Fermi &tm; or newer fully supported devices.
* Ecc events are available only on ECC enabled devices (see \ref nvmlDeviceGetTotalEccErrors)
* Power capping events are available only on Power Management enabled devices (see \ref nvmlDeviceGetPowerManagementMode)
*
* For Linux only.
*
* \b IMPORTANT: Operations on \a set are not thread safe
*
* This call starts recording of events on specific device.
* All events that occurred before this call are not recorded.
* Checking if some event occurred can be done with \ref nvmlEventSetWait_v2
*
* If function reports NVML_ERROR_UNKNOWN, event set is in undefined state and should be freed.
* If function reports NVML_ERROR_NOT_SUPPORTED, event set can still be used. None of the requested eventTypes
*     are registered in that case.
*
* @param device                               The identifier of the target device
* @param eventTypes                           Bitmask of \ref nvmlEventType to record
* @param set                                  Set to which add new event types
*
* @return
*         - \ref NVML_SUCCESS                 if the event has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a eventTypes is invalid or \a set is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the platform does not support this feature or some of requested event types
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlEventType
* @see nvmlDeviceGetSupportedEventTypes
* @see nvmlEventSetWait
* @see nvmlEventSetFree
*/
nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t device, unsigned long long eventTypes, nvmlEventSet_t set) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceRegisterEvents);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns information about events supported on device
*
* For Fermi &tm; or newer fully supported devices.
*
* Events are not supported on Windows. So this function returns an empty mask in \a eventTypes on Windows.
*
* @param device                               The identifier of the target device
* @param eventTypes                           Reference in which to return bitmask of supported events
*
* @return
*         - \ref NVML_SUCCESS                 if the eventTypes has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a eventType is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlEventType
* @see nvmlDeviceRegisterEvents
*/
nvmlReturn_t nvmlDeviceGetSupportedEventTypes(nvmlDevice_t device, unsigned long long* eventTypes) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedEventTypes);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eventTypes, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Waits on events and delivers events
*
* For Fermi &tm; or newer fully supported devices.
*
* If some events are ready to be delivered at the time of the call, function returns immediately.
* If there are no events ready to be delivered, function sleeps till event arrives
* but not longer than specified timeout. This function in certain conditions can return before
* specified timeout passes (e.g. when interrupt arrives)
*
* On Windows, in case of xid error, the function returns the most recent xid error type seen by the system.
* If there are multiple xid errors generated before nvmlEventSetWait is invoked then the last seen xid error
* type is returned for all xid error events.
*
* On Linux, every xid error event would return the associated event data and other information if applicable.
*
* In MIG mode, if device handle is provided, the API reports all the events for the available instances,
* only if the caller has appropriate privileges. In absence of required privileges, only the events which
* affect all the instances (i.e. whole device) are reported.
*
* This API does not currently support per-instance event reporting using MIG device handles.
*
* @param set                                  Reference to set of events to wait on
* @param data                                 Reference in which to return event data
* @param timeoutms                            Maximum amount of wait time in milliseconds for registered event
*
* @return
*         - \ref NVML_SUCCESS                 if the data has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a data is NULL
*         - \ref NVML_ERROR_TIMEOUT           if no event arrived in specified timeout or interrupt arrived
*         - \ref NVML_ERROR_GPU_IS_LOST       if a GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlEventType
* @see nvmlDeviceRegisterEvents
*/
nvmlReturn_t nvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t* data, unsigned int timeoutms) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlEventSetWait_v2);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_write(data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_write(&timeoutms, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(data, sizeof(nvmlEventData_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Releases events in the set
*
* For Fermi &tm; or newer fully supported devices.
*
* @param set                                  Reference to events to be released
*
* @return
*         - \ref NVML_SUCCESS                 if the event has been successfully released
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlDeviceRegisterEvents
*/
nvmlReturn_t nvmlEventSetFree(nvmlEventSet_t set) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlEventSetFree);
    if (request_id < 0 ||
        rpc_write(&set, sizeof(nvmlEventSet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Modify the drain state of a GPU.  This method forces a GPU to no longer accept new incoming requests.
* Any new NVML process will no longer see this GPU.  Persistence mode for this GPU must be turned off before
* this call is made.
* Must be called as administrator.
* For Linux only.
*
* For Pascal &tm; or newer fully supported devices.
* Some Kepler devices supported.
*
* @param pciInfo                              The PCI address of the GPU drain state to be modified
* @param newState                             The drain state that should be entered, see \ref nvmlEnableState_t
*
* @return
*         - \ref NVML_SUCCESS                 if counters were successfully reset
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex or \a newState is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
*         - \ref NVML_ERROR_IN_USE            if the device has persistence mode turned on
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceModifyDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t newState) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceModifyDrainState);
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(&newState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Query the drain state of a GPU.  This method is used to check if a GPU is in a currently draining
* state.
* For Linux only.
*
* For Pascal &tm; or newer fully supported devices.
* Some Kepler devices supported.
*
* @param pciInfo                              The PCI address of the GPU drain state to be queried
* @param currentState                         The current drain state for this GPU, see \ref nvmlEnableState_t
*
* @return
*         - \ref NVML_SUCCESS                 if counters were successfully reset
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex or \a currentState is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceQueryDrainState(nvmlPciInfo_t* pciInfo, nvmlEnableState_t* currentState) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceQueryDrainState);
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_read(currentState, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* This method will remove the specified GPU from the view of both NVML and the NVIDIA kernel driver
* as long as no other processes are attached. If other processes are attached, this call will return
* NVML_ERROR_IN_USE and the GPU will be returned to its original "draining" state. Note: the
* only situation where a process can still be attached after nvmlDeviceModifyDrainState() is called
* to initiate the draining state is if that process was using, and is still using, a GPU before the
* call was made. Also note, persistence mode counts as an attachment to the GPU thus it must be disabled
* prior to this call.
*
* For long-running NVML processes please note that this will change the enumeration of current GPUs.
* For example, if there are four GPUs present and GPU1 is removed, the new enumeration will be 0-2.
* Also, device handles after the removed GPU will not be valid and must be re-established.
* Must be run as administrator.
* For Linux only.
*
* For Pascal &tm; or newer fully supported devices.
* Some Kepler devices supported.
*
* @param pciInfo                              The PCI address of the GPU to be removed
* @param gpuState                             Whether the GPU is to be removed, from the OS
*                                             see \ref nvmlDetachGpuState_t
* @param linkState                            Requested upstream PCIe link state, see \ref nvmlPcieLinkState_t
*
* @return
*         - \ref NVML_SUCCESS                 if counters were successfully reset
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a nvmlIndex is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
*         - \ref NVML_ERROR_IN_USE            if the device is still in use and cannot be removed
*/
nvmlReturn_t nvmlDeviceRemoveGpu_v2(nvmlPciInfo_t* pciInfo, nvmlDetachGpuState_t gpuState, nvmlPcieLinkState_t linkState) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceRemoveGpu_v2);
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_write(&gpuState, sizeof(nvmlDetachGpuState_t)) < 0 ||
        rpc_write(&linkState, sizeof(nvmlPcieLinkState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Request the OS and the NVIDIA kernel driver to rediscover a portion of the PCI subsystem looking for GPUs that
* were previously removed. The portion of the PCI tree can be narrowed by specifying a domain, bus, and device.
* If all are zeroes then the entire PCI tree will be searched.  Please note that for long-running NVML processes
* the enumeration will change based on how many GPUs are discovered and where they are inserted in bus order.
*
* In addition, all newly discovered GPUs will be initialized and their ECC scrubbed which may take several seconds
* per GPU. Also, all device handles are no longer guaranteed to be valid post discovery.
*
* Must be run as administrator.
* For Linux only.
*
* For Pascal &tm; or newer fully supported devices.
* Some Kepler devices supported.
*
* @param pciInfo                              The PCI tree to be searched.  Only the domain, bus, and device
*                                             fields are used in this call.
*
* @return
*         - \ref NVML_SUCCESS                 if counters were successfully reset
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a pciInfo is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the operating system does not support this feature
*         - \ref NVML_ERROR_OPERATING_SYSTEM  if the operating system is denying this feature
*         - \ref NVML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceDiscoverGpus(nvmlPciInfo_t* pciInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceDiscoverGpus);
    if (request_id < 0 ||
        rpc_write(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pciInfo, sizeof(nvmlPciInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Request values for a list of fields for a device. This API allows multiple fields to be queried at once.
* If any of the underlying fieldIds are populated by the same driver call, the results for those field IDs
* will be populated from a single call rather than making a driver call for each fieldId.
*
* @param device                               The device handle of the GPU to request field values for
* @param valuesCount                          Number of entries in values that should be retrieved
* @param values                               Array of \a valuesCount structures to hold field values.
*                                             Each value's fieldId must be populated prior to this call
*
* @return
*         - \ref NVML_SUCCESS                 if any values in \a values were populated. Note that you must
*                                             check the nvmlReturn field of each value for each individual
*                                             status
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a values is NULL
*/
nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetFieldValues);
    if (request_id < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Clear values for a list of fields for a device. This API allows multiple fields to be cleared at once.
*
* @param device                               The device handle of the GPU to request field values for
* @param valuesCount                          Number of entries in values that should be cleared
* @param values                               Array of \a valuesCount structures to hold field values.
*                                             Each value's fieldId must be populated prior to this call
*
* @return
*         - \ref NVML_SUCCESS                 if any values in \a values were cleared. Note that you must
*                                             check the nvmlReturn field of each value for each individual
*                                             status
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a values is NULL
*/
nvmlReturn_t nvmlDeviceClearFieldValues(nvmlDevice_t device, int valuesCount, nvmlFieldValue_t* values) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceClearFieldValues);
    if (request_id < 0 ||
        rpc_write(&valuesCount, sizeof(int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(values, valuesCount * sizeof(nvmlFieldValue_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* This method is used to get the virtualization mode corresponding to the GPU.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                    Identifier of the target device
* @param pVirtualMode              Reference to virtualization mode. One of NVML_GPU_VIRTUALIZATION_?
*
* @return
*         - \ref NVML_SUCCESS                  if \a pVirtualMode is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a pVirtualMode is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t* pVirtualMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVirtualizationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pVirtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries if SR-IOV host operation is supported on a vGPU supported device.
*
* Checks whether SR-IOV host capability is supported by the device and the
* driver, and indicates device is in SR-IOV mode if both of these conditions
* are true.
*
* @param device                                The identifier of the target device
* @param pHostVgpuMode                         Reference in which to return the current vGPU mode
*
* @return
*         - \ref NVML_SUCCESS                  if device's vGPU mode has been successfully retrieved
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device handle is 0 or \a pVgpuMode is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED      if \a device doesn't support this feature.
*         - \ref NVML_ERROR_UNKNOWN            if any unexpected error occurred
*/
nvmlReturn_t nvmlDeviceGetHostVgpuMode(nvmlDevice_t device, nvmlHostVgpuMode_t* pHostVgpuMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetHostVgpuMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pHostVgpuMode, sizeof(nvmlHostVgpuMode_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* This method is used to set the virtualization mode corresponding to the GPU.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                    Identifier of the target device
* @param virtualMode               virtualization mode. One of NVML_GPU_VIRTUALIZATION_?
*
* @return
*         - \ref NVML_SUCCESS                  if \a pVirtualMode is set
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid or \a pVirtualMode is NULL
*         - \ref NVML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_SUPPORTED      if setting of virtualization mode is not supported.
*         - \ref NVML_ERROR_NO_PERMISSION      if setting of virtualization mode is not allowed for this client.
*/
nvmlReturn_t nvmlDeviceSetVirtualizationMode(nvmlDevice_t device, nvmlGpuVirtualizationMode_t virtualMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetVirtualizationMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&virtualMode, sizeof(nvmlGpuVirtualizationMode_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the vGPU Software licensable features.
*
* Identifies whether the system supports vGPU Software Licensing. If it does, return the list of licensable feature(s)
* and their current license status.
*
* @param device                    Identifier of the target device
* @param pGridLicensableFeatures   Pointer to structure in which vGPU software licensable features are returned
*
* @return
*         - \ref NVML_SUCCESS                 if licensable features are successfully retrieved
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a pGridLicensableFeatures is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGridLicensableFeatures_v4(nvmlDevice_t device, nvmlGridLicensableFeatures_t* pGridLicensableFeatures) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGridLicensableFeatures_v4);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pGridLicensableFeatures, sizeof(nvmlGridLicensableFeatures_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current utilization and process ID
*
* For Maxwell &tm; or newer fully supported devices.
*
* Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running.
* Utilization values are returned as an array of utilization sample structures in the caller-supplied buffer pointed at
* by \a utilization. One utilization sample structure is returned per process running, that had some non-zero utilization
* during the last sample period. It includes the CPU timestamp at which  the samples were recorded. Individual utilization values
* are returned as "unsigned int" values.
*
* To read utilization values, first determine the size of buffer required to hold the samples by invoking the function with
* \a utilization set to NULL. The caller should allocate a buffer of size
* processSamplesCount * sizeof(nvmlProcessUtilizationSample_t). Invoke the function again with the allocated buffer passed
* in \a utilization, and \a processSamplesCount set to the number of entries the buffer is sized for.
*
* On successful return, the function updates \a processSamplesCount with the number of process utilization sample
* structures that were actually written. This may differ from a previously read value as instances are created or
* destroyed.
*
* lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
* to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
* to a timeStamp retrieved from a previous query to read utilization since the previous query.
*
* @note On MIG-enabled GPUs, querying process utilization is not currently supported.
*
* @param device                    The identifier of the target device
* @param utilization               Pointer to caller-supplied buffer in which guest process utilization samples are returned
* @param processSamplesCount       Pointer to caller-supplied array size, and returns number of processes running
* @param lastSeenTimeStamp         Return only samples with timestamp greater than lastSeenTimeStamp.
* @return
*         - \ref NVML_SUCCESS                 if \a utilization has been populated
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetProcessUtilization(nvmlDevice_t device, nvmlProcessUtilizationSample_t* utilization, unsigned int* processSamplesCount, unsigned long long lastSeenTimeStamp) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetProcessUtilization);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_write(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(utilization, sizeof(nvmlProcessUtilizationSample_t)) < 0 ||
        rpc_read(processSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve GSP firmware version.
*
* The caller passes in buffer via \a version and corresponding GSP firmware numbered version
* is returned with the same parameter in string format.
*
* @param device                               Device handle
* @param version                              The retrieved GSP firmware version
*
* @return
*         - \ref NVML_SUCCESS                 if GSP firmware version is sucessfully retrieved
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or GSP \a version pointer is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if GSP firmware is not enabled for GPU
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGspFirmwareVersion(nvmlDevice_t device, char* version) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareVersion);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(version, sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve GSP firmware mode.
*
* The caller passes in integer pointers. GSP firmware enablement and default mode information is returned with
* corresponding parameters. The return value in \a isEnabled and \a defaultMode should be treated as boolean.
*
* @param device                               Device handle
* @param isEnabled                            Pointer to specify if GSP firmware is enabled
* @param defaultMode                          Pointer to specify if GSP firmware is supported by default on \a device
*
* @return
*         - \ref NVML_SUCCESS                 if GSP firmware mode is sucessfully retrieved
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or any of \a isEnabled or \a defaultMode is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGspFirmwareMode(nvmlDevice_t device, unsigned int* isEnabled, unsigned int* defaultMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGspFirmwareMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_write(defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isEnabled, sizeof(unsigned int)) < 0 ||
        rpc_read(defaultMode, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the requested vGPU driver capability.
*
* Refer to the \a nvmlVgpuDriverCapability_t structure for the specific capabilities that can be queried.
* The return value in \a capResult should be treated as a boolean, with a non-zero value indicating that the capability
* is supported.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param capability      Specifies the \a nvmlVgpuDriverCapability_t to be queried
* @param capResult       A boolean for the queried capability indicating that feature is supported
*
* @return
*      - \ref NVML_SUCCESS                      successful completion
*      - \ref NVML_ERROR_UNINITIALIZED          if the library has not been successfully initialized
*      - \ref NVML_ERROR_INVALID_ARGUMENT       if \a capability is invalid, or \a capResult is NULL
*      - \ref NVML_ERROR_NOT_SUPPORTED          the API is not supported in current state or \a devices not in vGPU mode
*      - \ref NVML_ERROR_UNKNOWN                on any unexpected error
*/
nvmlReturn_t nvmlGetVgpuDriverCapabilities(nvmlVgpuDriverCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGetVgpuDriverCapabilities);
    if (request_id < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuDriverCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the requested vGPU capability for GPU.
*
* Refer to the \a nvmlDeviceVgpuCapability_t structure for the specific capabilities that can be queried.
* The return value in \a capResult reports a non-zero value indicating that the capability
* is supported, and also reports the capability's data based on the queried capability.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device     The identifier of the target device
* @param capability Specifies the \a nvmlDeviceVgpuCapability_t to be queried
* @param capResult  Specifies that the queried capability is supported, and also returns capability's data
*
* @return
*      - \ref NVML_SUCCESS                      successful completion
*      - \ref NVML_ERROR_UNINITIALIZED          if the library has not been successfully initialized
*      - \ref NVML_ERROR_INVALID_ARGUMENT       if \a device is invalid, or \a capability is invalid, or \a capResult is NULL
*      - \ref NVML_ERROR_NOT_SUPPORTED          the API is not supported in current state or \a device not in vGPU mode
*      - \ref NVML_ERROR_UNKNOWN                on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuCapabilities(nvmlDevice_t device, nvmlDeviceVgpuCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuCapabilities);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlDeviceVgpuCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the supported vGPU types on a physical GPU (device).
*
* An array of supported vGPU types for the physical GPU indicated by \a device is returned in the caller-supplied buffer
* pointed at by \a vgpuTypeIds. The element count of nvmlVgpuTypeId_t array is passed in \a vgpuCount, and \a vgpuCount
* is used to return the number of vGPU types written to the buffer.
*
* If the supplied buffer is not large enough to accommodate the vGPU type array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlVgpuTypeId_t array required in \a vgpuCount.
* To query the number of vGPU types supported for the GPU, call this function with *vgpuCount = 0.
* The code will return NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if no vGPU types are supported.
*
* @param device                   The identifier of the target device
* @param vgpuCount                Pointer to caller-supplied array size, and returns number of vGPU types
* @param vgpuTypeIds              Pointer to caller-supplied array in which to return list of vGPU types
*
* @return
*         - \ref NVML_SUCCESS                      successful completion
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE      \a vgpuTypeIds buffer is too small, array element count is returned in \a vgpuCount
*         - \ref NVML_ERROR_INVALID_ARGUMENT       if \a vgpuCount is NULL or \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED          if vGPU is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN                on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetSupportedVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedVgpus);
    if (request_id < 0 ||
        rpc_write(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the currently creatable vGPU types on a physical GPU (device).
*
* An array of creatable vGPU types for the physical GPU indicated by \a device is returned in the caller-supplied buffer
* pointed at by \a vgpuTypeIds. The element count of nvmlVgpuTypeId_t array is passed in \a vgpuCount, and \a vgpuCount
* is used to return the number of vGPU types written to the buffer.
*
* The creatable vGPU types for a device may differ over time, as there may be restrictions on what type of vGPU types
* can concurrently run on a device.  For example, if only one vGPU type is allowed at a time on a device, then the creatable
* list will be restricted to whatever vGPU type is already running on the device.
*
* If the supplied buffer is not large enough to accommodate the vGPU type array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlVgpuTypeId_t array required in \a vgpuCount.
* To query the number of vGPU types createable for the GPU, call this function with *vgpuCount = 0.
* The code will return NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if no vGPU types are creatable.
*
* @param device                   The identifier of the target device
* @param vgpuCount                Pointer to caller-supplied array size, and returns number of vGPU types
* @param vgpuTypeIds              Pointer to caller-supplied array in which to return list of vGPU types
*
* @return
*         - \ref NVML_SUCCESS                      successful completion
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE      \a vgpuTypeIds buffer is too small, array element count is returned in \a vgpuCount
*         - \ref NVML_ERROR_INVALID_ARGUMENT       if \a vgpuCount is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED          if vGPU is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN                on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetCreatableVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuTypeId_t* vgpuTypeIds) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetCreatableVgpus);
    if (request_id < 0 ||
        rpc_write(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeIds, *vgpuCount * sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the class of a vGPU type. It will not exceed 64 characters in length (including the NUL terminator).
* See \ref nvmlConstants::NVML_DEVICE_NAME_BUFFER_SIZE.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param vgpuTypeClass            Pointer to string array to return class in
* @param size                     Size of string
*
* @return
*         - \ref NVML_SUCCESS                   successful completion
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a vgpuTypeId is invalid, or \a vgpuTypeClass is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE   if \a size is too small
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetClass(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeClass, unsigned int* size) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetClass);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuTypeClass, sizeof(char)) < 0 ||
        rpc_write(size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeClass, sizeof(char)) < 0 ||
        rpc_read(size, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the vGPU type name.
*
* The name is an alphanumeric string that denotes a particular vGPU, e.g. GRID M60-2Q. It will not
* exceed 64 characters in length (including the NUL terminator).  See \ref
* nvmlConstants::NVML_DEVICE_NAME_BUFFER_SIZE.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param vgpuTypeName             Pointer to buffer to return name
* @param size                     Size of buffer
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a name is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetName(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeName, unsigned int* size) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetName);
    if (request_id < 0 ||
        rpc_write(size, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuTypeName, *size * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(size, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuTypeName, *size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the GPU Instance Profile ID for the given vGPU type ID.
* The API will return a valid GPU Instance Profile ID for the MIG capable vGPU types, else INVALID_GPU_INSTANCE_PROFILE_ID is
* returned.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param gpuInstanceProfileId     GPU Instance Profile ID
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_NOT_SUPPORTED     if \a device is not in vGPU Host virtualization mode
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a gpuInstanceProfileId is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetGpuInstanceProfileId(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* gpuInstanceProfileId) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetGpuInstanceProfileId);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstanceProfileId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the device ID of a vGPU type.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param deviceID                 Device ID and vendor ID of the device contained in single 32 bit value
* @param subsystemID              Subsystem ID and subsystem vendor ID of the device contained in single 32 bit value
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a deviceId or \a subsystemID are NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetDeviceID(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* deviceID, unsigned long long* subsystemID) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetDeviceID);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_write(subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceID, sizeof(unsigned long long)) < 0 ||
        rpc_read(subsystemID, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the vGPU framebuffer size in bytes.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param fbSize                   Pointer to framebuffer size in bytes
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a fbSize is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetFramebufferSize(nvmlVgpuTypeId_t vgpuTypeId, unsigned long long* fbSize) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFramebufferSize);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbSize, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve count of vGPU's supported display heads.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param numDisplayHeads          Pointer to number of display heads
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a numDisplayHeads is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetNumDisplayHeads(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* numDisplayHeads) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetNumDisplayHeads);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(numDisplayHeads, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve vGPU display head's maximum supported resolution.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param displayIndex             Zero-based index of display head
* @param xdim                     Pointer to maximum number of pixels in X dimension
* @param ydim                     Pointer to maximum number of pixels in Y dimension
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a xdim or \a ydim are NULL, or \a displayIndex
*                                             is out of range.
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetResolution(nvmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int* xdim, unsigned int* ydim) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetResolution);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&displayIndex, sizeof(unsigned int)) < 0 ||
        rpc_write(xdim, sizeof(unsigned int)) < 0 ||
        rpc_write(ydim, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(xdim, sizeof(unsigned int)) < 0 ||
        rpc_read(ydim, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve license requirements for a vGPU type
*
* The license type and version required to run the specified vGPU type is returned as an alphanumeric string, in the form
* "<license name>,<version>", for example "GRID-Virtual-PC,2.0". If a vGPU is runnable with* more than one type of license,
* the licenses are delimited by a semicolon, for example "GRID-Virtual-PC,2.0;GRID-Virtual-WS,2.0;GRID-Virtual-WS-Ext,2.0".
*
* The total length of the returned string will not exceed 128 characters, including the NUL terminator.
* See \ref nvmlVgpuConstants::NVML_GRID_LICENSE_BUFFER_SIZE.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param vgpuTypeLicenseString    Pointer to buffer to return license info
* @param size                     Size of \a vgpuTypeLicenseString buffer
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a vgpuTypeLicenseString is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetLicense(nvmlVgpuTypeId_t vgpuTypeId, char* vgpuTypeLicenseString, unsigned int size) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetLicense);
    if (request_id < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuTypeLicenseString, size * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeLicenseString, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the static frame rate limit value of the vGPU type
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param frameRateLimit           Reference to return the frame rate limit value
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_NOT_SUPPORTED     if frame rate limiter is turned off for the vGPU type
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a frameRateLimit is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetFrameRateLimit(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* frameRateLimit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetFrameRateLimit);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the maximum number of vGPU instances creatable on a device for given vGPU type
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                   The identifier of the target device
* @param vgpuTypeId               Handle to vGPU type
* @param vgpuInstanceCount        Pointer to get the max number of vGPU instances
*                                 that can be created on a deicve for given vgpuTypeId
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid or is not supported on target device,
*                                             or \a vgpuInstanceCount is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetMaxInstances(nvmlDevice_t device, nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCount) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstances);
    if (request_id < 0 ||
        rpc_write(vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the maximum number of vGPU instances supported per VM for given vGPU type
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuTypeId               Handle to vGPU type
* @param vgpuInstanceCountPerVm   Pointer to get the max number of vGPU instances supported per VM for given \a vgpuTypeId
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a vgpuInstanceCountPerVm is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetMaxInstancesPerVm(nvmlVgpuTypeId_t vgpuTypeId, unsigned int* vgpuInstanceCountPerVm) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetMaxInstancesPerVm);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceCountPerVm, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the active vGPU instances on a device.
*
* An array of active vGPU instances is returned in the caller-supplied buffer pointed at by \a vgpuInstances. The
* array elememt count is passed in \a vgpuCount, and \a vgpuCount is used to return the number of vGPU instances
* written to the buffer.
*
* If the supplied buffer is not large enough to accommodate the vGPU instance array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlVgpuInstance_t array required in \a vgpuCount.
* To query the number of active vGPU instances, call this function with *vgpuCount = 0.  The code will return
* NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if no vGPU Types are supported.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param device                   The identifier of the target device
* @param vgpuCount                Pointer which passes in the array size as well as get
*                                 back the number of types
* @param vgpuInstances            Pointer to array in which to return list of vGPU instances
*
* @return
*         - \ref NVML_SUCCESS                  successful completion
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a device is invalid, or \a vgpuCount is NULL
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a size is too small
*         - \ref NVML_ERROR_NOT_SUPPORTED      if vGPU is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetActiveVgpus(nvmlDevice_t device, unsigned int* vgpuCount, nvmlVgpuInstance_t* vgpuInstances) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetActiveVgpus);
    if (request_id < 0 ||
        rpc_write(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(vgpuInstances, *vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuCount, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuInstances, *vgpuCount * sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the VM ID associated with a vGPU instance.
*
* The VM ID is returned as a string, not exceeding 80 characters in length (including the NUL terminator).
* See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
*
* The format of the VM ID varies by platform, and is indicated by the type identifier returned in \a vmIdType.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param vmId                     Pointer to caller-supplied buffer to hold VM ID
* @param size                     Size of buffer in bytes
* @param vmIdType                 Pointer to hold VM ID type
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vmId or \a vmIdType is NULL, or \a vgpuInstance is 0
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetVmID(nvmlVgpuInstance_t vgpuInstance, char* vmId, unsigned int size, nvmlVgpuVmIdType_t* vmIdType) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmID);
    if (request_id < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(vmId, size * sizeof(char)) < 0 ||
        rpc_write(vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vmId, size * sizeof(char)) < 0 ||
        rpc_read(vmIdType, sizeof(nvmlVgpuVmIdType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the UUID of a vGPU instance.
*
* The UUID is a globally unique identifier associated with the vGPU, and is returned as a 5-part hexadecimal string,
* not exceeding 80 characters in length (including the NULL terminator).
* See \ref nvmlConstants::NVML_DEVICE_UUID_BUFFER_SIZE.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param uuid                     Pointer to caller-supplied buffer to hold vGPU UUID
* @param size                     Size of buffer in bytes
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a uuid is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a size is too small
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetUUID(nvmlVgpuInstance_t vgpuInstance, char* uuid, unsigned int size) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetUUID);
    if (request_id < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(uuid, size * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(uuid, size * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the NVIDIA driver version installed in the VM associated with a vGPU.
*
* The version is returned as an alphanumeric string in the caller-supplied buffer \a version. The length of the version
* string will not exceed 80 characters in length (including the NUL terminator).
* See \ref nvmlConstants::NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
*
* nvmlVgpuInstanceGetVmDriverVersion() may be called at any time for a vGPU instance. The guest VM driver version is
* returned as "Not Available" if no NVIDIA driver is installed in the VM, or the VM has not yet booted to the point where the
* NVIDIA driver is loaded and initialized.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param version                  Caller-supplied buffer to return driver version string
* @param length                   Size of \a version buffer
*
* @return
*         - \ref NVML_SUCCESS                 if \a version has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetVmDriverVersion(nvmlVgpuInstance_t vgpuInstance, char* version, unsigned int length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetVmDriverVersion);
    if (request_id < 0 ||
        rpc_write(&length, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(version, length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(version, length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the framebuffer usage in bytes.
*
* Framebuffer usage is the amont of vGPU framebuffer memory that is currently in use by the VM.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             The identifier of the target instance
* @param fbUsage                  Pointer to framebuffer usage in bytes
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a fbUsage is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetFbUsage(nvmlVgpuInstance_t vgpuInstance, unsigned long long* fbUsage) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFbUsage);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbUsage, sizeof(unsigned long long)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* @deprecated Use \ref nvmlVgpuInstanceGetLicenseInfo_v2.
*
* Retrieve the current licensing state of the vGPU instance.
*
* If the vGPU is currently licensed, \a licensed is set to 1, otherwise it is set to 0.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param licensed                 Reference to return the licensing status
*
* @return
*         - \ref NVML_SUCCESS                 if \a licensed has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a licensed is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetLicenseStatus(nvmlVgpuInstance_t vgpuInstance, unsigned int* licensed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseStatus);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(licensed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(licensed, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the vGPU type of a vGPU instance.
*
* Returns the vGPU type ID of vgpu assigned to the vGPU instance.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param vgpuTypeId               Reference to return the vgpuTypeId
*
* @return
*         - \ref NVML_SUCCESS                 if \a vgpuTypeId has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a vgpuTypeId is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetType(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuTypeId_t* vgpuTypeId) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetType);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the frame rate limit set for the vGPU instance.
*
* Returns the value of the frame rate limit set for the vGPU instance
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param frameRateLimit           Reference to return the frame rate limit
*
* @return
*         - \ref NVML_SUCCESS                 if \a frameRateLimit has been set
*         - \ref NVML_ERROR_NOT_SUPPORTED     if frame rate limiter is turned off for the vGPU type
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a frameRateLimit is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetFrameRateLimit(nvmlVgpuInstance_t vgpuInstance, unsigned int* frameRateLimit) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFrameRateLimit);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(frameRateLimit, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the current ECC mode of vGPU instance.
*
* @param vgpuInstance            The identifier of the target vGPU instance
* @param eccMode                 Reference in which to return the current ECC mode
*
* @return
*         - \ref NVML_SUCCESS                 if the vgpuInstance's ECC mode has been successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a mode is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetEccMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* eccMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEccMode);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(eccMode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param encoderCapacity          Reference to an unsigned int for the encoder capacity
*
* @return
*         - \ref NVML_SUCCESS                 if \a encoderCapacity has been retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a encoderQueryType is invalid
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int* encoderCapacity) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance             Identifier of the target vGPU instance
* @param encoderCapacity          Unsigned int for the encoder capacity value
*
* @return
*         - \ref NVML_SUCCESS                 if \a encoderCapacity has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a encoderCapacity is out of range of 0-100.
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceSetEncoderCapacity(nvmlVgpuInstance_t vgpuInstance, unsigned int encoderCapacity) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceSetEncoderCapacity);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&encoderCapacity, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the current encoder statistics of a vGPU Instance
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param sessionCount                      Reference to an unsigned int for count of active encoder sessions
* @param averageFps                        Reference to an unsigned int for trailing average FPS of all active sessions
* @param averageLatency                    Reference to an unsigned int for encode latency in microseconds
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionCount, \a averageFps and \a averageLatency is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount , or \a averageFps or \a averageLatency is NULL
*                                              or \a vgpuInstance is 0.
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetEncoderStats(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, unsigned int* averageFps, unsigned int* averageLatency) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_write(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(averageFps, sizeof(unsigned int)) < 0 ||
        rpc_read(averageLatency, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves information about all active encoder sessions on a vGPU Instance.
*
* An array of active encoder sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accommodate the active session array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlEncoderSessionInfo_t array required in \a sessionCount.
* To query the number of active encoder sessions, call this function with *sessionCount = 0. The code will return
* NVML_SUCCESS with number of active encoder sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param sessionCount                      Reference to caller supplied array size, and returns
*                                          the number of sessions.
* @param sessionInfo                       Reference to caller supplied array in which the list
*                                          of session information us returned.
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is
                                                returned in \a sessionCount
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a sessionCount is NULL, or \a vgpuInstance is 0.
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetEncoderSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlEncoderSessionInfo_t* sessionInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetEncoderSessions);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, sizeof(nvmlEncoderSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the active frame buffer capture sessions statistics of a vGPU Instance
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param fbcStats                          Reference to nvmlFBCStats_t structure containing NvFBC stats
*
* @return
*         - \ref NVML_SUCCESS                  if \a fbcStats is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a vgpuInstance is 0, or \a fbcStats is NULL
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetFBCStats(nvmlVgpuInstance_t vgpuInstance, nvmlFBCStats_t* fbcStats) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(fbcStats, sizeof(nvmlFBCStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves information about active frame buffer capture sessions on a vGPU Instance.
*
* An array of active FBC sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accomodate the active session array, the function returns
* NVML_ERROR_INSUFFICIENT_SIZE, with the element count of nvmlFBCSessionInfo_t array required in \a sessionCount.
* To query the number of active FBC sessions, call this function with *sessionCount = 0.  The code will return
* NVML_SUCCESS with number of active FBC sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @note hResolution, vResolution, averageFPS and averageLatency data for a FBC session returned in \a sessionInfo may
*       be zero if there are no new frames captured since the session started.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfo                       Reference in which to return the session information
*
* @return
*         - \ref NVML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a vgpuInstance is 0, or \a sessionCount is NULL.
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetFBCSessions(nvmlVgpuInstance_t vgpuInstance, unsigned int* sessionCount, nvmlFBCSessionInfo_t* sessionInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetFBCSessions);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_write(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(sessionCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sessionInfo, sizeof(nvmlFBCSessionInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the GPU Instance ID for the given vGPU Instance.
* The API will return a valid GPU Instance ID for MIG backed vGPU Instance, else INVALID_GPU_INSTANCE_ID is returned.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param gpuInstanceId                     GPU Instance ID
*
* @return
*         - \ref NVML_SUCCESS                  successful completion
*         - \ref NVML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a vgpuInstance is 0, or \a gpuInstanceId is NULL.
*         - \ref NVML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN            on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetGpuInstanceId(nvmlVgpuInstance_t vgpuInstance, unsigned int* gpuInstanceId) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuInstanceId);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the PCI Id of the given vGPU Instance i.e. the PCI Id of the GPU as seen inside the VM.
*
* The vGPU PCI id is returned as "00000000:00:00.0" if NVIDIA driver is not installed on the vGPU instance.
*
* @param vgpuInstance                         Identifier of the target vGPU instance
* @param vgpuPciId                            Caller-supplied buffer to return vGPU PCI Id string
* @param length                               Size of the vgpuPciId buffer
*
* @return
*         - \ref NVML_SUCCESS                 if vGPU PCI Id is sucessfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a vgpuPciId is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running on the vGPU instance
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a length is too small, \a length is set to required length
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetGpuPciId(nvmlVgpuInstance_t vgpuInstance, char* vgpuPciId, unsigned int* length) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetGpuPciId);
    if (request_id < 0 ||
        rpc_write(length, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(vgpuPciId, *length * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(length, sizeof(unsigned int)) < 0 ||
        rpc_read(vgpuPciId, *length * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the requested capability for a given vGPU type. Refer to the \a nvmlVgpuCapability_t structure
* for the specific capabilities that can be queried. The return value in \a capResult should be treated as
* a boolean, with a non-zero value indicating that the capability is supported.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuTypeId                           Handle to vGPU type
* @param capability                           Specifies the \a nvmlVgpuCapability_t to be queried
* @param capResult                            A boolean for the queried capability indicating that feature is supported
*
* @return
*         - \ref NVML_SUCCESS                 successful completion
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a capability is invalid, or \a capResult is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuTypeGetCapabilities(nvmlVgpuTypeId_t vgpuTypeId, nvmlVgpuCapability_t capability, unsigned int* capResult) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuTypeGetCapabilities);
    if (request_id < 0 ||
        rpc_write(&vgpuTypeId, sizeof(nvmlVgpuTypeId_t)) < 0 ||
        rpc_write(&capability, sizeof(nvmlVgpuCapability_t)) < 0 ||
        rpc_write(capResult, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(capResult, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns vGPU metadata structure for a running vGPU. The structure contains information about the vGPU and its associated VM
* such as the currently installed NVIDIA guest driver version, together with host driver version and an opaque data section
* containing internal state.
*
* nvmlVgpuInstanceGetMetadata() may be called at any time for a vGPU instance. Some fields in the returned structure are
* dependent on information obtained from the guest VM, which may not yet have reached a state where that information
* is available. The current state of these dependent fields is reflected in the info structure's \ref nvmlVgpuGuestInfoState_t field.
*
* The VMM may choose to read and save the vGPU's VM info as persistent metadata associated with the VM, and provide
* it to Virtual GPU Manager when creating a vGPU for subsequent instances of the VM.
*
* The caller passes in a buffer via \a vgpuMetadata, with the size of the buffer in \a bufferSize. If the vGPU Metadata structure
* is too large to fit in the supplied buffer, the function returns NVML_ERROR_INSUFFICIENT_SIZE with the size needed
* in \a bufferSize.
*
* @param vgpuInstance             vGPU instance handle
* @param vgpuMetadata             Pointer to caller-supplied buffer into which vGPU metadata is written
* @param bufferSize               Size of vgpuMetadata buffer
*
* @return
*         - \ref NVML_SUCCESS                   vGPU metadata structure was successfully returned
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE   vgpuMetadata buffer is too small, required size is returned in \a bufferSize
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a bufferSize is NULL or \a vgpuInstance is 0; if \a vgpuMetadata is NULL and the value of \a bufferSize is not 0.
*         - \ref NVML_ERROR_NOT_FOUND           if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetMetadata(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuMetadata_t* vgpuMetadata, unsigned int* bufferSize) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetMetadata);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns a vGPU metadata structure for the physical GPU indicated by \a device. The structure contains information about
* the GPU and the currently installed NVIDIA host driver version that's controlling it, together with an opaque data section
* containing internal state.
*
* The caller passes in a buffer via \a pgpuMetadata, with the size of the buffer in \a bufferSize. If the \a pgpuMetadata
* structure is too large to fit in the supplied buffer, the function returns NVML_ERROR_INSUFFICIENT_SIZE with the size needed
* in \a bufferSize.
*
* @param device                The identifier of the target device
* @param pgpuMetadata          Pointer to caller-supplied buffer into which \a pgpuMetadata is written
* @param bufferSize            Pointer to size of \a pgpuMetadata buffer
*
* @return
*         - \ref NVML_SUCCESS                   GPU metadata structure was successfully returned
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE   pgpuMetadata buffer is too small, required size is returned in \a bufferSize
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a bufferSize is NULL or \a device is invalid; if \a pgpuMetadata is NULL and the value of \a bufferSize is not 0.
*         - \ref NVML_ERROR_NOT_SUPPORTED       vGPU is not supported by the system
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuMetadata(nvmlDevice_t device, nvmlVgpuPgpuMetadata_t* pgpuMetadata, unsigned int* bufferSize) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuMetadata);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Takes a vGPU instance metadata structure read from \ref nvmlVgpuInstanceGetMetadata(), and a vGPU metadata structure for a
* physical GPU read from \ref nvmlDeviceGetVgpuMetadata(), and returns compatibility information of the vGPU instance and the
* physical GPU.
*
* The caller passes in a buffer via \a compatibilityInfo, into which a compatibility information structure is written. The
* structure defines the states in which the vGPU / VM may be booted on the physical GPU. If the vGPU / VM compatibility
* with the physical GPU is limited, a limit code indicates the factor limiting compatibility.
* (see \ref nvmlVgpuPgpuCompatibilityLimitCode_t for details).
*
* Note: vGPU compatibility does not take into account dynamic capacity conditions that may limit a system's ability to
*       boot a given vGPU or associated VM.
*
* @param vgpuMetadata          Pointer to caller-supplied vGPU metadata structure
* @param pgpuMetadata          Pointer to caller-supplied GPU metadata structure
* @param compatibilityInfo     Pointer to caller-supplied buffer to hold compatibility info
*
* @return
*         - \ref NVML_SUCCESS                   vGPU metadata structure was successfully returned
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a vgpuMetadata or \a pgpuMetadata or \a bufferSize are NULL
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlGetVgpuCompatibility(nvmlVgpuMetadata_t* vgpuMetadata, nvmlVgpuPgpuMetadata_t* pgpuMetadata, nvmlVgpuPgpuCompatibility_t* compatibilityInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGetVgpuCompatibility);
    if (request_id < 0 ||
        rpc_write(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_write(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_write(compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuMetadata, sizeof(nvmlVgpuMetadata_t)) < 0 ||
        rpc_read(pgpuMetadata, sizeof(nvmlVgpuPgpuMetadata_t)) < 0 ||
        rpc_read(compatibilityInfo, sizeof(nvmlVgpuPgpuCompatibility_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the properties of the physical GPU indicated by the device in an ascii-encoded string format.
*
* The caller passes in a buffer via \a pgpuMetadata, with the size of the buffer in \a bufferSize. If the
* string is too large to fit in the supplied buffer, the function returns NVML_ERROR_INSUFFICIENT_SIZE with the size needed
* in \a bufferSize.
*
* @param device                The identifier of the target device
* @param pgpuMetadata          Pointer to caller-supplied buffer into which \a pgpuMetadata is written
* @param bufferSize            Pointer to size of \a pgpuMetadata buffer
*
* @return
*         - \ref NVML_SUCCESS                   GPU metadata structure was successfully returned
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE   \a pgpuMetadata buffer is too small, required size is returned in \a bufferSize
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a bufferSize is NULL or \a device is invalid; if \a pgpuMetadata is NULL and the value of \a bufferSize is not 0.
*         - \ref NVML_ERROR_NOT_SUPPORTED       if vGPU is not supported by the system
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetPgpuMetadataString(nvmlDevice_t device, char* pgpuMetadata, unsigned int* bufferSize) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetPgpuMetadataString);
    if (request_id < 0 ||
        rpc_write(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pgpuMetadata, *bufferSize * sizeof(char)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(bufferSize, sizeof(unsigned int)) < 0 ||
        rpc_read(pgpuMetadata, *bufferSize * sizeof(char)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the vGPU Software scheduler logs.
* \a pSchedulerLog points to a caller-allocated structure to contain the logs. The number of elements returned will
* never exceed \a NVML_SCHEDULER_SW_MAX_LOG_ENTRIES.
*
* To get the entire logs, call the function atleast 5 times a second.
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                The identifier of the target \a device
* @param pSchedulerLog         Reference in which \a pSchedulerLog is written
*
* @return
*         - \ref NVML_SUCCESS                   vGPU scheduler logs were successfully obtained
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a pSchedulerLog is NULL or \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED       The API is not supported in current state or \a device not in vGPU host mode
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuSchedulerLog(nvmlDevice_t device, nvmlVgpuSchedulerLog_t* pSchedulerLog) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerLog);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSchedulerLog, sizeof(nvmlVgpuSchedulerLog_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the vGPU scheduler state.
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                The identifier of the target \a device
* @param pSchedulerState       Reference in which \a pSchedulerState is returned
*
* @return
*         - \ref NVML_SUCCESS                   vGPU scheduler state is successfully obtained
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a pSchedulerState is NULL or \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED       The API is not supported in current state or \a device not in vGPU host mode
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuSchedulerState(nvmlDevice_t device, nvmlVgpuSchedulerGetState_t* pSchedulerState) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pSchedulerState, sizeof(nvmlVgpuSchedulerGetState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Returns the vGPU scheduler capabilities.
* The list of supported vGPU schedulers returned in \a nvmlVgpuSchedulerCapabilities_t is from
* the NVML_VGPU_SCHEDULER_POLICY_*. This list enumerates the supported scheduler policies
* if the engine is Graphics type.
* The other values in \a nvmlVgpuSchedulerCapabilities_t are also applicable if the engine is
* Graphics type. For other engine types, it is BEST EFFORT policy.
* If ARR is supported and enabled, scheduling frequency and averaging factor are applicable
* else timeSlice is applicable.
*
* For Pascal &tm; or newer fully supported devices.
*
* @param device                The identifier of the target \a device
* @param pCapabilities         Reference in which \a pCapabilities is written
*
* @return
*         - \ref NVML_SUCCESS                   vGPU scheduler capabilities were successfully obtained
*         - \ref NVML_ERROR_INVALID_ARGUMENT    if \a pCapabilities is NULL or \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED       The API is not supported in current state or \a device not in vGPU host mode
*         - \ref NVML_ERROR_UNKNOWN             on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuSchedulerCapabilities(nvmlDevice_t device, nvmlVgpuSchedulerCapabilities_t* pCapabilities) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuSchedulerCapabilities);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pCapabilities, sizeof(nvmlVgpuSchedulerCapabilities_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Query the ranges of supported vGPU versions.
*
* This function gets the linear range of supported vGPU versions that is preset for the NVIDIA vGPU Manager and the range set by an administrator.
* If the preset range has not been overridden by \ref nvmlSetVgpuVersion, both ranges are the same.
*
* The caller passes pointers to the following \ref nvmlVgpuVersion_t structures, into which the NVIDIA vGPU Manager writes the ranges:
* 1. \a supported structure that represents the preset range of vGPU versions supported by the NVIDIA vGPU Manager.
* 2. \a current structure that represents the range of supported vGPU versions set by an administrator. By default, this range is the same as the preset range.
*
* @param supported  Pointer to the structure in which the preset range of vGPU versions supported by the NVIDIA vGPU Manager is written
* @param current    Pointer to the structure in which the range of supported vGPU versions set by an administrator is written
*
* @return
* - \ref NVML_SUCCESS                 The vGPU version range structures were successfully obtained.
* - \ref NVML_ERROR_NOT_SUPPORTED     The API is not supported.
* - \ref NVML_ERROR_INVALID_ARGUMENT  The \a supported parameter or the \a current parameter is NULL.
* - \ref NVML_ERROR_UNKNOWN           An error occurred while the data was being fetched.
*/
nvmlReturn_t nvmlGetVgpuVersion(nvmlVgpuVersion_t* supported, nvmlVgpuVersion_t* current) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGetVgpuVersion);
    if (request_id < 0 ||
        rpc_write(supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_write(current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(supported, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_read(current, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Override the preset range of vGPU versions supported by the NVIDIA vGPU Manager with a range set by an administrator.
*
* This function configures the NVIDIA vGPU Manager with a range of supported vGPU versions set by an administrator. This range must be a subset of the
* preset range that the NVIDIA vGPU Manager supports. The custom range set by an administrator takes precedence over the preset range and is advertised to
* the guest VM for negotiating the vGPU version. See \ref nvmlGetVgpuVersion for details of how to query the preset range of versions supported.
*
* This function takes a pointer to vGPU version range structure \ref nvmlVgpuVersion_t as input to override the preset vGPU version range that the NVIDIA vGPU Manager supports.
*
* After host system reboot or driver reload, the range of supported versions reverts to the range that is preset for the NVIDIA vGPU Manager.
*
* @note 1. The range set by the administrator must be a subset of the preset range that the NVIDIA vGPU Manager supports. Otherwise, an error is returned.
*       2. If the range of supported guest driver versions does not overlap the range set by the administrator, the guest driver fails to load.
*       3. If the range of supported guest driver versions overlaps the range set by the administrator, the guest driver will load with a negotiated
*          vGPU version that is the maximum value in the overlapping range.
*       4. No VMs must be running on the host when this function is called. If a VM is running on the host, the call to this function fails.
*
* @param vgpuVersion   Pointer to a caller-supplied range of supported vGPU versions.
*
* @return
* - \ref NVML_SUCCESS                 The preset range of supported vGPU versions was successfully overridden.
* - \ref NVML_ERROR_NOT_SUPPORTED     The API is not supported.
* - \ref NVML_ERROR_IN_USE            The range was not overridden because a VM is running on the host.
* - \ref NVML_ERROR_INVALID_ARGUMENT  The \a vgpuVersion parameter specifies a range that is outside the range supported by the NVIDIA vGPU Manager or if \a vgpuVersion is NULL.
*/
nvmlReturn_t nvmlSetVgpuVersion(nvmlVgpuVersion_t* vgpuVersion) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlSetVgpuVersion);
    if (request_id < 0 ||
        rpc_write(vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuVersion, sizeof(nvmlVgpuVersion_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves current utilization for vGPUs on a physical GPU (device).
*
* For Kepler &tm; or newer fully supported devices.
*
* Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for vGPU instances running
* on a device. Utilization values are returned as an array of utilization sample structures in the caller-supplied buffer
* pointed at by \a utilizationSamples. One utilization sample structure is returned per vGPU instance, and includes the
* CPU timestamp at which the samples were recorded. Individual utilization values are returned as "unsigned int" values
* in nvmlValue_t unions. The function sets the caller-supplied \a sampleValType to NVML_VALUE_TYPE_UNSIGNED_INT to
* indicate the returned value type.
*
* To read utilization values, first determine the size of buffer required to hold the samples by invoking the function with
* \a utilizationSamples set to NULL. The function will return NVML_ERROR_INSUFFICIENT_SIZE, with the current vGPU instance
* count in \a vgpuInstanceSamplesCount, or NVML_SUCCESS if the current vGPU instance count is zero. The caller should allocate
* a buffer of size vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t). Invoke the function again with
* the allocated buffer passed in \a utilizationSamples, and \a vgpuInstanceSamplesCount set to the number of entries the
* buffer is sized for.
*
* On successful return, the function updates \a vgpuInstanceSampleCount with the number of vGPU utilization sample
* structures that were actually written. This may differ from a previously read value as vGPU instances are created or
* destroyed.
*
* lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
* to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
* to a timeStamp retrieved from a previous query to read utilization since the previous query.
*
* @param device                        The identifier for the target device
* @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
* @param sampleValType                 Pointer to caller-supplied buffer to hold the type of returned sample values
* @param vgpuInstanceSamplesCount      Pointer to caller-supplied array size, and returns number of vGPU instances
* @param utilizationSamples            Pointer to caller-supplied buffer in which vGPU utilization samples are returned
* @return
*         - \ref NVML_SUCCESS                 if utilization samples are successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a vgpuInstanceSamplesCount or \a sampleValType is
*                                             NULL, or a sample count of 0 is passed with a non-NULL \a utilizationSamples
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if supplied \a vgpuInstanceSamplesCount is too small to return samples for all
*                                             vGPU instances currently executing on the device
*         - \ref NVML_ERROR_NOT_SUPPORTED     if vGPU is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, nvmlValueType_t* sampleValType, unsigned int* vgpuInstanceSamplesCount, nvmlVgpuInstanceUtilizationSample_t* utilizationSamples) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuUtilization);
    if (request_id < 0 ||
        rpc_write(vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_write(utilizationSamples, *vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuInstanceSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(sampleValType, sizeof(nvmlValueType_t)) < 0 ||
        rpc_read(utilizationSamples, *vgpuInstanceSamplesCount * sizeof(nvmlVgpuInstanceUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves current utilization for processes running on vGPUs on a physical GPU (device).
*
* For Maxwell &tm; or newer fully supported devices.
*
* Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running on
* vGPU instances active on a device. Utilization values are returned as an array of utilization sample structures in the
* caller-supplied buffer pointed at by \a utilizationSamples. One utilization sample structure is returned per process running
* on vGPU instances, that had some non-zero utilization during the last sample period. It includes the CPU timestamp at which
* the samples were recorded. Individual utilization values are returned as "unsigned int" values.
*
* To read utilization values, first determine the size of buffer required to hold the samples by invoking the function with
* \a utilizationSamples set to NULL. The function will return NVML_ERROR_INSUFFICIENT_SIZE, with the current vGPU instance
* count in \a vgpuProcessSamplesCount. The caller should allocate a buffer of size
* vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t). Invoke the function again with
* the allocated buffer passed in \a utilizationSamples, and \a vgpuProcessSamplesCount set to the number of entries the
* buffer is sized for.
*
* On successful return, the function updates \a vgpuSubProcessSampleCount with the number of vGPU sub process utilization sample
* structures that were actually written. This may differ from a previously read value depending on the number of processes that are active
* in any given sample period.
*
* lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
* to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
* to a timeStamp retrieved from a previous query to read utilization since the previous query.
*
* @param device                        The identifier for the target device
* @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
* @param vgpuProcessSamplesCount       Pointer to caller-supplied array size, and returns number of processes running on vGPU instances
* @param utilizationSamples            Pointer to caller-supplied buffer in which vGPU sub process utilization samples are returned
* @return
*         - \ref NVML_SUCCESS                 if utilization samples are successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid, \a vgpuProcessSamplesCount or a sample count of 0 is
*                                             passed with a non-NULL \a utilizationSamples
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if supplied \a vgpuProcessSamplesCount is too small to return samples for all
*                                             vGPU instances currently executing on the device
*         - \ref NVML_ERROR_NOT_SUPPORTED     if vGPU is not supported by the device
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_NOT_FOUND         if sample entries are not found
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetVgpuProcessUtilization(nvmlDevice_t device, unsigned long long lastSeenTimeStamp, unsigned int* vgpuProcessSamplesCount, nvmlVgpuProcessUtilizationSample_t* utilizationSamples) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetVgpuProcessUtilization);
    if (request_id < 0 ||
        rpc_write(vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&lastSeenTimeStamp, sizeof(unsigned long long)) < 0 ||
        rpc_write(utilizationSamples, *vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(vgpuProcessSamplesCount, sizeof(unsigned int)) < 0 ||
        rpc_read(utilizationSamples, *vgpuProcessSamplesCount * sizeof(nvmlVgpuProcessUtilizationSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries the state of per process accounting mode on vGPU.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance            The identifier of the target vGPU instance
* @param mode                    Reference in which to return the current accounting mode
*
* @return
*         - \ref NVML_SUCCESS                 if the mode has been successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a mode is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature
*         - \ref NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running on the vGPU instance
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetAccountingMode(nvmlVgpuInstance_t vgpuInstance, nvmlEnableState_t* mode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingMode);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(mode, sizeof(nvmlEnableState_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries list of processes running on vGPU that can be queried for accounting stats. The list of processes
* returned can be in running or terminated state.
*
* For Maxwell &tm; or newer fully supported devices.
*
* To just query the maximum number of processes that can be queried, call this function with *count = 0 and
* pids=NULL. The return code will be NVML_ERROR_INSUFFICIENT_SIZE, or NVML_SUCCESS if list is empty.
*
* For more details see \ref nvmlVgpuInstanceGetAccountingStats.
*
* @note In case of PID collision some processes might not be accessible before the circular buffer is full.
*
* @param vgpuInstance            The identifier of the target vGPU instance
* @param count                   Reference in which to provide the \a pids array size, and
*                                to return the number of elements ready to be queried
* @param pids                    Reference in which to return list of process ids
*
* @return
*         - \ref NVML_SUCCESS                 if pids were successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a count is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to expected value)
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*
* @see nvmlVgpuInstanceGetAccountingPids
*/
nvmlReturn_t nvmlVgpuInstanceGetAccountingPids(nvmlVgpuInstance_t vgpuInstance, unsigned int* count, unsigned int* pids) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingPids);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(pids, *count * sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Queries process's accounting stats.
*
* For Maxwell &tm; or newer fully supported devices.
*
* Accounting stats capture GPU utilization and other statistics across the lifetime of a process, and
* can be queried during life time of the process or after its termination.
* The time field in \ref nvmlAccountingStats_t is reported as 0 during the lifetime of the process and
* updated to actual running time after its termination.
* Accounting stats are kept in a circular buffer, newly created processes overwrite information about old
* processes.
*
* See \ref nvmlAccountingStats_t for description of each returned metric.
* List of processes that can be queried can be retrieved from \ref nvmlVgpuInstanceGetAccountingPids.
*
* @note Accounting Mode needs to be on. See \ref nvmlVgpuInstanceGetAccountingMode.
* @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
*         queried since they don't contribute to GPU utilization.
* @note In case of pid collision stats of only the latest process (that terminated last) will be reported
*
* @param vgpuInstance            The identifier of the target vGPU instance
* @param pid                     Process Id of the target process to query stats for
* @param stats                   Reference in which to return the process's accounting stats
*
* @return
*         - \ref NVML_SUCCESS                 if stats have been successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a stats is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*                                             or \a stats is not found
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetAccountingStats(nvmlVgpuInstance_t vgpuInstance, unsigned int pid, nvmlAccountingStats_t* stats) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetAccountingStats);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(&pid, sizeof(unsigned int)) < 0 ||
        rpc_write(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(stats, sizeof(nvmlAccountingStats_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Clears accounting information of the vGPU instance that have already terminated.
*
* For Maxwell &tm; or newer fully supported devices.
* Requires root/admin permissions.
*
* @note Accounting Mode needs to be on. See \ref nvmlVgpuInstanceGetAccountingMode.
* @note Only compute and graphics applications stats are reported and can be cleared since monitoring applications
*         stats don't contribute to GPU utilization.
*
* @param vgpuInstance            The identifier of the target vGPU instance
*
* @return
*         - \ref NVML_SUCCESS                 if accounting information has been cleared
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is invalid
*         - \ref NVML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceClearAccountingPids(nvmlVgpuInstance_t vgpuInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceClearAccountingPids);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Query the license information of the vGPU instance.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance              Identifier of the target vGPU instance
* @param licenseInfo               Pointer to vGPU license information structure
*
* @return
*         - \ref NVML_SUCCESS                 if information is successfully retrieved
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a vgpuInstance is 0, or \a licenseInfo is NULL
*         - \ref NVML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref NVML_ERROR_DRIVER_NOT_LOADED if NVIDIA driver is not running on the vGPU instance
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlVgpuInstanceGetLicenseInfo_v2(nvmlVgpuInstance_t vgpuInstance, nvmlVgpuLicenseInfo_t* licenseInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlVgpuInstanceGetLicenseInfo_v2);
    if (request_id < 0 ||
        rpc_write(&vgpuInstance, sizeof(nvmlVgpuInstance_t)) < 0 ||
        rpc_write(licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(licenseInfo, sizeof(nvmlVgpuLicenseInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieves the number of excluded GPU devices in the system.
*
* For all products.
*
* @param deviceCount                          Reference in which to return the number of excluded devices
*
* @return
*         - \ref NVML_SUCCESS                 if \a deviceCount has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a deviceCount is NULL
*/
nvmlReturn_t nvmlGetExcludedDeviceCount(unsigned int* deviceCount) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceCount);
    if (request_id < 0 ||
        rpc_write(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(deviceCount, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Acquire the device information for an excluded GPU device, based on its index.
*
* For all products.
*
* Valid indices are derived from the \a deviceCount returned by
*   \ref nvmlGetExcludedDeviceCount(). For example, if \a deviceCount is 2 the valid indices
*   are 0 and 1, corresponding to GPU 0 and GPU 1.
*
* @param index                                The index of the target GPU, >= 0 and < \a deviceCount
* @param info                                 Reference in which to return the device information
*
* @return
*         - \ref NVML_SUCCESS                  if \a device has been set
*         - \ref NVML_ERROR_INVALID_ARGUMENT   if \a index is invalid or \a info is NULL
*
* @see nvmlGetExcludedDeviceCount
*/
nvmlReturn_t nvmlGetExcludedDeviceInfoByIndex(unsigned int index, nvmlExcludedDeviceInfo_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGetExcludedDeviceInfoByIndex);
    if (request_id < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlExcludedDeviceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set MIG mode for the device.
*
* For Ampere &tm; or newer fully supported devices.
* Requires root user.
*
* This mode determines whether a GPU instance can be created.
*
* This API may unbind or reset the device to activate the requested mode. Thus, the attributes associated with the
* device, such as minor number, might change. The caller of this API is expected to query such attributes again.
*
* On certain platforms like pass-through virtualization, where reset functionality may not be exposed directly, VM
* reboot is required. \a activationStatus would return \ref NVML_ERROR_RESET_REQUIRED for such cases.
*
* \a activationStatus would return the appropriate error code upon unsuccessful activation. For example, if device
* unbind fails because the device isn't idle, \ref NVML_ERROR_IN_USE would be returned. The caller of this API
* is expected to idle the device and retry setting the \a mode.
*
* @note On Windows, only disabling MIG mode is supported. \a activationStatus would return \ref
*       NVML_ERROR_NOT_SUPPORTED as GPU reset is not supported on Windows through this API.
*
* @param device                               The identifier of the target device
* @param mode                                 The mode to be set, \ref NVML_DEVICE_MIG_DISABLE or
*                                             \ref NVML_DEVICE_MIG_ENABLE
* @param activationStatus                     The activationStatus status
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device,\a mode or \a activationStatus are invalid
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't support MIG mode
*/
nvmlReturn_t nvmlDeviceSetMigMode(nvmlDevice_t device, unsigned int mode, nvmlReturn_t* activationStatus) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetMigMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&mode, sizeof(unsigned int)) < 0 ||
        rpc_write(activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(activationStatus, sizeof(nvmlReturn_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get MIG mode for the device.
*
* For Ampere &tm; or newer fully supported devices.
*
* Changing MIG modes may require device unbind or reset. The "pending" MIG mode refers to the target mode following the
* next activation trigger.
*
* @param device                               The identifier of the target device
* @param currentMode                          Returns the current mode, \ref NVML_DEVICE_MIG_DISABLE or
*                                             \ref NVML_DEVICE_MIG_ENABLE
* @param pendingMode                          Returns the pending mode, \ref NVML_DEVICE_MIG_DISABLE or
*                                             \ref NVML_DEVICE_MIG_ENABLE
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a currentMode or \a pendingMode are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't support MIG mode
*/
nvmlReturn_t nvmlDeviceGetMigMode(nvmlDevice_t device, unsigned int* currentMode, unsigned int* pendingMode) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigMode);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(currentMode, sizeof(unsigned int)) < 0 ||
        rpc_write(pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(currentMode, sizeof(unsigned int)) < 0 ||
        rpc_read(pendingMode, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instance profile information.
*
* Information provided by this API is immutable throughout the lifetime of a MIG mode.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               The identifier of the target device
* @param profile                              One of the NVML_GPU_INSTANCE_PROFILE_*
* @param info                                 Returns detailed profile information
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profile or \a info are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profile isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfo(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceProfileInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Versioned wrapper around \ref nvmlDeviceGetGpuInstanceProfileInfo that accepts a versioned
* \ref nvmlGpuInstanceProfileInfo_v2_t or later output structure.
*
* @note The caller must set the \ref nvmlGpuInstanceProfileInfo_v2_t.version field to the
* appropriate version prior to calling this function. For example:
* \code
*     nvmlGpuInstanceProfileInfo_v2_t profileInfo =
*         { .version = nvmlGpuInstanceProfileInfo_v2 };
*     nvmlReturn_t result = nvmlDeviceGetGpuInstanceProfileInfoV(device,
*                                                                profile,
*                                                                &profileInfo);
* \endcode
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               The identifier of the target device
* @param profile                              One of the NVML_GPU_INSTANCE_PROFILE_*
* @param info                                 Returns detailed profile information
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profile, \a info, or \a info->version are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profile isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlDeviceGetGpuInstanceProfileInfoV(nvmlDevice_t device, unsigned int profile, nvmlGpuInstanceProfileInfo_v2_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceProfileInfoV);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instance placements.
*
* A placement represents the location of a GPU instance within a device. This API only returns all the possible
* placements for the given profile.
* A created GPU instance occupies memory slices described by its placement. Creation of new GPU instance will
* fail if there is overlap with the already occupied memory slices.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param device                               The identifier of the target device
* @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
* @param placements                           Returns placements allowed for the profile. Can be NULL to discover number
*                                             of allowed placements for this profile. If non-NULL must be large enough
*                                             to accommodate the placements supported by the profile.
* @param count                                Returns number of allowed placemenets for the profile.
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profileId or \a count are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlDeviceGetGpuInstancePossiblePlacements_v2(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstancePlacement_t* placements, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(placements, *count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(placements, *count * sizeof(nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instance profile capacity.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param device                               The identifier of the target device
* @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
* @param count                                Returns remaining instance count for the profile ID
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profileId or \a count are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlDeviceGetGpuInstanceRemainingCapacity(nvmlDevice_t device, unsigned int profileId, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceRemainingCapacity);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Create GPU instance.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* If the parent device is unbound, reset or the GPU instance is destroyed explicitly, the GPU instance handle would
* become invalid. The GPU instance must be recreated to acquire a valid handle.
*
* @param device                               The identifier of the target device
* @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
* @param gpuInstance                          Returns the GPU instance handle
*
* @return
*         - \ref NVML_SUCCESS                       Upon success
*         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a device, \a profile, \a profileId or \a gpuInstance are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED           If \a device doesn't have MIG mode enabled or in vGPU guest
*         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested GPU instance could not be created
*/
nvmlReturn_t nvmlDeviceCreateGpuInstance(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstance);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Create GPU instance with the specified placement.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* If the parent device is unbound, reset or the GPU instance is destroyed explicitly, the GPU instance handle would
* become invalid. The GPU instance must be recreated to acquire a valid handle.
*
* @param device                               The identifier of the target device
* @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
* @param placement                            The requested placement. See \ref nvmlDeviceGetGpuInstancePossiblePlacements_v2
* @param gpuInstance                          Returns the GPU instance handle
*
* @return
*         - \ref NVML_SUCCESS                       Upon success
*         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a device, \a profile, \a profileId, \a placement or \a gpuInstance
*                                                   are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED           If \a device doesn't have MIG mode enabled or in vGPU guest
*         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested GPU instance could not be created
*/
nvmlReturn_t nvmlDeviceCreateGpuInstanceWithPlacement(nvmlDevice_t device, unsigned int profileId, const nvmlGpuInstancePlacement_t* placement, nvmlGpuInstance_t* gpuInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceCreateGpuInstanceWithPlacement);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&placement, sizeof(const nvmlGpuInstancePlacement_t)) < 0 ||
        rpc_write(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Destroy GPU instance.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param gpuInstance                          The GPU instance handle
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or in vGPU guest
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_IN_USE            If the GPU instance is in use. This error would be returned if processes
*                                             (e.g. CUDA application) or compute instances are active on the
*                                             GPU instance.
*/
nvmlReturn_t nvmlGpuInstanceDestroy(nvmlGpuInstance_t gpuInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instances for given profile ID.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param device                               The identifier of the target device
* @param profileId                            The GPU instance profile ID. See \ref nvmlDeviceGetGpuInstanceProfileInfo
* @param gpuInstances                         Returns pre-exiting GPU instances, the buffer must be large enough to
*                                             accommodate the instances supported by the profile.
*                                             See \ref nvmlDeviceGetGpuInstanceProfileInfo
* @param count                                The count of returned GPU instances
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a profileId, \a gpuInstances or \a count are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlDeviceGetGpuInstances(nvmlDevice_t device, unsigned int profileId, nvmlGpuInstance_t* gpuInstances, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstances);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(gpuInstances, *count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(gpuInstances, *count * sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instances for given instance ID.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param device                               The identifier of the target device
* @param id                                   The GPU instance ID
* @param gpuInstance                          Returns GPU instance
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a id or \a gpuInstance are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_NOT_FOUND         If the GPU instance is not found.
*/
nvmlReturn_t nvmlDeviceGetGpuInstanceById(nvmlDevice_t device, unsigned int id, nvmlGpuInstance_t* gpuInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceById);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_write(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instance information.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param gpuInstance                          The GPU instance handle
* @param info                                 Return GPU instance information
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance or \a info are invalid
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlGpuInstanceGetInfo(nvmlGpuInstance_t gpuInstance, nvmlGpuInstanceInfo_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetInfo);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlGpuInstanceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instance profile information.
*
* Information provided by this API is immutable throughout the lifetime of a MIG mode.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profile                              One of the NVML_COMPUTE_INSTANCE_PROFILE_*
* @param engProfile                           One of the NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_*
* @param info                                 Returns detailed profile information
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profile, \a engProfile or \a info are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profile isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfo(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceProfileInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Versioned wrapper around \ref nvmlGpuInstanceGetComputeInstanceProfileInfo that accepts a versioned
* \ref nvmlComputeInstanceProfileInfo_v2_t or later output structure.
*
* @note The caller must set the \ref nvmlGpuInstanceProfileInfo_v2_t.version field to the
* appropriate version prior to calling this function. For example:
* \code
*     nvmlComputeInstanceProfileInfo_v2_t profileInfo =
*         { .version = nvmlComputeInstanceProfileInfo_v2 };
*     nvmlReturn_t result = nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance,
*                                                                         profile,
*                                                                         engProfile,
*                                                                         &profileInfo);
* \endcode
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profile                              One of the NVML_COMPUTE_INSTANCE_PROFILE_*
* @param engProfile                           One of the NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_*
* @param info                                 Returns detailed profile information
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profile, \a engProfile, \a info, or \a info->version are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profile isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceProfileInfoV(nvmlGpuInstance_t gpuInstance, unsigned int profile, unsigned int engProfile, nvmlComputeInstanceProfileInfo_v2_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profile, sizeof(unsigned int)) < 0 ||
        rpc_write(&engProfile, sizeof(unsigned int)) < 0 ||
        rpc_write(info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceProfileInfo_v2_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instance profile capacity.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profileId                            The compute instance profile ID.
*                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
* @param count                                Returns remaining instance count for the profile ID
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profileId or \a availableCount are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceRemainingCapacity(nvmlGpuInstance_t gpuInstance, unsigned int profileId, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instance placements.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* A placement represents the location of a compute instance within a GPU instance. This API only returns all the possible
* placements for the given profile.
* A created compute instance occupies compute slices described by its placement. Creation of new compute instance will
* fail if there is overlap with the already occupied compute slices.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profileId                            The compute instance profile ID. See \ref  nvmlGpuInstanceGetComputeInstanceProfileInfo
* @param placements                           Returns placements allowed for the profile. Can be NULL to discover number
*                                             of allowed placements for this profile. If non-NULL must be large enough
*                                             to accommodate the placements supported by the profile.
* @param count                                Returns number of allowed placemenets for the profile.
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profileId or \a count are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlGpuInstanceGetComputeInstancePossiblePlacements(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstancePlacement_t* placements, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(placements, sizeof(nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Create compute instance.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* If the parent device is unbound, reset or the parent GPU instance is destroyed or the compute instance is destroyed
* explicitly, the compute instance handle would become invalid. The compute instance must be recreated to acquire
* a valid handle.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profileId                            The compute instance profile ID.
*                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
* @param computeInstance                      Returns the compute instance handle
*
* @return
*         - \ref NVML_SUCCESS                       Upon success
*         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a gpuInstance, \a profile, \a profileId or \a computeInstance
*                                                   are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED           If \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested compute instance could not be created
*/
nvmlReturn_t nvmlGpuInstanceCreateComputeInstance(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstance);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Create compute instance with the specified placement.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* If the parent device is unbound, reset or the parent GPU instance is destroyed or the compute instance is destroyed
* explicitly, the compute instance handle would become invalid. The compute instance must be recreated to acquire
* a valid handle.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profileId                            The compute instance profile ID.
*                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
* @param placement                            The requested placement. See \ref nvmlGpuInstanceGetComputeInstancePossiblePlacements
* @param computeInstance                      Returns the compute instance handle
*
* @return
*         - \ref NVML_SUCCESS                       Upon success
*         - \ref NVML_ERROR_UNINITIALIZED           If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT        If \a gpuInstance, \a profile, \a profileId or \a computeInstance
*                                                   are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED           If \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_INSUFFICIENT_RESOURCES  If the requested compute instance could not be created
*/
nvmlReturn_t nvmlGpuInstanceCreateComputeInstanceWithPlacement(nvmlGpuInstance_t gpuInstance, unsigned int profileId, const nvmlComputeInstancePlacement_t* placement, nvmlComputeInstance_t* computeInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceCreateComputeInstanceWithPlacement);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(&placement, sizeof(const nvmlComputeInstancePlacement_t)) < 0 ||
        rpc_write(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Destroy compute instance.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param computeInstance                      The compute instance handle
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a computeInstance is invalid
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_IN_USE            If the compute instance is in use. This error would be returned if
*                                             processes (e.g. CUDA application) are active on the compute instance.
*/
nvmlReturn_t nvmlComputeInstanceDestroy(nvmlComputeInstance_t computeInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlComputeInstanceDestroy);
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instances for given profile ID.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param profileId                            The compute instance profile ID.
*                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
* @param computeInstances                     Returns pre-exiting compute instances, the buffer must be large enough to
*                                             accommodate the instances supported by the profile.
*                                             See \ref nvmlGpuInstanceGetComputeInstanceProfileInfo
* @param count                                The count of returned compute instances
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a gpuInstance, \a profileId, \a computeInstances or \a count
*                                             are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a profileId isn't supported
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlGpuInstanceGetComputeInstances(nvmlGpuInstance_t gpuInstance, unsigned int profileId, nvmlComputeInstance_t* computeInstances, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstances);
    if (request_id < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&profileId, sizeof(unsigned int)) < 0 ||
        rpc_write(computeInstances, *count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_read(computeInstances, *count * sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instance for given instance ID.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
* Requires privileged user.
*
* @param gpuInstance                          The identifier of the target GPU instance
* @param id                                   The compute instance ID
* @param computeInstance                      Returns compute instance
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a device, \a ID or \a computeInstance are invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*         - \ref NVML_ERROR_NOT_FOUND         If the compute instance is not found.
*/
nvmlReturn_t nvmlGpuInstanceGetComputeInstanceById(nvmlGpuInstance_t gpuInstance, unsigned int id, nvmlComputeInstance_t* computeInstance) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpuInstanceGetComputeInstanceById);
    if (request_id < 0 ||
        rpc_write(&gpuInstance, sizeof(nvmlGpuInstance_t)) < 0 ||
        rpc_write(&id, sizeof(unsigned int)) < 0 ||
        rpc_write(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instance information.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param computeInstance                      The compute instance handle
* @param info                                 Return compute instance information
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_UNINITIALIZED     If library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  If \a computeInstance or \a info are invalid
*         - \ref NVML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
*/
nvmlReturn_t nvmlComputeInstanceGetInfo_v2(nvmlComputeInstance_t computeInstance, nvmlComputeInstanceInfo_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlComputeInstanceGetInfo_v2);
    if (request_id < 0 ||
        rpc_write(&computeInstance, sizeof(nvmlComputeInstance_t)) < 0 ||
        rpc_write(info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlComputeInstanceInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Test if the given handle refers to a MIG device.
*
* A MIG device handle is an NVML abstraction which maps to a MIG compute instance.
* These overloaded references can be used (with some restrictions) interchangeably
* with a GPU device handle to execute queries at a per-compute instance granularity.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               NVML handle to test
* @param isMigDevice                          True when handle refers to a MIG device
*
* @return
*         - \ref NVML_SUCCESS                 if \a device status was successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device handle or \a isMigDevice reference is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceIsMigDeviceHandle(nvmlDevice_t device, unsigned int* isMigDevice) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceIsMigDeviceHandle);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(isMigDevice, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get GPU instance ID for the given MIG device handle.
*
* GPU instance IDs are unique per device and remain valid until the GPU instance is destroyed.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               Target MIG device handle
* @param id                                   GPU instance ID
*
* @return
*         - \ref NVML_SUCCESS                 if instance ID was successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a id reference is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGpuInstanceId(nvmlDevice_t device, unsigned int* id) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuInstanceId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(id, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get compute instance ID for the given MIG device handle.
*
* Compute instance IDs are unique per GPU instance and remain valid until the compute instance
* is destroyed.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               Target MIG device handle
* @param id                                   Compute instance ID
*
* @return
*         - \ref NVML_SUCCESS                 if instance ID was successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a id reference is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetComputeInstanceId(nvmlDevice_t device, unsigned int* id) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetComputeInstanceId);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(id, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(id, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get the maximum number of MIG devices that can exist under a given parent NVML device.
*
* Returns zero if MIG is not supported or enabled.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               Target device handle
* @param count                                Count of MIG devices
*
* @return
*         - \ref NVML_SUCCESS                 if \a count was successfully retrieved
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a count reference is invalid
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMaxMigDeviceCount(nvmlDevice_t device, unsigned int* count) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMaxMigDeviceCount);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(count, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(count, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get MIG device handle for the given index under its parent NVML device.
*
* If the compute instance is destroyed either explicitly or by destroying,
* resetting or unbinding the parent GPU instance or the GPU device itself
* the MIG device handle would remain invalid and must be requested again
* using this API. Handles may be reused and their properties can change in
* the process.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param device                               Reference to the parent GPU device handle
* @param index                                Index of the MIG device
* @param migDevice                            Reference to the MIG device handle
*
* @return
*         - \ref NVML_SUCCESS                 if \a migDevice handle was successfully created
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a index or \a migDevice reference is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_NOT_FOUND         if no valid MIG device was found at \a index
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice_t device, unsigned int index, nvmlDevice_t* migDevice) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMigDeviceHandleByIndex);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&index, sizeof(unsigned int)) < 0 ||
        rpc_write(migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get parent device handle from a MIG device handle.
*
* For Ampere &tm; or newer fully supported devices.
* Supported on Linux only.
*
* @param migDevice                            MIG device handle
* @param device                               Device handle
*
* @return
*         - \ref NVML_SUCCESS                 if \a device handle was successfully created
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a migDevice or \a device is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetDeviceHandleFromMigDeviceHandle(nvmlDevice_t migDevice, nvmlDevice_t* device) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle);
    if (request_id < 0 ||
        rpc_write(&migDevice, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get the type of the GPU Bus (PCIe, PCI, ...)
*
* @param device                               The identifier of the target device
* @param type                                 The PCI Bus type
*
* return
*         - \ref NVML_SUCCESS                 if the bus \a type is successfully retreived
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \device is invalid or \type is NULL
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetBusType(nvmlDevice_t device, nvmlBusType_t* type) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetBusType);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(type, sizeof(nvmlBusType_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve performance monitor samples from the associated subdevice.
*
* @param device
* @param pDynamicPstatesInfo
*
* @return
*         - \ref NVML_SUCCESS                 if \a pDynamicPstatesInfo has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a pDynamicPstatesInfo is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetDynamicPstatesInfo(nvmlDevice_t device, nvmlGpuDynamicPstatesInfo_t* pDynamicPstatesInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetDynamicPstatesInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pDynamicPstatesInfo, sizeof(nvmlGpuDynamicPstatesInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Sets the speed of a specified fan.
*
* WARNING: This function changes the fan control policy to manual. It means that YOU have to monitor
*          the temperature and adjust the fan speed accordingly.
*          If you set the fan speed too low you can burn your GPU!
*          Use nvmlDeviceSetDefaultFanSpeed_v2 to restore default control policy.
*
* For all cuda-capable discrete products with fans that are Maxwell or Newer.
*
* device                                The identifier of the target device
* fan                                   The index of the fan, starting at zero
* speed                                 The target speed of the fan [0-100] in % of max speed
*
* return
*        NVML_SUCCESS                   if the fan speed has been set
*        NVML_ERROR_UNINITIALIZED       if the library has not been successfully initialized
*        NVML_ERROR_INVALID_ARGUMENT    if the device is not valid, or the speed is outside acceptable ranges,
*                                              or if the fan index doesn't reference an actual fan.
*        NVML_ERROR_NOT_SUPPORTED       if the device is older than Maxwell.
*        NVML_ERROR_UNKNOWN             if there was an unexpected error.
*/
nvmlReturn_t nvmlDeviceSetFanSpeed_v2(nvmlDevice_t device, unsigned int fan, unsigned int speed) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetFanSpeed_v2);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&fan, sizeof(unsigned int)) < 0 ||
        rpc_write(&speed, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the GPCCLK VF offset value
* @param[in]   device                         The identifier of the target device
* @param[out]  offset                         The retrieved GPCCLK VF offset value
*
* @return
*         - \ref NVML_SUCCESS                 if \a offset has been successfully queried
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a offset is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGpcClkVfOffset(nvmlDevice_t device, int* offset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(offset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the GPCCLK VF offset value
* @param[in]   device                         The identifier of the target device
* @param[in]   offset                         The GPCCLK VF offset value to set
*
* @return
*         - \ref NVML_SUCCESS                 if \a offset has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a offset is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetGpcClkVfOffset(nvmlDevice_t device, int offset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetGpcClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the MemClk (Memory Clock) VF offset value.
* @param[in]   device                         The identifier of the target device
* @param[out]  offset                         The retrieved MemClk VF offset value
*
* @return
*         - \ref NVML_SUCCESS                 if \a offset has been successfully queried
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a offset is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMemClkVfOffset(nvmlDevice_t device, int* offset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(offset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set the MemClk (Memory Clock) VF offset value. It requires elevated privileges.
* @param[in]   device                         The identifier of the target device
* @param[in]   offset                         The MemClk VF offset value to set
*
* @return
*         - \ref NVML_SUCCESS                 if \a offset has been set
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a offset is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceSetMemClkVfOffset(nvmlDevice_t device, int offset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetMemClkVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&offset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve min and max clocks of some clock domain for a given PState
*
* @param device                               The identifier of the target device
* @param type                                 Clock domain
* @param pstate                               PState to query
* @param minClockMHz                          Reference in which to return min clock frequency
* @param maxClockMHz                          Reference in which to return max clock frequency
*
* @return
*         - \ref NVML_SUCCESS                 if everything worked
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device, \a type or \a pstate are invalid or both
*                                                  \a minClockMHz and \a maxClockMHz are NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*/
nvmlReturn_t nvmlDeviceGetMinMaxClockOfPState(nvmlDevice_t device, nvmlClockType_t type, nvmlPstates_t pstate, unsigned int* minClockMHz, unsigned int* maxClockMHz) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMinMaxClockOfPState);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&type, sizeof(nvmlClockType_t)) < 0 ||
        rpc_write(&pstate, sizeof(nvmlPstates_t)) < 0 ||
        rpc_write(minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_write(maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_read(maxClockMHz, sizeof(unsigned int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get all supported Performance States (P-States) for the device.
*
* The returned array would contain a contiguous list of valid P-States supported by
* the device. If the number of supported P-States is fewer than the size of the array
* supplied missing elements would contain \a NVML_PSTATE_UNKNOWN.
*
* The number of elements in the returned list will never exceed \a NVML_MAX_GPU_PERF_PSTATES.
*
* @param device                               The identifier of the target device
* @param pstates                              Container to return the list of performance states
*                                             supported by device
* @param size                                 Size of the supplied \a pstates array in bytes
*
* @return
*         - \ref NVML_SUCCESS                 if \a pstates array has been retrieved
*         - \ref NVML_ERROR_INSUFFICIENT_SIZE if the the container supplied was not large enough to
*                                             hold the resulting list
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device or \a pstates is invalid
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support performance state readings
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetSupportedPerformanceStates(nvmlDevice_t device, nvmlPstates_t* pstates, unsigned int size) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetSupportedPerformanceStates);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(pstates, sizeof(nvmlPstates_t)) < 0 ||
        rpc_write(&size, sizeof(unsigned int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(pstates, sizeof(nvmlPstates_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the GPCCLK min max VF offset value.
* @param[in]   device                         The identifier of the target device
* @param[out]  minOffset                      The retrieved GPCCLK VF min offset value
* @param[out]  maxOffset                      The retrieved GPCCLK VF max offset value
*
* @return
*         - \ref NVML_SUCCESS                 if \a offset has been successfully queried
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a offset is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetGpcClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpcClkMinMaxVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minOffset, sizeof(int)) < 0 ||
        rpc_write(maxOffset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minOffset, sizeof(int)) < 0 ||
        rpc_read(maxOffset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Retrieve the MemClk (Memory Clock) min max VF offset value.
* @param[in]   device                         The identifier of the target device
* @param[out]  minOffset                      The retrieved MemClk VF min offset value
* @param[out]  maxOffset                      The retrieved MemClk VF max offset value
*
* @return
*         - \ref NVML_SUCCESS                 if \a offset has been successfully queried
*         - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*         - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a offset is NULL
*         - \ref NVML_ERROR_NOT_SUPPORTED     if the device does not support this feature
*         - \ref NVML_ERROR_UNKNOWN           on any unexpected error
*/
nvmlReturn_t nvmlDeviceGetMemClkMinMaxVfOffset(nvmlDevice_t device, int* minOffset, int* maxOffset) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetMemClkMinMaxVfOffset);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(minOffset, sizeof(int)) < 0 ||
        rpc_write(maxOffset, sizeof(int)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(minOffset, sizeof(int)) < 0 ||
        rpc_read(maxOffset, sizeof(int)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Get fabric information associated with the device.
*
* %HOPPER_OR_NEWER%
*
* On Hopper + NVSwitch systems, GPU is registered with the NVIDIA Fabric Manager
* Upon successful registration, the GPU is added to the NVLink fabric to enable
* peer-to-peer communication.
* This API reports the current state of the GPU in the NVLink fabric
* along with other useful information.
*
* @param device                               The identifier of the target device
* @param gpuFabricInfo                        Information about GPU fabric state
*
* @return
*         - \ref NVML_SUCCESS                 Upon success
*         - \ref NVML_ERROR_NOT_SUPPORTED     If \a device doesn't support gpu fabric
*/
nvmlReturn_t nvmlDeviceGetGpuFabricInfo(nvmlDevice_t device, nvmlGpuFabricInfo_t* gpuFabricInfo) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceGetGpuFabricInfo);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpuFabricInfo, sizeof(nvmlGpuFabricInfo_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Calculate GPM metrics from two samples.
*
* For Hopper &tm; or newer fully supported devices.
*
* @param metricsGet             IN/OUT: populated \a nvmlGpmMetricsGet_t struct
*
* @return
*         - \ref NVML_SUCCESS on success
*         - Nonzero NVML_ERROR_? enum on error
*/
nvmlReturn_t nvmlGpmMetricsGet(nvmlGpmMetricsGet_t* metricsGet) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpmMetricsGet);
    if (request_id < 0 ||
        rpc_write(metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(metricsGet, sizeof(nvmlGpmMetricsGet_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Free an allocated sample buffer that was allocated with \ref nvmlGpmSampleAlloc()
*
* For Hopper &tm; or newer fully supported devices.
*
* @param gpmSample              Sample to free
*
* @return
*         - \ref NVML_SUCCESS                on success
*         - \ref NVML_ERROR_INVALID_ARGUMENT if an invalid pointer is provided
*/
nvmlReturn_t nvmlGpmSampleFree(nvmlGpmSample_t gpmSample) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpmSampleFree);
    if (request_id < 0 ||
        rpc_write(&gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Allocate a sample buffer to be used with NVML GPM . You will need to allocate
* at least two of these buffers to use with the NVML GPM feature
*
* For Hopper &tm; or newer fully supported devices.
*
* @param gpmSample             Where  the allocated sample will be stored
*
* @return
*         - \ref NVML_SUCCESS                on success
*         - \ref NVML_ERROR_INVALID_ARGUMENT if an invalid pointer is provided
*         - \ref NVML_ERROR_MEMORY           if system memory is insufficient
*/
nvmlReturn_t nvmlGpmSampleAlloc(nvmlGpmSample_t* gpmSample) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpmSampleAlloc);
    if (request_id < 0 ||
        rpc_write(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Read a sample of GPM metrics into the provided \a gpmSample buffer. After
* two samples are gathered, you can call nvmlGpmMetricGet on those samples to
* retrive metrics
*
* For Hopper &tm; or newer fully supported devices.
*
* @param device                Device to get samples for
* @param gpmSample             Buffer to read samples into
*
* @return
*         - \ref NVML_SUCCESS on success
*         - Nonzero NVML_ERROR_? enum on error
*/
nvmlReturn_t nvmlGpmSampleGet(nvmlDevice_t device, nvmlGpmSample_t gpmSample) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpmSampleGet);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Read a sample of GPM metrics into the provided \a gpmSample buffer for a MIG GPU Instance.
*
* After two samples are gathered, you can call nvmlGpmMetricGet on those
* samples to retrive metrics
*
* For Hopper &tm; or newer fully supported devices.
*
* @param device                Device to get samples for
* @param gpuInstanceId         MIG GPU Instance ID
* @param gpmSample             Buffer to read samples into
*
* @return
*         - \ref NVML_SUCCESS on success
*         - Nonzero NVML_ERROR_? enum on error
*/
nvmlReturn_t nvmlGpmMigSampleGet(nvmlDevice_t device, unsigned int gpuInstanceId, nvmlGpmSample_t gpmSample) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpmMigSampleGet);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(&gpuInstanceId, sizeof(unsigned int)) < 0 ||
        rpc_write(&gpmSample, sizeof(nvmlGpmSample_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Indicate whether the supplied device supports GPM
*
* @param device                NVML device to query for
* @param gpmSupport            Structure to indicate GPM support \a nvmlGpmSupport_t. Indicates
*                              GPM support per system for the supplied device
*
* @return
*         - NVML_SUCCESS on success
*         - Nonzero NVML_ERROR_? enum if there is an error in processing the query
*/
nvmlReturn_t nvmlGpmQueryDeviceSupport(nvmlDevice_t device, nvmlGpmSupport_t* gpmSupport) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlGpmQueryDeviceSupport);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(gpmSupport, sizeof(nvmlGpmSupport_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}

/**
* Set NvLink Low Power Threshold for device.
*
* %HOPPER_OR_NEWER%
*
* @param device                               The identifier of the target device
* @param info                                 Reference to \a nvmlNvLinkPowerThres_t struct
*                                             input parameters
*
* @return
*        - \ref NVML_SUCCESS                 if the \a Threshold is successfully set
*        - \ref NVML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
*        - \ref NVML_ERROR_INVALID_ARGUMENT  if \a device is invalid or \a Threshold is not within range
*        - \ref NVML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
*
**/
nvmlReturn_t nvmlDeviceSetNvLinkDeviceLowPowerThreshold(nvmlDevice_t device, nvmlNvLinkPowerThres_t* info) {
    nvmlReturn_t return_value;
    
    int request_id = rpc_start_request(RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold);
    if (request_id < 0 ||
        rpc_write(&device, sizeof(nvmlDevice_t)) < 0 ||
        rpc_write(info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_wait_for_response(request_id) < 0 ||
        rpc_read(info, sizeof(nvmlNvLinkPowerThres_t)) < 0 ||
        rpc_end_request(&return_value, request_id) < 0)
        return NVML_ERROR_GPU_IS_LOST;
    return return_value;
}


std::unordered_map<std::string, void *> functionMap = {
    {"nvmlInit_v2", (void *)nvmlInit_v2},
    {"nvmlInitWithFlags", (void *)nvmlInitWithFlags},
    {"nvmlShutdown", (void *)nvmlShutdown},
    {"nvmlSystemGetDriverVersion", (void *)nvmlSystemGetDriverVersion},
    {"nvmlSystemGetNVMLVersion", (void *)nvmlSystemGetNVMLVersion},
    {"nvmlSystemGetCudaDriverVersion", (void *)nvmlSystemGetCudaDriverVersion},
    {"nvmlSystemGetCudaDriverVersion_v2", (void *)nvmlSystemGetCudaDriverVersion_v2},
    {"nvmlSystemGetProcessName", (void *)nvmlSystemGetProcessName},
    {"nvmlUnitGetCount", (void *)nvmlUnitGetCount},
    {"nvmlUnitGetHandleByIndex", (void *)nvmlUnitGetHandleByIndex},
    {"nvmlUnitGetUnitInfo", (void *)nvmlUnitGetUnitInfo},
    {"nvmlUnitGetLedState", (void *)nvmlUnitGetLedState},
    {"nvmlUnitGetPsuInfo", (void *)nvmlUnitGetPsuInfo},
    {"nvmlUnitGetTemperature", (void *)nvmlUnitGetTemperature},
    {"nvmlUnitGetFanSpeedInfo", (void *)nvmlUnitGetFanSpeedInfo},
    {"nvmlUnitGetDevices", (void *)nvmlUnitGetDevices},
    {"nvmlSystemGetHicVersion", (void *)nvmlSystemGetHicVersion},
    {"nvmlDeviceGetCount_v2", (void *)nvmlDeviceGetCount_v2},
    {"nvmlDeviceGetAttributes_v2", (void *)nvmlDeviceGetAttributes_v2},
    {"nvmlDeviceGetHandleByIndex_v2", (void *)nvmlDeviceGetHandleByIndex_v2},
    {"nvmlDeviceGetHandleBySerial", (void *)nvmlDeviceGetHandleBySerial},
    {"nvmlDeviceGetHandleByUUID", (void *)nvmlDeviceGetHandleByUUID},
    {"nvmlDeviceGetHandleByPciBusId_v2", (void *)nvmlDeviceGetHandleByPciBusId_v2},
    {"nvmlDeviceGetName", (void *)nvmlDeviceGetName},
    {"nvmlDeviceGetBrand", (void *)nvmlDeviceGetBrand},
    {"nvmlDeviceGetIndex", (void *)nvmlDeviceGetIndex},
    {"nvmlDeviceGetSerial", (void *)nvmlDeviceGetSerial},
    {"nvmlDeviceGetMemoryAffinity", (void *)nvmlDeviceGetMemoryAffinity},
    {"nvmlDeviceGetCpuAffinityWithinScope", (void *)nvmlDeviceGetCpuAffinityWithinScope},
    {"nvmlDeviceGetCpuAffinity", (void *)nvmlDeviceGetCpuAffinity},
    {"nvmlDeviceSetCpuAffinity", (void *)nvmlDeviceSetCpuAffinity},
    {"nvmlDeviceClearCpuAffinity", (void *)nvmlDeviceClearCpuAffinity},
    {"nvmlDeviceGetTopologyCommonAncestor", (void *)nvmlDeviceGetTopologyCommonAncestor},
    {"nvmlDeviceGetTopologyNearestGpus", (void *)nvmlDeviceGetTopologyNearestGpus},
    {"nvmlSystemGetTopologyGpuSet", (void *)nvmlSystemGetTopologyGpuSet},
    {"nvmlDeviceGetP2PStatus", (void *)nvmlDeviceGetP2PStatus},
    {"nvmlDeviceGetUUID", (void *)nvmlDeviceGetUUID},
    {"nvmlVgpuInstanceGetMdevUUID", (void *)nvmlVgpuInstanceGetMdevUUID},
    {"nvmlDeviceGetMinorNumber", (void *)nvmlDeviceGetMinorNumber},
    {"nvmlDeviceGetBoardPartNumber", (void *)nvmlDeviceGetBoardPartNumber},
    {"nvmlDeviceGetInforomVersion", (void *)nvmlDeviceGetInforomVersion},
    {"nvmlDeviceGetInforomImageVersion", (void *)nvmlDeviceGetInforomImageVersion},
    {"nvmlDeviceGetInforomConfigurationChecksum", (void *)nvmlDeviceGetInforomConfigurationChecksum},
    {"nvmlDeviceValidateInforom", (void *)nvmlDeviceValidateInforom},
    {"nvmlDeviceGetDisplayMode", (void *)nvmlDeviceGetDisplayMode},
    {"nvmlDeviceGetDisplayActive", (void *)nvmlDeviceGetDisplayActive},
    {"nvmlDeviceGetPersistenceMode", (void *)nvmlDeviceGetPersistenceMode},
    {"nvmlDeviceGetPciInfo_v3", (void *)nvmlDeviceGetPciInfo_v3},
    {"nvmlDeviceGetMaxPcieLinkGeneration", (void *)nvmlDeviceGetMaxPcieLinkGeneration},
    {"nvmlDeviceGetGpuMaxPcieLinkGeneration", (void *)nvmlDeviceGetGpuMaxPcieLinkGeneration},
    {"nvmlDeviceGetMaxPcieLinkWidth", (void *)nvmlDeviceGetMaxPcieLinkWidth},
    {"nvmlDeviceGetCurrPcieLinkGeneration", (void *)nvmlDeviceGetCurrPcieLinkGeneration},
    {"nvmlDeviceGetCurrPcieLinkWidth", (void *)nvmlDeviceGetCurrPcieLinkWidth},
    {"nvmlDeviceGetPcieThroughput", (void *)nvmlDeviceGetPcieThroughput},
    {"nvmlDeviceGetPcieReplayCounter", (void *)nvmlDeviceGetPcieReplayCounter},
    {"nvmlDeviceGetClockInfo", (void *)nvmlDeviceGetClockInfo},
    {"nvmlDeviceGetMaxClockInfo", (void *)nvmlDeviceGetMaxClockInfo},
    {"nvmlDeviceGetApplicationsClock", (void *)nvmlDeviceGetApplicationsClock},
    {"nvmlDeviceGetDefaultApplicationsClock", (void *)nvmlDeviceGetDefaultApplicationsClock},
    {"nvmlDeviceResetApplicationsClocks", (void *)nvmlDeviceResetApplicationsClocks},
    {"nvmlDeviceGetClock", (void *)nvmlDeviceGetClock},
    {"nvmlDeviceGetMaxCustomerBoostClock", (void *)nvmlDeviceGetMaxCustomerBoostClock},
    {"nvmlDeviceGetSupportedMemoryClocks", (void *)nvmlDeviceGetSupportedMemoryClocks},
    {"nvmlDeviceGetSupportedGraphicsClocks", (void *)nvmlDeviceGetSupportedGraphicsClocks},
    {"nvmlDeviceGetAutoBoostedClocksEnabled", (void *)nvmlDeviceGetAutoBoostedClocksEnabled},
    {"nvmlDeviceSetAutoBoostedClocksEnabled", (void *)nvmlDeviceSetAutoBoostedClocksEnabled},
    {"nvmlDeviceSetDefaultAutoBoostedClocksEnabled", (void *)nvmlDeviceSetDefaultAutoBoostedClocksEnabled},
    {"nvmlDeviceGetFanSpeed", (void *)nvmlDeviceGetFanSpeed},
    {"nvmlDeviceGetFanSpeed_v2", (void *)nvmlDeviceGetFanSpeed_v2},
    {"nvmlDeviceGetTargetFanSpeed", (void *)nvmlDeviceGetTargetFanSpeed},
    {"nvmlDeviceSetDefaultFanSpeed_v2", (void *)nvmlDeviceSetDefaultFanSpeed_v2},
    {"nvmlDeviceGetMinMaxFanSpeed", (void *)nvmlDeviceGetMinMaxFanSpeed},
    {"nvmlDeviceGetFanControlPolicy_v2", (void *)nvmlDeviceGetFanControlPolicy_v2},
    {"nvmlDeviceSetFanControlPolicy", (void *)nvmlDeviceSetFanControlPolicy},
    {"nvmlDeviceGetNumFans", (void *)nvmlDeviceGetNumFans},
    {"nvmlDeviceGetTemperature", (void *)nvmlDeviceGetTemperature},
    {"nvmlDeviceGetTemperatureThreshold", (void *)nvmlDeviceGetTemperatureThreshold},
    {"nvmlDeviceSetTemperatureThreshold", (void *)nvmlDeviceSetTemperatureThreshold},
    {"nvmlDeviceGetThermalSettings", (void *)nvmlDeviceGetThermalSettings},
    {"nvmlDeviceGetPerformanceState", (void *)nvmlDeviceGetPerformanceState},
    {"nvmlDeviceGetCurrentClocksThrottleReasons", (void *)nvmlDeviceGetCurrentClocksThrottleReasons},
    {"nvmlDeviceGetSupportedClocksThrottleReasons", (void *)nvmlDeviceGetSupportedClocksThrottleReasons},
    {"nvmlDeviceGetPowerState", (void *)nvmlDeviceGetPowerState},
    {"nvmlDeviceGetPowerManagementMode", (void *)nvmlDeviceGetPowerManagementMode},
    {"nvmlDeviceGetPowerManagementLimit", (void *)nvmlDeviceGetPowerManagementLimit},
    {"nvmlDeviceGetPowerManagementLimitConstraints", (void *)nvmlDeviceGetPowerManagementLimitConstraints},
    {"nvmlDeviceGetPowerManagementDefaultLimit", (void *)nvmlDeviceGetPowerManagementDefaultLimit},
    {"nvmlDeviceGetPowerUsage", (void *)nvmlDeviceGetPowerUsage},
    {"nvmlDeviceGetTotalEnergyConsumption", (void *)nvmlDeviceGetTotalEnergyConsumption},
    {"nvmlDeviceGetEnforcedPowerLimit", (void *)nvmlDeviceGetEnforcedPowerLimit},
    {"nvmlDeviceGetGpuOperationMode", (void *)nvmlDeviceGetGpuOperationMode},
    {"nvmlDeviceGetMemoryInfo", (void *)nvmlDeviceGetMemoryInfo},
    {"nvmlDeviceGetMemoryInfo_v2", (void *)nvmlDeviceGetMemoryInfo_v2},
    {"nvmlDeviceGetComputeMode", (void *)nvmlDeviceGetComputeMode},
    {"nvmlDeviceGetCudaComputeCapability", (void *)nvmlDeviceGetCudaComputeCapability},
    {"nvmlDeviceGetEccMode", (void *)nvmlDeviceGetEccMode},
    {"nvmlDeviceGetDefaultEccMode", (void *)nvmlDeviceGetDefaultEccMode},
    {"nvmlDeviceGetBoardId", (void *)nvmlDeviceGetBoardId},
    {"nvmlDeviceGetMultiGpuBoard", (void *)nvmlDeviceGetMultiGpuBoard},
    {"nvmlDeviceGetTotalEccErrors", (void *)nvmlDeviceGetTotalEccErrors},
    {"nvmlDeviceGetDetailedEccErrors", (void *)nvmlDeviceGetDetailedEccErrors},
    {"nvmlDeviceGetMemoryErrorCounter", (void *)nvmlDeviceGetMemoryErrorCounter},
    {"nvmlDeviceGetUtilizationRates", (void *)nvmlDeviceGetUtilizationRates},
    {"nvmlDeviceGetEncoderUtilization", (void *)nvmlDeviceGetEncoderUtilization},
    {"nvmlDeviceGetEncoderCapacity", (void *)nvmlDeviceGetEncoderCapacity},
    {"nvmlDeviceGetEncoderStats", (void *)nvmlDeviceGetEncoderStats},
    {"nvmlDeviceGetEncoderSessions", (void *)nvmlDeviceGetEncoderSessions},
    {"nvmlDeviceGetDecoderUtilization", (void *)nvmlDeviceGetDecoderUtilization},
    {"nvmlDeviceGetFBCStats", (void *)nvmlDeviceGetFBCStats},
    {"nvmlDeviceGetFBCSessions", (void *)nvmlDeviceGetFBCSessions},
    {"nvmlDeviceGetDriverModel", (void *)nvmlDeviceGetDriverModel},
    {"nvmlDeviceGetVbiosVersion", (void *)nvmlDeviceGetVbiosVersion},
    {"nvmlDeviceGetBridgeChipInfo", (void *)nvmlDeviceGetBridgeChipInfo},
    {"nvmlDeviceGetComputeRunningProcesses_v3", (void *)nvmlDeviceGetComputeRunningProcesses_v3},
    {"nvmlDeviceGetGraphicsRunningProcesses_v3", (void *)nvmlDeviceGetGraphicsRunningProcesses_v3},
    {"nvmlDeviceGetMPSComputeRunningProcesses_v3", (void *)nvmlDeviceGetMPSComputeRunningProcesses_v3},
    {"nvmlDeviceOnSameBoard", (void *)nvmlDeviceOnSameBoard},
    {"nvmlDeviceGetAPIRestriction", (void *)nvmlDeviceGetAPIRestriction},
    {"nvmlDeviceGetSamples", (void *)nvmlDeviceGetSamples},
    {"nvmlDeviceGetBAR1MemoryInfo", (void *)nvmlDeviceGetBAR1MemoryInfo},
    {"nvmlDeviceGetViolationStatus", (void *)nvmlDeviceGetViolationStatus},
    {"nvmlDeviceGetIrqNum", (void *)nvmlDeviceGetIrqNum},
    {"nvmlDeviceGetNumGpuCores", (void *)nvmlDeviceGetNumGpuCores},
    {"nvmlDeviceGetPowerSource", (void *)nvmlDeviceGetPowerSource},
    {"nvmlDeviceGetMemoryBusWidth", (void *)nvmlDeviceGetMemoryBusWidth},
    {"nvmlDeviceGetPcieLinkMaxSpeed", (void *)nvmlDeviceGetPcieLinkMaxSpeed},
    {"nvmlDeviceGetPcieSpeed", (void *)nvmlDeviceGetPcieSpeed},
    {"nvmlDeviceGetAdaptiveClockInfoStatus", (void *)nvmlDeviceGetAdaptiveClockInfoStatus},
    {"nvmlDeviceGetAccountingMode", (void *)nvmlDeviceGetAccountingMode},
    {"nvmlDeviceGetAccountingStats", (void *)nvmlDeviceGetAccountingStats},
    {"nvmlDeviceGetAccountingPids", (void *)nvmlDeviceGetAccountingPids},
    {"nvmlDeviceGetAccountingBufferSize", (void *)nvmlDeviceGetAccountingBufferSize},
    {"nvmlDeviceGetRetiredPages", (void *)nvmlDeviceGetRetiredPages},
    {"nvmlDeviceGetRetiredPages_v2", (void *)nvmlDeviceGetRetiredPages_v2},
    {"nvmlDeviceGetRetiredPagesPendingStatus", (void *)nvmlDeviceGetRetiredPagesPendingStatus},
    {"nvmlDeviceGetRemappedRows", (void *)nvmlDeviceGetRemappedRows},
    {"nvmlDeviceGetRowRemapperHistogram", (void *)nvmlDeviceGetRowRemapperHistogram},
    {"nvmlDeviceGetArchitecture", (void *)nvmlDeviceGetArchitecture},
    {"nvmlUnitSetLedState", (void *)nvmlUnitSetLedState},
    {"nvmlDeviceSetPersistenceMode", (void *)nvmlDeviceSetPersistenceMode},
    {"nvmlDeviceSetComputeMode", (void *)nvmlDeviceSetComputeMode},
    {"nvmlDeviceSetEccMode", (void *)nvmlDeviceSetEccMode},
    {"nvmlDeviceClearEccErrorCounts", (void *)nvmlDeviceClearEccErrorCounts},
    {"nvmlDeviceSetDriverModel", (void *)nvmlDeviceSetDriverModel},
    {"nvmlDeviceSetGpuLockedClocks", (void *)nvmlDeviceSetGpuLockedClocks},
    {"nvmlDeviceResetGpuLockedClocks", (void *)nvmlDeviceResetGpuLockedClocks},
    {"nvmlDeviceSetMemoryLockedClocks", (void *)nvmlDeviceSetMemoryLockedClocks},
    {"nvmlDeviceResetMemoryLockedClocks", (void *)nvmlDeviceResetMemoryLockedClocks},
    {"nvmlDeviceSetApplicationsClocks", (void *)nvmlDeviceSetApplicationsClocks},
    {"nvmlDeviceGetClkMonStatus", (void *)nvmlDeviceGetClkMonStatus},
    {"nvmlDeviceSetPowerManagementLimit", (void *)nvmlDeviceSetPowerManagementLimit},
    {"nvmlDeviceSetGpuOperationMode", (void *)nvmlDeviceSetGpuOperationMode},
    {"nvmlDeviceSetAPIRestriction", (void *)nvmlDeviceSetAPIRestriction},
    {"nvmlDeviceSetAccountingMode", (void *)nvmlDeviceSetAccountingMode},
    {"nvmlDeviceClearAccountingPids", (void *)nvmlDeviceClearAccountingPids},
    {"nvmlDeviceGetNvLinkState", (void *)nvmlDeviceGetNvLinkState},
    {"nvmlDeviceGetNvLinkVersion", (void *)nvmlDeviceGetNvLinkVersion},
    {"nvmlDeviceGetNvLinkCapability", (void *)nvmlDeviceGetNvLinkCapability},
    {"nvmlDeviceGetNvLinkRemotePciInfo_v2", (void *)nvmlDeviceGetNvLinkRemotePciInfo_v2},
    {"nvmlDeviceGetNvLinkErrorCounter", (void *)nvmlDeviceGetNvLinkErrorCounter},
    {"nvmlDeviceResetNvLinkErrorCounters", (void *)nvmlDeviceResetNvLinkErrorCounters},
    {"nvmlDeviceSetNvLinkUtilizationControl", (void *)nvmlDeviceSetNvLinkUtilizationControl},
    {"nvmlDeviceGetNvLinkUtilizationControl", (void *)nvmlDeviceGetNvLinkUtilizationControl},
    {"nvmlDeviceGetNvLinkUtilizationCounter", (void *)nvmlDeviceGetNvLinkUtilizationCounter},
    {"nvmlDeviceFreezeNvLinkUtilizationCounter", (void *)nvmlDeviceFreezeNvLinkUtilizationCounter},
    {"nvmlDeviceResetNvLinkUtilizationCounter", (void *)nvmlDeviceResetNvLinkUtilizationCounter},
    {"nvmlDeviceGetNvLinkRemoteDeviceType", (void *)nvmlDeviceGetNvLinkRemoteDeviceType},
    {"nvmlEventSetCreate", (void *)nvmlEventSetCreate},
    {"nvmlDeviceRegisterEvents", (void *)nvmlDeviceRegisterEvents},
    {"nvmlDeviceGetSupportedEventTypes", (void *)nvmlDeviceGetSupportedEventTypes},
    {"nvmlEventSetWait_v2", (void *)nvmlEventSetWait_v2},
    {"nvmlEventSetFree", (void *)nvmlEventSetFree},
    {"nvmlDeviceModifyDrainState", (void *)nvmlDeviceModifyDrainState},
    {"nvmlDeviceQueryDrainState", (void *)nvmlDeviceQueryDrainState},
    {"nvmlDeviceRemoveGpu_v2", (void *)nvmlDeviceRemoveGpu_v2},
    {"nvmlDeviceDiscoverGpus", (void *)nvmlDeviceDiscoverGpus},
    {"nvmlDeviceGetFieldValues", (void *)nvmlDeviceGetFieldValues},
    {"nvmlDeviceClearFieldValues", (void *)nvmlDeviceClearFieldValues},
    {"nvmlDeviceGetVirtualizationMode", (void *)nvmlDeviceGetVirtualizationMode},
    {"nvmlDeviceGetHostVgpuMode", (void *)nvmlDeviceGetHostVgpuMode},
    {"nvmlDeviceSetVirtualizationMode", (void *)nvmlDeviceSetVirtualizationMode},
    {"nvmlDeviceGetGridLicensableFeatures_v4", (void *)nvmlDeviceGetGridLicensableFeatures_v4},
    {"nvmlDeviceGetProcessUtilization", (void *)nvmlDeviceGetProcessUtilization},
    {"nvmlDeviceGetGspFirmwareVersion", (void *)nvmlDeviceGetGspFirmwareVersion},
    {"nvmlDeviceGetGspFirmwareMode", (void *)nvmlDeviceGetGspFirmwareMode},
    {"nvmlGetVgpuDriverCapabilities", (void *)nvmlGetVgpuDriverCapabilities},
    {"nvmlDeviceGetVgpuCapabilities", (void *)nvmlDeviceGetVgpuCapabilities},
    {"nvmlDeviceGetSupportedVgpus", (void *)nvmlDeviceGetSupportedVgpus},
    {"nvmlDeviceGetCreatableVgpus", (void *)nvmlDeviceGetCreatableVgpus},
    {"nvmlVgpuTypeGetClass", (void *)nvmlVgpuTypeGetClass},
    {"nvmlVgpuTypeGetName", (void *)nvmlVgpuTypeGetName},
    {"nvmlVgpuTypeGetGpuInstanceProfileId", (void *)nvmlVgpuTypeGetGpuInstanceProfileId},
    {"nvmlVgpuTypeGetDeviceID", (void *)nvmlVgpuTypeGetDeviceID},
    {"nvmlVgpuTypeGetFramebufferSize", (void *)nvmlVgpuTypeGetFramebufferSize},
    {"nvmlVgpuTypeGetNumDisplayHeads", (void *)nvmlVgpuTypeGetNumDisplayHeads},
    {"nvmlVgpuTypeGetResolution", (void *)nvmlVgpuTypeGetResolution},
    {"nvmlVgpuTypeGetLicense", (void *)nvmlVgpuTypeGetLicense},
    {"nvmlVgpuTypeGetFrameRateLimit", (void *)nvmlVgpuTypeGetFrameRateLimit},
    {"nvmlVgpuTypeGetMaxInstances", (void *)nvmlVgpuTypeGetMaxInstances},
    {"nvmlVgpuTypeGetMaxInstancesPerVm", (void *)nvmlVgpuTypeGetMaxInstancesPerVm},
    {"nvmlDeviceGetActiveVgpus", (void *)nvmlDeviceGetActiveVgpus},
    {"nvmlVgpuInstanceGetVmID", (void *)nvmlVgpuInstanceGetVmID},
    {"nvmlVgpuInstanceGetUUID", (void *)nvmlVgpuInstanceGetUUID},
    {"nvmlVgpuInstanceGetVmDriverVersion", (void *)nvmlVgpuInstanceGetVmDriverVersion},
    {"nvmlVgpuInstanceGetFbUsage", (void *)nvmlVgpuInstanceGetFbUsage},
    {"nvmlVgpuInstanceGetLicenseStatus", (void *)nvmlVgpuInstanceGetLicenseStatus},
    {"nvmlVgpuInstanceGetType", (void *)nvmlVgpuInstanceGetType},
    {"nvmlVgpuInstanceGetFrameRateLimit", (void *)nvmlVgpuInstanceGetFrameRateLimit},
    {"nvmlVgpuInstanceGetEccMode", (void *)nvmlVgpuInstanceGetEccMode},
    {"nvmlVgpuInstanceGetEncoderCapacity", (void *)nvmlVgpuInstanceGetEncoderCapacity},
    {"nvmlVgpuInstanceSetEncoderCapacity", (void *)nvmlVgpuInstanceSetEncoderCapacity},
    {"nvmlVgpuInstanceGetEncoderStats", (void *)nvmlVgpuInstanceGetEncoderStats},
    {"nvmlVgpuInstanceGetEncoderSessions", (void *)nvmlVgpuInstanceGetEncoderSessions},
    {"nvmlVgpuInstanceGetFBCStats", (void *)nvmlVgpuInstanceGetFBCStats},
    {"nvmlVgpuInstanceGetFBCSessions", (void *)nvmlVgpuInstanceGetFBCSessions},
    {"nvmlVgpuInstanceGetGpuInstanceId", (void *)nvmlVgpuInstanceGetGpuInstanceId},
    {"nvmlVgpuInstanceGetGpuPciId", (void *)nvmlVgpuInstanceGetGpuPciId},
    {"nvmlVgpuTypeGetCapabilities", (void *)nvmlVgpuTypeGetCapabilities},
    {"nvmlVgpuInstanceGetMetadata", (void *)nvmlVgpuInstanceGetMetadata},
    {"nvmlDeviceGetVgpuMetadata", (void *)nvmlDeviceGetVgpuMetadata},
    {"nvmlGetVgpuCompatibility", (void *)nvmlGetVgpuCompatibility},
    {"nvmlDeviceGetPgpuMetadataString", (void *)nvmlDeviceGetPgpuMetadataString},
    {"nvmlDeviceGetVgpuSchedulerLog", (void *)nvmlDeviceGetVgpuSchedulerLog},
    {"nvmlDeviceGetVgpuSchedulerState", (void *)nvmlDeviceGetVgpuSchedulerState},
    {"nvmlDeviceGetVgpuSchedulerCapabilities", (void *)nvmlDeviceGetVgpuSchedulerCapabilities},
    {"nvmlGetVgpuVersion", (void *)nvmlGetVgpuVersion},
    {"nvmlSetVgpuVersion", (void *)nvmlSetVgpuVersion},
    {"nvmlDeviceGetVgpuUtilization", (void *)nvmlDeviceGetVgpuUtilization},
    {"nvmlDeviceGetVgpuProcessUtilization", (void *)nvmlDeviceGetVgpuProcessUtilization},
    {"nvmlVgpuInstanceGetAccountingMode", (void *)nvmlVgpuInstanceGetAccountingMode},
    {"nvmlVgpuInstanceGetAccountingPids", (void *)nvmlVgpuInstanceGetAccountingPids},
    {"nvmlVgpuInstanceGetAccountingStats", (void *)nvmlVgpuInstanceGetAccountingStats},
    {"nvmlVgpuInstanceClearAccountingPids", (void *)nvmlVgpuInstanceClearAccountingPids},
    {"nvmlVgpuInstanceGetLicenseInfo_v2", (void *)nvmlVgpuInstanceGetLicenseInfo_v2},
    {"nvmlGetExcludedDeviceCount", (void *)nvmlGetExcludedDeviceCount},
    {"nvmlGetExcludedDeviceInfoByIndex", (void *)nvmlGetExcludedDeviceInfoByIndex},
    {"nvmlDeviceSetMigMode", (void *)nvmlDeviceSetMigMode},
    {"nvmlDeviceGetMigMode", (void *)nvmlDeviceGetMigMode},
    {"nvmlDeviceGetGpuInstanceProfileInfo", (void *)nvmlDeviceGetGpuInstanceProfileInfo},
    {"nvmlDeviceGetGpuInstanceProfileInfoV", (void *)nvmlDeviceGetGpuInstanceProfileInfoV},
    {"nvmlDeviceGetGpuInstancePossiblePlacements_v2", (void *)nvmlDeviceGetGpuInstancePossiblePlacements_v2},
    {"nvmlDeviceGetGpuInstanceRemainingCapacity", (void *)nvmlDeviceGetGpuInstanceRemainingCapacity},
    {"nvmlDeviceCreateGpuInstance", (void *)nvmlDeviceCreateGpuInstance},
    {"nvmlDeviceCreateGpuInstanceWithPlacement", (void *)nvmlDeviceCreateGpuInstanceWithPlacement},
    {"nvmlGpuInstanceDestroy", (void *)nvmlGpuInstanceDestroy},
    {"nvmlDeviceGetGpuInstances", (void *)nvmlDeviceGetGpuInstances},
    {"nvmlDeviceGetGpuInstanceById", (void *)nvmlDeviceGetGpuInstanceById},
    {"nvmlGpuInstanceGetInfo", (void *)nvmlGpuInstanceGetInfo},
    {"nvmlGpuInstanceGetComputeInstanceProfileInfo", (void *)nvmlGpuInstanceGetComputeInstanceProfileInfo},
    {"nvmlGpuInstanceGetComputeInstanceProfileInfoV", (void *)nvmlGpuInstanceGetComputeInstanceProfileInfoV},
    {"nvmlGpuInstanceGetComputeInstanceRemainingCapacity", (void *)nvmlGpuInstanceGetComputeInstanceRemainingCapacity},
    {"nvmlGpuInstanceGetComputeInstancePossiblePlacements", (void *)nvmlGpuInstanceGetComputeInstancePossiblePlacements},
    {"nvmlGpuInstanceCreateComputeInstance", (void *)nvmlGpuInstanceCreateComputeInstance},
    {"nvmlGpuInstanceCreateComputeInstanceWithPlacement", (void *)nvmlGpuInstanceCreateComputeInstanceWithPlacement},
    {"nvmlComputeInstanceDestroy", (void *)nvmlComputeInstanceDestroy},
    {"nvmlGpuInstanceGetComputeInstances", (void *)nvmlGpuInstanceGetComputeInstances},
    {"nvmlGpuInstanceGetComputeInstanceById", (void *)nvmlGpuInstanceGetComputeInstanceById},
    {"nvmlComputeInstanceGetInfo_v2", (void *)nvmlComputeInstanceGetInfo_v2},
    {"nvmlDeviceIsMigDeviceHandle", (void *)nvmlDeviceIsMigDeviceHandle},
    {"nvmlDeviceGetGpuInstanceId", (void *)nvmlDeviceGetGpuInstanceId},
    {"nvmlDeviceGetComputeInstanceId", (void *)nvmlDeviceGetComputeInstanceId},
    {"nvmlDeviceGetMaxMigDeviceCount", (void *)nvmlDeviceGetMaxMigDeviceCount},
    {"nvmlDeviceGetMigDeviceHandleByIndex", (void *)nvmlDeviceGetMigDeviceHandleByIndex},
    {"nvmlDeviceGetDeviceHandleFromMigDeviceHandle", (void *)nvmlDeviceGetDeviceHandleFromMigDeviceHandle},
    {"nvmlDeviceGetBusType", (void *)nvmlDeviceGetBusType},
    {"nvmlDeviceGetDynamicPstatesInfo", (void *)nvmlDeviceGetDynamicPstatesInfo},
    {"nvmlDeviceSetFanSpeed_v2", (void *)nvmlDeviceSetFanSpeed_v2},
    {"nvmlDeviceGetGpcClkVfOffset", (void *)nvmlDeviceGetGpcClkVfOffset},
    {"nvmlDeviceSetGpcClkVfOffset", (void *)nvmlDeviceSetGpcClkVfOffset},
    {"nvmlDeviceGetMemClkVfOffset", (void *)nvmlDeviceGetMemClkVfOffset},
    {"nvmlDeviceSetMemClkVfOffset", (void *)nvmlDeviceSetMemClkVfOffset},
    {"nvmlDeviceGetMinMaxClockOfPState", (void *)nvmlDeviceGetMinMaxClockOfPState},
    {"nvmlDeviceGetSupportedPerformanceStates", (void *)nvmlDeviceGetSupportedPerformanceStates},
    {"nvmlDeviceGetGpcClkMinMaxVfOffset", (void *)nvmlDeviceGetGpcClkMinMaxVfOffset},
    {"nvmlDeviceGetMemClkMinMaxVfOffset", (void *)nvmlDeviceGetMemClkMinMaxVfOffset},
    {"nvmlDeviceGetGpuFabricInfo", (void *)nvmlDeviceGetGpuFabricInfo},
    {"nvmlGpmMetricsGet", (void *)nvmlGpmMetricsGet},
    {"nvmlGpmSampleFree", (void *)nvmlGpmSampleFree},
    {"nvmlGpmSampleAlloc", (void *)nvmlGpmSampleAlloc},
    {"nvmlGpmSampleGet", (void *)nvmlGpmSampleGet},
    {"nvmlGpmMigSampleGet", (void *)nvmlGpmMigSampleGet},
    {"nvmlGpmQueryDeviceSupport", (void *)nvmlGpmQueryDeviceSupport},
    {"nvmlDeviceSetNvLinkDeviceLowPowerThreshold", (void *)nvmlDeviceSetNvLinkDeviceLowPowerThreshold},
    {"nvmlInit", (void *)nvmlInit_v2},
    {"nvmlDeviceGetPciInfo", (void *)nvmlDeviceGetPciInfo_v3},
    {"nvmlDeviceGetCount", (void *)nvmlDeviceGetCount_v2},
    {"nvmlDeviceGetHandleByIndex", (void *)nvmlDeviceGetHandleByIndex_v2},
    {"nvmlDeviceGetHandleByPciBusId", (void *)nvmlDeviceGetHandleByPciBusId_v2},
    {"nvmlDeviceGetNvLinkRemotePciInfo", (void *)nvmlDeviceGetNvLinkRemotePciInfo_v2},
    {"nvmlDeviceRemoveGpu", (void *)nvmlDeviceRemoveGpu_v2},
    {"nvmlDeviceGetGridLicensableFeatures", (void *)nvmlDeviceGetGridLicensableFeatures_v4},
    {"nvmlEventSetWait", (void *)nvmlEventSetWait_v2},
    {"nvmlDeviceGetAttributes", (void *)nvmlDeviceGetAttributes_v2},
    {"nvmlComputeInstanceGetInfo", (void *)nvmlComputeInstanceGetInfo_v2},
    {"nvmlDeviceGetComputeRunningProcesses", (void *)nvmlDeviceGetComputeRunningProcesses_v3},
    {"nvmlDeviceGetGraphicsRunningProcesses", (void *)nvmlDeviceGetGraphicsRunningProcesses_v3},
    {"nvmlDeviceGetMPSComputeRunningProcesses", (void *)nvmlDeviceGetMPSComputeRunningProcesses_v3},
    {"nvmlGetBlacklistDeviceCount", (void *)nvmlGetExcludedDeviceCount},
    {"nvmlGetBlacklistDeviceInfoByIndex", (void *)nvmlGetExcludedDeviceInfoByIndex},
    {"nvmlDeviceGetGpuInstancePossiblePlacements", (void *)nvmlDeviceGetGpuInstancePossiblePlacements_v2},
    {"nvmlVgpuInstanceGetLicenseInfo", (void *)nvmlVgpuInstanceGetLicenseInfo_v2},
    {"cuDeviceTotalMem", (void *)cuDeviceTotalMem_v2},
    {"cuCtxCreate", (void *)cuCtxCreate_v2},
    {"cuCtxCreate_v3", (void *)cuCtxCreate_v3},
    {"cuModuleGetGlobal", (void *)cuModuleGetGlobal_v2},
    {"cuMemGetInfo", (void *)cuMemGetInfo_v2},
    {"cuMemAlloc", (void *)cuMemAlloc_v2},
    {"cuMemAllocPitch", (void *)cuMemAllocPitch_v2},
    {"cuMemFree", (void *)cuMemFree_v2},
    {"cuMemGetAddressRange", (void *)cuMemGetAddressRange_v2},
    {"cuMemAllocHost", (void *)cuMemAllocHost_v2},
    {"cuMemHostGetDevicePointer", (void *)cuMemHostGetDevicePointer_v2},
    {"cuMemcpyHtoD", (void *)cuMemcpyHtoD_v2},
    {"cuMemcpyDtoH", (void *)cuMemcpyDtoH_v2},
    {"cuMemcpyDtoD", (void *)cuMemcpyDtoD_v2},
    {"cuMemcpyDtoA", (void *)cuMemcpyDtoA_v2},
    {"cuMemcpyAtoD", (void *)cuMemcpyAtoD_v2},
    {"cuMemcpyHtoA", (void *)cuMemcpyHtoA_v2},
    {"cuMemcpyAtoH", (void *)cuMemcpyAtoH_v2},
    {"cuMemcpyAtoA", (void *)cuMemcpyAtoA_v2},
    {"cuMemcpyHtoAAsync", (void *)cuMemcpyHtoAAsync_v2},
    {"cuMemcpyAtoHAsync", (void *)cuMemcpyAtoHAsync_v2},
    {"cuMemcpy2D", (void *)cuMemcpy2D_v2},
    {"cuMemcpy2DUnaligned", (void *)cuMemcpy2DUnaligned_v2},
    {"cuMemcpy3D", (void *)cuMemcpy3D_v2},
    {"cuMemcpyHtoDAsync", (void *)cuMemcpyHtoDAsync_v2},
    {"cuMemcpyDtoHAsync", (void *)cuMemcpyDtoHAsync_v2},
    {"cuMemcpyDtoDAsync", (void *)cuMemcpyDtoDAsync_v2},
    {"cuMemcpy2DAsync", (void *)cuMemcpy2DAsync_v2},
    {"cuMemcpy3DAsync", (void *)cuMemcpy3DAsync_v2},
    {"cuMemsetD8", (void *)cuMemsetD8_v2},
    {"cuMemsetD16", (void *)cuMemsetD16_v2},
    {"cuMemsetD32", (void *)cuMemsetD32_v2},
    {"cuMemsetD2D8", (void *)cuMemsetD2D8_v2},
    {"cuMemsetD2D16", (void *)cuMemsetD2D16_v2},
    {"cuMemsetD2D32", (void *)cuMemsetD2D32_v2},
    {"cuArrayCreate", (void *)cuArrayCreate_v2},
    {"cuArrayGetDescriptor", (void *)cuArrayGetDescriptor_v2},
    {"cuArray3DCreate", (void *)cuArray3DCreate_v2},
    {"cuArray3DGetDescriptor", (void *)cuArray3DGetDescriptor_v2},
    {"cuTexRefSetAddress", (void *)cuTexRefSetAddress_v2},
    {"cuTexRefGetAddress", (void *)cuTexRefGetAddress_v2},
    {"cuGraphicsResourceGetMappedPointer", (void *)cuGraphicsResourceGetMappedPointer_v2},
    {"cuCtxDestroy", (void *)cuCtxDestroy_v2},
    {"cuCtxPopCurrent", (void *)cuCtxPopCurrent_v2},
    {"cuCtxPushCurrent", (void *)cuCtxPushCurrent_v2},
    {"cuStreamDestroy", (void *)cuStreamDestroy_v2},
    {"cuEventDestroy", (void *)cuEventDestroy_v2},
    {"cuTexRefSetAddress2D", (void *)cuTexRefSetAddress2D_v3},
    {"cuLinkCreate", (void *)cuLinkCreate_v2},
    {"cuLinkAddData", (void *)cuLinkAddData_v2},
    {"cuLinkAddFile", (void *)cuLinkAddFile_v2},
    {"cuMemHostRegister", (void *)cuMemHostRegister_v2},
    {"cuGraphicsResourceSetMapFlags", (void *)cuGraphicsResourceSetMapFlags_v2},
    {"cuStreamBeginCapture", (void *)cuStreamBeginCapture_v2},
    {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease_v2},
    {"cuDevicePrimaryCtxReset", (void *)cuDevicePrimaryCtxReset_v2},
    {"cuDevicePrimaryCtxSetFlags", (void *)cuDevicePrimaryCtxSetFlags_v2},
    {"cuDeviceGetUuid_v2", (void *)cuDeviceGetUuid_v2},
    {"cuIpcOpenMemHandle", (void *)cuIpcOpenMemHandle_v2},
    {"cuGraphInstantiate", (void *)cuGraphInstantiateWithFlags},
    {"cuGraphExecUpdate", (void *)cuGraphExecUpdate_v2},
    {"cuGetProcAddress", (void *)cuGetProcAddress_v2},
    {"cuGraphAddKernelNode", (void *)cuGraphAddKernelNode_v2},
    {"cuGraphKernelNodeGetParams", (void *)cuGraphKernelNodeGetParams_v2},
    {"cuGraphKernelNodeSetParams", (void *)cuGraphKernelNodeSetParams_v2},
    {"cuGraphExecKernelNodeSetParams", (void *)cuGraphExecKernelNodeSetParams_v2},
    {"cuStreamWriteValue32", (void *)cuStreamWriteValue32_v2},
    {"cuStreamWaitValue32", (void *)cuStreamWaitValue32_v2},
    {"cuStreamWriteValue64", (void *)cuStreamWriteValue64_v2},
    {"cuStreamWaitValue64", (void *)cuStreamWaitValue64_v2},
    {"cuStreamBatchMemOp", (void *)cuStreamBatchMemOp_v2},
    {"cuStreamGetCaptureInfo", (void *)cuStreamGetCaptureInfo_v2},
    {"cuStreamGetCaptureInfo_v2", (void *)cuStreamGetCaptureInfo_v2},
    {"cudaSignalExternalSemaphoresAsync", (void *)cudaSignalExternalSemaphoresAsync_v2},
    {"cudaWaitExternalSemaphoresAsync", (void *)cudaWaitExternalSemaphoresAsync_v2},
    {"cudaStreamGetCaptureInfo", (void *)cudaStreamGetCaptureInfo_v2},
    {"cudaGetDeviceProperties", (void *)cudaGetDeviceProperties_v2},

};

void *get_function_pointer(const char *name)
{
    auto it = functionMap.find(name);
    if (it != functionMap.end())
        return it->second;
    return nullptr;
}
