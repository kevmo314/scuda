use libc::RTLD_NEXT;
use libc::{c_char, c_int, c_uint, c_void};
use nvml_wrapper_sys::bindings::{
    nvmlReturn_enum_NVML_ERROR_GPU_IS_LOST, nvmlReturn_enum_NVML_ERROR_INSUFFICIENT_SIZE,
    nvmlReturn_enum_NVML_SUCCESS, nvmlReturn_t,
};
use std::ffi::{CStr, CString};
use std::net::{IpAddr, Ipv6Addr};
use std::sync::LazyLock;
use tarpc::client::{NewClient, RpcError};
use tarpc::tokio_serde::formats::Json;
use tarpc::{client, context};
use tokio::runtime::{self, Runtime};

use crate::api;

static RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .build()
        .unwrap()
});

static CLIENT: LazyLock<api::ScudaClient> = LazyLock::new(|| {
    let server_addr = (IpAddr::V6(Ipv6Addr::LOCALHOST), 31337);
    let mut transport = tarpc::serde_transport::tcp::connect(server_addr, Json::default);
    transport.config_mut().max_frame_length(usize::MAX);
    let tx = RUNTIME.block_on(transport).unwrap();
    let NewClient { client, dispatch } = api::ScudaClient::new(client::Config::default(), tx);
    RUNTIME.spawn(dispatch);

    client
});

fn handle_response(response: Result<nvmlReturn_t, RpcError>) -> nvmlReturn_t {
    match response {
        Ok(response) => response,
        Err(e) => {
            eprintln!("Error: {:?}", e);
            nvmlReturn_enum_NVML_ERROR_GPU_IS_LOST
        }
    }
}

// 4.11 Initialization and Cleanup
#[no_mangle]
pub extern "C" fn nvmlInitWithFlags(flags: u32) -> nvmlReturn_t {
    handle_response(RUNTIME.block_on(CLIENT.nvmlInitWithFlags(context::current(), flags)))
}

#[no_mangle]
pub extern "C" fn nvmlInit_v2() -> nvmlReturn_t {
    handle_response(RUNTIME.block_on(CLIENT.nvmlInit_v2(context::current())))
}

#[no_mangle]
pub extern "C" fn nvmlShutdown() -> nvmlReturn_t {
    handle_response(RUNTIME.block_on(CLIENT.nvmlShutdown(context::current())))
}

// 4.12 Error reporting
#[no_mangle]
pub extern "C" fn nvmlErrorString(result: nvmlReturn_t) -> *const c_char {
    match RUNTIME.block_on(CLIENT.nvmlErrorString(context::current(), result)) {
        Ok(response) => CString::new(response).unwrap().into_raw(),
        Err(_) => CString::new("Unknown error").unwrap().into_raw(),
    }
}

// 4.14 System Queries
#[no_mangle]
pub extern "C" fn nvmlSystemGetCudaDriverVersion(cuda_driver_version: *mut c_int) -> nvmlReturn_t {
    match RUNTIME.block_on(CLIENT.nvmlSystemGetCudaDriverVersion(context::current())) {
        Ok((result, version)) => {
            unsafe {
                *cuda_driver_version = version;
            }
            result
        }
        Err(_) => nvmlReturn_enum_NVML_ERROR_GPU_IS_LOST,
    }
}

#[no_mangle]
pub extern "C" fn nvmlSystemGetCudaDriverVersion_v2(
    cuda_driver_version: *mut c_int,
) -> nvmlReturn_t {
    match RUNTIME.block_on(CLIENT.nvmlSystemGetCudaDriverVersion_v2(context::current())) {
        Ok((result, version)) => {
            unsafe {
                *cuda_driver_version = version;
            }
            result
        }
        Err(_) => nvmlReturn_enum_NVML_ERROR_GPU_IS_LOST,
    }
}

#[no_mangle]
pub extern "C" fn nvmlSystemGetDriverVersion(
    driver_version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    match RUNTIME.block_on(CLIENT.nvmlSystemGetDriverVersion(context::current(), length)) {
        Ok((result, version)) => {
            let version = CString::new(version).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(version.as_ptr(), driver_version, length as usize);
            }
            result
        }
        Err(_) => nvmlReturn_enum_NVML_ERROR_INSUFFICIENT_SIZE,
    }
}

#[no_mangle]
pub extern "C" fn nvmlSystemGetNVMLVersion(
    nvml_version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    match RUNTIME.block_on(CLIENT.nvmlSystemGetNVMLVersion(context::current(), length)) {
        Ok((result, version)) => {
            let version = CString::new(version).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(version.as_ptr(), nvml_version, length as usize);
            }
            result
        }
        Err(_) => nvmlReturn_enum_NVML_ERROR_INSUFFICIENT_SIZE,
    }
}

#[no_mangle]
pub extern "C" fn nvmlSystemGetProcessName(
    pid: c_uint,
    process_name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t {
    let length = length as u32;
    let (result, name) = RUNTIME
        .block_on(CLIENT.nvmlSystemGetProcessName(context::current(), pid, length))
        .unwrap();
    if result == nvmlReturn_enum_NVML_SUCCESS {
        let name = CString::new(name).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(name.as_ptr(), process_name, length as usize);
        }
    }
    result
}

#[no_mangle]
pub extern "C" fn nvmlDeviceGetCount_v2(device_count: *mut c_uint) -> nvmlReturn_t {
    unsafe {
        *device_count = 1;
    }
    nvmlReturn_enum_NVML_SUCCESS
}

extern "C" {
    pub fn dlvsym(
        handle: *mut c_void,
        symbol: *const c_char,
        version: *const c_char,
    ) -> *mut c_void;
}

#[no_mangle]
pub extern "C" fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void {
    unsafe {
        let symbol_name = CStr::from_ptr(name).to_str().unwrap();

        match symbol_name {
            "nvmlInitWithFlags" => nvmlInitWithFlags as *mut c_void,
            "nvmlSystemGetCudaDriverVersion_v2" => nvmlSystemGetCudaDriverVersion_v2 as *mut c_void,
            "nvmlDeviceGetCount_v2" => nvmlDeviceGetCount_v2 as *mut c_void,
            "dlsym" => dlvsym as *mut c_void,
            _ => {
                let dlsym = CString::new("dlsym").unwrap();
                let glibc = CString::new("GLIBC_2.2.5").unwrap();
                let real_dlsym: unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void =
                    std::mem::transmute(dlvsym(RTLD_NEXT, dlsym.as_ptr(), glibc.as_ptr()));
                real_dlsym(handle, name)
            }
        }
    }
}
