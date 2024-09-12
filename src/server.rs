use std::{
    ffi::{CStr},
    future::Future,
    net::SocketAddr,
    sync::LazyLock,
};

use crate::api::Scuda;
use libc::c_char;
use nvml_wrapper_sys::bindings::{
    nvmlDevice_t, nvmlReturn_enum_NVML_SUCCESS, nvmlReturn_t, NvmlLib
};

use tarpc::context;

#[derive(Clone)]
pub struct ScudaServer(pub SocketAddr);

static NVML_LIB: LazyLock<NvmlLib> =
    LazyLock::new(|| unsafe { NvmlLib::new("libnvidia-ml.so").unwrap() });

impl Scuda for ScudaServer {
    // // 4.11 Initialization and Cleanup
    // async fn nvmlInitWithFlags(self, _: context::Context, flags: u32) -> nvmlReturn_t {
    //     let res = unsafe { NVML_LIB.nvmlInitWithFlags(flags) };

    //     println!("{}", res);

    //     res
    // }

    async fn nvmlInit_v2(self, _: ::tarpc::context::Context) -> nvmlReturn_t {
        unsafe { NVML_LIB.nvmlInit_v2() }
    }

    async fn nvmlShutdown(self, _: context::Context) -> nvmlReturn_t {
        unsafe { NVML_LIB.nvmlShutdown() }
    }

    // 4.12 Error reporting
    async fn nvmlErrorString(self, _: context::Context, result: nvmlReturn_t) -> String {
        let error_string = unsafe { CStr::from_ptr(NVML_LIB.nvmlErrorString(result)) };
        error_string.to_string_lossy().into_owned()
    }

    // 4.14 System Queries
    async fn nvmlSystemGetCudaDriverVersion(self, _: context::Context) -> (nvmlReturn_t, i32) {
        let mut cuda_driver_version = 0;
        let result = unsafe { NVML_LIB.nvmlSystemGetCudaDriverVersion(&mut cuda_driver_version) };
        (result, cuda_driver_version)
    }

    async fn nvmlSystemGetCudaDriverVersion_v2(self, _: context::Context) -> (nvmlReturn_t, i32) {
        let mut cuda_driver_version = 0;
        let result =
            unsafe { NVML_LIB.nvmlSystemGetCudaDriverVersion_v2(&mut cuda_driver_version) };
        (result, cuda_driver_version)
    }

    async fn nvmlSystemGetDriverVersion(
        self,
        _: context::Context,
        length: u32,
    ) -> (nvmlReturn_t, String) {
        let mut driver_version = vec![0u8; length as usize];
        let driver_version_ptr = driver_version.as_mut_ptr() as *mut c_char;
        let result = unsafe { NVML_LIB.nvmlSystemGetDriverVersion(driver_version_ptr, length) };
        if result != nvmlReturn_enum_NVML_SUCCESS {
            return (result, String::new());
        }
        let driver_version = unsafe { CStr::from_ptr(driver_version_ptr) };
        (result, driver_version.to_string_lossy().into_owned())
    }

    async fn nvmlSystemGetNVMLVersion(
        self,
        _: context::Context,
        length: u32,
    ) -> (nvmlReturn_t, String) {
        let mut nvml_version = vec![0u8; length as usize];
        let nvml_version_ptr = nvml_version.as_mut_ptr() as *mut c_char;
        let result = unsafe { NVML_LIB.nvmlSystemGetNVMLVersion(nvml_version_ptr, length) };
        if result != nvmlReturn_enum_NVML_SUCCESS {
            return (result, String::new());
        }
        let nvml_version = unsafe { CStr::from_ptr(nvml_version_ptr) };
        (result, nvml_version.to_string_lossy().into_owned())
    }

    async fn nvmlSystemGetProcessName(
        self,
        _: context::Context,
        pid: u32,
        length: u32,
    ) -> (nvmlReturn_t, String) {
        let mut process_name = vec![0u8; length as usize];
        let process_name_ptr = process_name.as_mut_ptr() as *mut c_char;
        let result = unsafe { NVML_LIB.nvmlSystemGetProcessName(pid, process_name_ptr, length) };
        if result != nvmlReturn_enum_NVML_SUCCESS {
            return (result, String::new());
        }
        let process_name = unsafe { CStr::from_ptr(process_name_ptr) };
        (result, process_name.to_string_lossy().into_owned())
    }

    async fn nvml_device_get_name(self, _: context::Context, device: u64, length: u32) -> String {
        let mut name = vec![0u8; length as usize];
        let name_ptr = name.as_mut_ptr() as *mut c_char;
        let lib = unsafe { NvmlLib::new("libnvidia-ml.so").unwrap() };
        let result = unsafe { lib.nvmlDeviceGetName(device as nvmlDevice_t, name_ptr, length) };
        // set the first five bytes to MUGIT
        name[0] = 77;
        name[1] = 85;
        name[2] = 71;
        name[3] = 73;
        name[4] = 84;
        if result != nvmlReturn_enum_NVML_SUCCESS {
            return String::new();
        }
        String::from_utf8(name).unwrap()
    }

    async fn nvml_system_get_cuda_driver_version_v2(self, _: context::Context) -> i32 {
        let mut cuda_driver_version = 0;
        let lib = unsafe { NvmlLib::new("libnvidia-ml.so").unwrap() };
        unsafe {
            let r = lib.nvmlInit_v2();
            if r != nvmlReturn_enum_NVML_SUCCESS {
                println!("Failed to initialize NVML: {}", r);
                return 543534;
            }
        }
        let result = unsafe { lib.nvmlSystemGetCudaDriverVersion_v2(&mut cuda_driver_version) };
        println!(
            "CUDA Driver Version: {} result: {}",
            cuda_driver_version, result
        );
        if result != nvmlReturn_enum_NVML_SUCCESS {
            return 123;
        }
        cuda_driver_version
    }

    // 4.16 resource requests
    async fn nvmlDeviceGetFanSpeed(self, _: ::tarpc::context::Context, device: u64, speed: u32) -> nvmlReturn_t {
        let result = unsafe { NVML_LIB.nvmlDeviceGetFanSpeed(device as nvmlDevice_t, speed as *mut u32) };

        println!("Fan speed response: {}", result);

        result
    }
}

pub async fn spawn(fut: impl Future<Output = ()> + Send + 'static) {
    tokio::spawn(fut);
}
