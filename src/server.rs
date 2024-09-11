use std::{future::Future, net::SocketAddr};

use crate::api::Scuda;
use libc::c_char;
use nvml_wrapper_sys::bindings::{nvmlDevice_t, nvmlReturn_enum_NVML_SUCCESS, NvmlLib};
use tarpc::context;

#[derive(Clone)]
pub struct ScudaServer(pub SocketAddr);

impl Scuda for ScudaServer {
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
}

pub async fn spawn(fut: impl Future<Output = ()> + Send + 'static) {
    tokio::spawn(fut);
}
