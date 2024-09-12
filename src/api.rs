#![allow(non_snake_case)] // easier to match the C API.

use nvml_wrapper_sys::bindings::nvmlReturn_t;
#[tarpc::service]
pub trait Scuda {
    // 4.11 Initialization and Cleanup

    // commmenting out for now
    // async fn nvmlInitWithFlags(flags: u32) -> nvmlReturn_t;

    async fn nvmlInit_v2() -> nvmlReturn_t;
    async fn nvmlShutdown() -> nvmlReturn_t;

    // 4.12 Error reporting
    async fn nvmlErrorString(result: nvmlReturn_t) -> String;

    // 4.14 System Queries
    async fn nvmlSystemGetCudaDriverVersion() -> (nvmlReturn_t, i32);
    async fn nvmlSystemGetCudaDriverVersion_v2() -> (nvmlReturn_t, i32);
    async fn nvmlSystemGetDriverVersion(length: u32) -> (nvmlReturn_t, String);
    async fn nvmlSystemGetNVMLVersion(length: u32) -> (nvmlReturn_t, String);
    async fn nvmlSystemGetProcessName(pid: u32, length: u32) -> (nvmlReturn_t, String);

    async fn nvml_device_get_name(device: u64, length: u32) -> String;

    async fn nvml_system_get_cuda_driver_version_v2() -> i32;

    // 4.16 Resource requests
    async fn nvmlDeviceGetFanSpeed(device: u64, speed: u32) -> nvmlReturn_t;
}
