#[tarpc::service]
pub trait Scuda {
    async fn nvml_device_get_name(device: u64, length: u32) -> String;

    async fn nvml_system_get_cuda_driver_version_v2() -> i32;
}
