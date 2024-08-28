use futures::StreamExt;
use scuda_api::Scuda;
use std::{
    future::{self, Future},
    net::{IpAddr, Ipv6Addr, SocketAddr},
};

use libc::{c_char, c_int};
use nvml_wrapper_sys::bindings::{nvmlDevice_t, nvmlReturn_enum_NVML_SUCCESS, NvmlLib};
use tarpc::{
    context,
    server::{self, incoming::Incoming, Channel},
    tokio_serde::formats::Json,
};

#[derive(Clone)]
pub struct ScudaServer(pub SocketAddr);

impl scuda_api::Scuda for ScudaServer {
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

async fn spawn(fut: impl Future<Output = ()> + Send + 'static) {
    tokio::spawn(fut);
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server_addr = (IpAddr::V6(Ipv6Addr::LOCALHOST), 31337);
    let mut listener = tarpc::serde_transport::tcp::listen(&server_addr, Json::default).await?;
    listener.config_mut().max_frame_length(usize::MAX);
    listener
        .filter_map(|r| future::ready(r.ok()))
        .map(server::BaseChannel::with_defaults)1
        .max_channels_per_key(1, |t| t.transport().peer_addr().unwrap().ip())
        .map(|channel| {
            let server = ScudaServer(channel.transport().peer_addr().unwrap());
            channel.execute(server.serve()).for_each(spawn)
        })
        .buffer_unordered(10)
        .for_each(|_| async {})
        .await;

    Ok(())
}
