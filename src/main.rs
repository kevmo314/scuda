use futures::StreamExt;
use scuda::{
    api::Scuda,
    server::{spawn, ScudaServer},
};
use std::{
    future::{self},
    net::{IpAddr, Ipv6Addr},
};

use tarpc::{
    server::{self, incoming::Incoming, Channel},
    tokio_serde::formats::Json,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let server_addr = (IpAddr::V6(Ipv6Addr::LOCALHOST), 31337);
    let mut listener = tarpc::serde_transport::tcp::listen(&server_addr, Json::default).await?;
    listener.config_mut().max_frame_length(usize::MAX);
    listener
        .filter_map(|r| future::ready(r.ok()))
        .map(server::BaseChannel::with_defaults)
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
