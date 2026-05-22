#include <sys/socket.h>

#include "rpc.h"
#include <algorithm>
#include <deque>
#include <errno.h>
#include <iostream>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <vector>


void *_rpc_read_id_dispatch(void *p) {
  conn_t *conn = (conn_t *)p;

  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return NULL;

  while (!conn->closed) {
    while (conn->read_id != 0 && !conn->closed)
      pthread_cond_wait(&conn->read_cond, &conn->read_mutex);
    if (conn->closed)
      break;

    // the read id is zero so it's our turn to read the next int which is the
    // request id of the next request.
    int bytes = rpc_read(conn, &conn->read_id, sizeof(int));
    if (bytes <= 0 || conn->read_id == 0) {
      conn->closed = 1;
      pthread_cond_broadcast(&conn->read_cond);
      break;
    }
    if (pthread_cond_broadcast(&conn->read_cond) < 0)
      break;
  }
  pthread_mutex_unlock(&conn->read_mutex);
  conn->rpc_thread = 0;
  return NULL;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a
// request that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on
// entry.
int rpc_dispatch(conn_t *conn, int parity) {
  if (conn->rpc_thread == 0 &&
      pthread_create(&conn->rpc_thread, nullptr, _rpc_read_id_dispatch, (void *)conn) < 0) {
    return -1;
  }

  if (pthread_mutex_lock(&conn->read_mutex) < 0) {
    return -1;
  }

  int op;

  while (!conn->closed && (conn->read_id < 2 || conn->read_id % 2 != parity))
    pthread_cond_wait(&conn->read_cond, &conn->read_mutex);

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  if (rpc_read(conn, &op, sizeof(int)) < 0) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  return op;
}

// rpc_read_start waits for a response with a specific request id on the
// given connection. this function is used to wait for a response to a request
// that was sent with rpc_write_end.
//
// it is not necessary to call rpc_read_start() if it is the first call in
// the sequence because by convention, the handler owns the read lock on entry.
int rpc_read_start(conn_t *conn, int write_id) {
  if (pthread_mutex_lock(&conn->read_mutex) < 0)
    return -1;

  // wait for the active read id to be the request id we are waiting for
  while (!conn->closed && conn->read_id != write_id)
    if (pthread_cond_wait(&conn->read_cond, &conn->read_mutex) < 0)
      return -1;

  if (conn->closed) {
    pthread_mutex_unlock(&conn->read_mutex);
    return -1;
  }

  return 0;
}

int rpc_read(conn_t *conn, void *data, size_t size) {
  if (conn->http2 != nullptr) {
    extern int rpc_http2_read(conn_t *, void *, size_t);
    return rpc_http2_read(conn, data, size);
  }
  char *cursor = static_cast<char *>(data);
  size_t total = 0;
  while (total < size) {
    ssize_t bytes_read =
        recv(conn->connfd, cursor + total, size - total, MSG_WAITALL);
    if (bytes_read == 0) {
      return total == 0 ? 0 : -1;
    }
    if (bytes_read < 0) {
      if (errno == EINTR) {
        continue;
      }
      if (errno != EBADF && errno != ECONNRESET) {
        printf("recv error: %s\n", strerror(errno));
      }
      return -1;
    }
    total += static_cast<size_t>(bytes_read);
  }
  return static_cast<int>(total);
}

int rpc_drain(conn_t *conn, size_t size) {
  char buffer[64 * 1024];
  size_t offset = 0;
  while (offset < size) {
    size_t chunk = std::min(sizeof(buffer), size - offset);
    if (rpc_read(conn, buffer, chunk) < 0) {
      return -1;
    }
    offset += chunk;
  }
  return 0;
}

// rpc_read_end releases the response lock on the given connection.
int rpc_read_end(conn_t *conn) {
  int read_id = conn->read_id;
  bool completes_local_request =
      read_id >= 2 && (read_id % 2) == conn->local_request_parity;
  conn->read_id = 0;
  if (pthread_cond_broadcast(&conn->read_cond) < 0 ||
      pthread_mutex_unlock(&conn->read_mutex) < 0)
    return -1;
  if (completes_local_request &&
      pthread_mutex_unlock(&conn->call_mutex) < 0)
    return -1;
  return read_id;
}

// rpc_wait_for_response is a convenience function that sends the current
// request and then waits for the corresponding response. this pattern is
// so common that having this function keeps the codegen much cleaner.
int rpc_wait_for_response(conn_t *conn) {
  int write_id = rpc_write_end(conn);
  if (write_id < 0 || rpc_read_start(conn, write_id) < 0) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  return 0;
}

// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_request(conn_t *conn, const int op) {
  if (conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->call_mutex) < 0) {
    return -1;
  }
  if (conn->closed) {
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    pthread_mutex_unlock(&conn->call_mutex);
    return -1;
  }

  conn->write_iov_count = 2;               // skip 2 for the header
  conn->request_id = conn->request_id + 2; // leave the last bit the same
  conn->write_id = conn->request_id;
  conn->write_op = op;
  return 0;
}
// rpc_write_start_request starts a new request builder on the given connection
// index with a specific op code.
//
// only one request can be active at a time, so this function will take the
// request lock from the connection.
int rpc_write_start_response(conn_t *conn, const int read_id) {
  if (conn->closed) {
    return -1;
  }
  if (pthread_mutex_lock(&conn->write_mutex) < 0) {
#ifdef VERBOSE
    std::cout << "rpc_write_start failed due to rpc_open() < 0 || "
                 "conns[index].write_mutex lock"
              << std::endl;
#endif
    return -1;
  }

  conn->write_iov_count = 1; // skip 1 for the header
  conn->write_id = read_id;
  conn->write_op = -1;
  return 0;
}

int rpc_write(conn_t *conn, const void *data, const size_t size) {
  conn->write_iov[conn->write_iov_count++] = (struct iovec){(void *)data, size};
  return 0;
}

// rpc_write_end finalizes the current request builder on the given connection
// index and sends the request to the server.
//
// the request lock is released after the request is sent and the function
// returns the request id which can be used to wait for a response.
int rpc_write_end(conn_t *conn) {
  if (conn->closed) {
    pthread_mutex_unlock(&conn->write_mutex);
    return -1;
  }
  conn->write_iov[0] = {&conn->write_id, sizeof(int)};
  if (conn->write_op != -1) {
    conn->write_iov[1] = {&conn->write_op, sizeof(unsigned int)};
  }

  if (conn->http2 != nullptr) {
    extern int rpc_http2_writev(conn_t *, struct iovec *, int);
    int result = rpc_http2_writev(conn, conn->write_iov, conn->write_iov_count);
    pthread_mutex_unlock(&conn->write_mutex);
    return result == 0 ? conn->write_id : -1;
  }

  struct iovec iov[128];
  int iov_count = conn->write_iov_count;
  memcpy(iov, conn->write_iov, sizeof(struct iovec) * iov_count);
  struct iovec *iov_cursor = iov;

  while (iov_count > 0) {
    ssize_t written = writev(conn->connfd, iov_cursor, iov_count);
    if (written < 0) {
      if (errno == EINTR) {
        continue;
      }
      pthread_mutex_unlock(&conn->write_mutex);
      return -1;
    }
    if (written == 0) {
      pthread_mutex_unlock(&conn->write_mutex);
      return -1;
    }

    size_t remaining = static_cast<size_t>(written);
    while (iov_count > 0 && remaining >= iov_cursor[0].iov_len) {
      remaining -= iov_cursor[0].iov_len;
      ++iov_cursor;
      --iov_count;
    }
    if (iov_count > 0 && remaining != 0) {
      iov_cursor[0].iov_base =
          static_cast<char *>(iov_cursor[0].iov_base) + remaining;
      iov_cursor[0].iov_len -= remaining;
    }
  }

  if (pthread_mutex_unlock(&conn->write_mutex) < 0)
    return -1;
  return conn->write_id;
}

namespace {

constexpr const char kH2ClientPreface[] = "PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
constexpr size_t kH2PrefaceLen = 24;
constexpr uint32_t kH2InitialWindow = 0x7fffffffU;
constexpr uint32_t kH2MaxFrame = (16 * 1024 * 1024) - 1;
constexpr size_t kH2WindowUpdateThreshold = 64 * 1024 * 1024;

enum {
  H2_DATA = 0x0,
  H2_HEADERS = 0x1,
  H2_SETTINGS = 0x4,
  H2_WINDOW_UPDATE = 0x8,
};

enum {
  H2_FLAG_END_STREAM = 0x1,
  H2_FLAG_END_HEADERS = 0x4,
  H2_FLAG_ACK = 0x1,
};

enum {
  H2_SETTINGS_INITIAL_WINDOW_SIZE = 0x4,
  H2_SETTINGS_MAX_FRAME_SIZE = 0x5,
};

struct h2_buffer {
  std::vector<unsigned char> data;
  size_t offset = 0;
};

struct h2_tunnel {
  int netfd = -1;
  bool server = false;
  bool got_preface = false;
  bool response_sent = false;
  int32_t stream_id = 1;
  uint32_t peer_max_frame = 16384;
  std::deque<h2_buffer> local_out;
  size_t queued_local_bytes = 0;
  size_t pending_conn_window_update = 0;
  size_t pending_stream_window_update = 0;
  pthread_mutex_t send_mutex = PTHREAD_MUTEX_INITIALIZER;
};

void queue_bytes(std::deque<h2_buffer> &queue, size_t &queued,
                 const unsigned char *data, size_t len) {
  if (len == 0) {
    return;
  }
  h2_buffer buffer;
  buffer.data.assign(data, data + len);
  queued += len;
  queue.push_back(std::move(buffer));
}

uint32_t read_u24(const unsigned char *p) {
  return (static_cast<uint32_t>(p[0]) << 16) |
         (static_cast<uint32_t>(p[1]) << 8) | static_cast<uint32_t>(p[2]);
}

uint32_t read_u32(const unsigned char *p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

} // namespace

static int h2_write_all_locked(h2_tunnel *tunnel, const struct iovec *iov,
                               int iov_count) {
  std::vector<struct iovec> local(iov, iov + iov_count);
  struct iovec *cursor = local.data();
  int count = iov_count;
  while (count > 0) {
    ssize_t n = writev(tunnel->netfd, cursor, count);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (n == 0) {
      return -1;
    }
    size_t written = static_cast<size_t>(n);
    while (count > 0 && written >= cursor[0].iov_len) {
      written -= cursor[0].iov_len;
      ++cursor;
      --count;
    }
    if (count > 0 && written != 0) {
      cursor[0].iov_base = static_cast<char *>(cursor[0].iov_base) + written;
      cursor[0].iov_len -= written;
    }
  }
  return 0;
}

static int h2_send_frame_direct(h2_tunnel *tunnel, uint8_t type, uint8_t flags,
                                int32_t stream_id, const void *payload,
                                size_t payload_len) {
  unsigned char header[9];
  header[0] = static_cast<unsigned char>((payload_len >> 16) & 0xff);
  header[1] = static_cast<unsigned char>((payload_len >> 8) & 0xff);
  header[2] = static_cast<unsigned char>(payload_len & 0xff);
  header[3] = type;
  header[4] = flags;
  uint32_t sid = static_cast<uint32_t>(stream_id) & 0x7fffffffU;
  header[5] = static_cast<unsigned char>((sid >> 24) & 0xff);
  header[6] = static_cast<unsigned char>((sid >> 16) & 0xff);
  header[7] = static_cast<unsigned char>((sid >> 8) & 0xff);
  header[8] = static_cast<unsigned char>(sid & 0xff);

  struct iovec iov[2];
  iov[0] = {header, sizeof(header)};
  iov[1] = {const_cast<void *>(payload), payload_len};
  pthread_mutex_lock(&tunnel->send_mutex);
  int result = h2_write_all_locked(tunnel, iov, payload_len == 0 ? 1 : 2);
  pthread_mutex_unlock(&tunnel->send_mutex);
  return result;
}

static int h2_send_settings_direct(h2_tunnel *tunnel) {
  unsigned char payload[12];
  payload[0] = 0;
  payload[1] = H2_SETTINGS_INITIAL_WINDOW_SIZE;
  payload[2] = static_cast<unsigned char>((kH2InitialWindow >> 24) & 0xff);
  payload[3] = static_cast<unsigned char>((kH2InitialWindow >> 16) & 0xff);
  payload[4] = static_cast<unsigned char>((kH2InitialWindow >> 8) & 0xff);
  payload[5] = static_cast<unsigned char>(kH2InitialWindow & 0xff);
  payload[6] = 0;
  payload[7] = H2_SETTINGS_MAX_FRAME_SIZE;
  payload[8] = static_cast<unsigned char>((kH2MaxFrame >> 24) & 0xff);
  payload[9] = static_cast<unsigned char>((kH2MaxFrame >> 16) & 0xff);
  payload[10] = static_cast<unsigned char>((kH2MaxFrame >> 8) & 0xff);
  payload[11] = static_cast<unsigned char>(kH2MaxFrame & 0xff);
  return h2_send_frame_direct(tunnel, H2_SETTINGS, 0, 0, payload,
                              sizeof(payload));
}

static int h2_send_window_direct(h2_tunnel *tunnel, int32_t stream_id,
                                 uint32_t inc) {
  unsigned char payload[4];
  payload[0] = static_cast<unsigned char>((inc >> 24) & 0xff);
  payload[1] = static_cast<unsigned char>((inc >> 16) & 0xff);
  payload[2] = static_cast<unsigned char>((inc >> 8) & 0xff);
  payload[3] = static_cast<unsigned char>(inc & 0xff);
  return h2_send_frame_direct(tunnel, H2_WINDOW_UPDATE, 0, stream_id, payload,
                              sizeof(payload));
}

static int h2_send_client_headers_direct(h2_tunnel *tunnel) {
  unsigned char block[] = {
      0x83, // :method: POST
      0x86, // :scheme: http
      0x84, // :path: /
      0x01, 0x06, 'l', 'u', 'p', 'i', 'n', 'e', // :authority
  };
  return h2_send_frame_direct(tunnel, H2_HEADERS, H2_FLAG_END_HEADERS,
                              tunnel->stream_id, block, sizeof(block));
}

static int h2_send_server_headers_direct(h2_tunnel *tunnel) {
  unsigned char block[] = {0x88}; // :status: 200
  tunnel->response_sent = true;
  return h2_send_frame_direct(tunnel, H2_HEADERS, H2_FLAG_END_HEADERS,
                              tunnel->stream_id, block, sizeof(block));
}

static int h2_read_exact(int fd, void *data, size_t size) {
  char *cursor = static_cast<char *>(data);
  size_t total = 0;
  while (total < size) {
    ssize_t n = read(fd, cursor + total, size - total);
    if (n < 0) {
      if (errno == EINTR) {
        continue;
      }
      return -1;
    }
    if (n == 0) {
      return -1;
    }
    total += static_cast<size_t>(n);
  }
  return 0;
}

static int h2_process_frame_direct(h2_tunnel *tunnel, uint8_t type,
                                   uint8_t flags, int32_t stream_id,
                                   const unsigned char *payload,
                                   size_t payload_len) {
  if (type == H2_SETTINGS) {
    if ((flags & H2_FLAG_ACK) == 0) {
      for (size_t offset = 0; offset + 6 <= payload_len; offset += 6) {
        uint16_t id = (static_cast<uint16_t>(payload[offset]) << 8) |
                      static_cast<uint16_t>(payload[offset + 1]);
        uint32_t value = read_u32(payload + offset + 2);
        if (id == H2_SETTINGS_MAX_FRAME_SIZE) {
          tunnel->peer_max_frame = std::min<uint32_t>(value, kH2MaxFrame);
        }
      }
      return h2_send_frame_direct(tunnel, H2_SETTINGS, H2_FLAG_ACK, 0, nullptr,
                                  0);
    }
    return 0;
  }
  if (type == H2_HEADERS) {
    if (tunnel->server) {
      tunnel->stream_id = stream_id;
      if (!tunnel->response_sent) {
        return h2_send_server_headers_direct(tunnel);
      }
    }
    return 0;
  }
  if (type == H2_DATA) {
    queue_bytes(tunnel->local_out, tunnel->queued_local_bytes, payload,
                payload_len);
    tunnel->pending_conn_window_update += payload_len;
    tunnel->pending_stream_window_update += payload_len;
    if (tunnel->pending_conn_window_update >= kH2WindowUpdateThreshold) {
      if (h2_send_window_direct(
              tunnel, 0,
              static_cast<uint32_t>(tunnel->pending_conn_window_update)) < 0) {
        return -1;
      }
      tunnel->pending_conn_window_update = 0;
    }
    if (tunnel->pending_stream_window_update >= kH2WindowUpdateThreshold) {
      if (h2_send_window_direct(
              tunnel, stream_id,
              static_cast<uint32_t>(tunnel->pending_stream_window_update)) <
          0) {
        return -1;
      }
      tunnel->pending_stream_window_update = 0;
    }
    return 0;
  }
  return 0;
}

int rpc_http2_read(conn_t *conn, void *data, size_t size) {
  auto *tunnel = static_cast<h2_tunnel *>(conn->http2);
  auto *out = static_cast<unsigned char *>(data);
  size_t copied = 0;
  while (copied < size) {
    while (!tunnel->local_out.empty() && copied < size) {
      h2_buffer &front = tunnel->local_out.front();
      size_t available = front.data.size() - front.offset;
      size_t chunk = std::min(available, size - copied);
      memcpy(out + copied, front.data.data() + front.offset, chunk);
      front.offset += chunk;
      tunnel->queued_local_bytes -= chunk;
      copied += chunk;
      if (front.offset == front.data.size()) {
        tunnel->local_out.pop_front();
      }
    }
    if (copied == size) {
      return static_cast<int>(size);
    }

    if (tunnel->server && !tunnel->got_preface) {
      char preface[kH2PrefaceLen];
      if (h2_read_exact(tunnel->netfd, preface, sizeof(preface)) < 0 ||
          memcmp(preface, kH2ClientPreface, sizeof(preface)) != 0) {
        return -1;
      }
      tunnel->got_preface = true;
    }

    unsigned char header[9];
    if (h2_read_exact(tunnel->netfd, header, sizeof(header)) < 0) {
      return -1;
    }
    uint32_t len = read_u24(header);
    if (len > kH2MaxFrame) {
      return -1;
    }
    std::vector<unsigned char> payload(len);
    if (len != 0 && h2_read_exact(tunnel->netfd, payload.data(), len) < 0) {
      return -1;
    }
    uint8_t type = header[3];
    uint8_t flags = header[4];
    int32_t stream_id =
        static_cast<int32_t>(read_u32(header + 5) & 0x7fffffffU);
    if (h2_process_frame_direct(tunnel, type, flags, stream_id, payload.data(),
                                payload.size()) < 0) {
      return -1;
    }
  }
  return static_cast<int>(size);
}

int rpc_http2_writev(conn_t *conn, struct iovec *iov, int iov_count) {
  auto *tunnel = static_cast<h2_tunnel *>(conn->http2);
  struct cursor {
    const unsigned char *base = nullptr;
    size_t len = 0;
  };
  std::vector<cursor> cursors;
  cursors.reserve(iov_count);
  for (int i = 0; i < iov_count; ++i) {
    if (iov[i].iov_len == 0) {
      continue;
    }
    cursors.push_back(
        {static_cast<const unsigned char *>(iov[i].iov_base), iov[i].iov_len});
  }

  size_t index = 0;
  while (index < cursors.size()) {
    size_t frame_len = 0;
    size_t end = index;
    while (end < cursors.size() &&
           frame_len + cursors[end].len <= tunnel->peer_max_frame) {
      frame_len += cursors[end].len;
      ++end;
    }
    if (frame_len == 0) {
      frame_len = tunnel->peer_max_frame;
    }

    unsigned char header[9];
    header[0] = static_cast<unsigned char>((frame_len >> 16) & 0xff);
    header[1] = static_cast<unsigned char>((frame_len >> 8) & 0xff);
    header[2] = static_cast<unsigned char>(frame_len & 0xff);
    header[3] = H2_DATA;
    header[4] = 0;
    uint32_t sid = static_cast<uint32_t>(tunnel->stream_id) & 0x7fffffffU;
    header[5] = static_cast<unsigned char>((sid >> 24) & 0xff);
    header[6] = static_cast<unsigned char>((sid >> 16) & 0xff);
    header[7] = static_cast<unsigned char>((sid >> 8) & 0xff);
    header[8] = static_cast<unsigned char>(sid & 0xff);

    std::vector<struct iovec> out;
    out.reserve((end - index) + 2);
    out.push_back({header, sizeof(header)});
    size_t remaining = frame_len;
    while (remaining > 0) {
      size_t chunk = std::min(remaining, cursors[index].len);
      out.push_back({const_cast<unsigned char *>(cursors[index].base), chunk});
      cursors[index].base += chunk;
      cursors[index].len -= chunk;
      remaining -= chunk;
      if (cursors[index].len == 0) {
        ++index;
      }
    }

    pthread_mutex_lock(&tunnel->send_mutex);
    int result =
        h2_write_all_locked(tunnel, out.data(), static_cast<int>(out.size()));
    pthread_mutex_unlock(&tunnel->send_mutex);
    if (result < 0) {
        return -1;
    }
  }
  return 0;
}

static int h2_init_direct(conn_t *conn, bool server) {
  auto *tunnel = new h2_tunnel();
  tunnel->netfd = conn->connfd;
  tunnel->server = server;
  tunnel->got_preface = !server;
  tunnel->peer_max_frame = kH2MaxFrame;
  conn->http2 = tunnel;

  if (!server) {
    struct iovec preface_iov = {
        const_cast<char *>(kH2ClientPreface), kH2PrefaceLen};
    pthread_mutex_lock(&tunnel->send_mutex);
    int preface_result = h2_write_all_locked(tunnel, &preface_iov, 1);
    pthread_mutex_unlock(&tunnel->send_mutex);
    if (preface_result < 0) {
      return -1;
    }
  }
  if (h2_send_settings_direct(tunnel) < 0 ||
      h2_send_window_direct(tunnel, 0, kH2InitialWindow - 65535U) < 0) {
    return -1;
  }
  if (!server && h2_send_client_headers_direct(tunnel) < 0) {
    return -1;
  }
  return 0;
}

int rpc_http2_client_init(conn_t *conn) { return h2_init_direct(conn, false); }

int rpc_http2_server_init(conn_t *conn) { return h2_init_direct(conn, true); }
