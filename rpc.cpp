#include <sys/socket.h>

#include "rpc.h"
#include <algorithm>
#include <deque>
#include <errno.h>
#include <iostream>
#include <nghttp2/nghttp2.h>
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

constexpr uint32_t kH2InitialWindow = 0x7fffffffU;
constexpr uint32_t kH2MaxFrame = (16 * 1024 * 1024) - 1;
constexpr size_t kH2FrameHeaderLen = 9;

struct h2_buffer {
  std::vector<unsigned char> data;
  size_t offset = 0;
};

struct h2_transport {
  int netfd = -1;
  bool server = false;
  bool response_sent = false;
  bool redirect = false;
  int32_t stream_id = 1;
  int response_status = 0;
  std::string redirect_location;
  nghttp2_session *session = nullptr;
  std::deque<h2_buffer> local_out;
  size_t queued_local_bytes = 0;
  pthread_mutex_t send_mutex = PTHREAD_MUTEX_INITIALIZER;
};

struct h2_write_cursor {
  const unsigned char *base = nullptr;
  size_t len = 0;
};

struct h2_write_source {
  std::vector<h2_write_cursor> cursors;
  size_t index = 0;
  size_t pending_len = 0;

  size_t remaining() const {
    size_t total = 0;
    for (size_t i = index; i < cursors.size(); ++i) {
      total += cursors[i].len;
    }
    return total;
  }
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

int h2_write_all_locked(h2_transport *transport, const struct iovec *iov,
                        int iov_count) {
  std::vector<struct iovec> local(iov, iov + iov_count);
  struct iovec *cursor = local.data();
  int count = iov_count;
  while (count > 0) {
    ssize_t n = writev(transport->netfd, cursor, count);
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

ssize_t h2_send_callback(nghttp2_session *, const uint8_t *data, size_t length,
                         int, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  struct iovec iov = {const_cast<uint8_t *>(data), length};
  return h2_write_all_locked(transport, &iov, 1) == 0
             ? static_cast<ssize_t>(length)
             : NGHTTP2_ERR_CALLBACK_FAILURE;
}

ssize_t h2_data_source_read_callback(nghttp2_session *, int32_t, uint8_t *,
                                     size_t length, uint32_t *data_flags,
                                     nghttp2_data_source *source, void *) {
  auto *write_source = static_cast<h2_write_source *>(source->ptr);
  size_t remaining = write_source->remaining();
  if (remaining == 0) {
    *data_flags |= NGHTTP2_DATA_FLAG_EOF | NGHTTP2_DATA_FLAG_NO_END_STREAM;
    write_source->pending_len = 0;
    return 0;
  }
  size_t chunk = std::min(remaining, length);
  *data_flags |= NGHTTP2_DATA_FLAG_NO_COPY;
  if (chunk == remaining) {
    *data_flags |= NGHTTP2_DATA_FLAG_EOF | NGHTTP2_DATA_FLAG_NO_END_STREAM;
  }
  write_source->pending_len = chunk;
  return static_cast<ssize_t>(chunk);
}

ssize_t h2_data_source_read_length_callback(nghttp2_session *, uint8_t, int32_t,
                                            int32_t session_remote_window_size,
                                            int32_t stream_remote_window_size,
                                            uint32_t remote_max_frame_size,
                                            void *) {
  int32_t window =
      std::min(session_remote_window_size, stream_remote_window_size);
  if (window <= 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }
  size_t max_len = std::min<size_t>(kH2MaxFrame, remote_max_frame_size);
  max_len = std::min<size_t>(max_len, static_cast<size_t>(window));
  return static_cast<ssize_t>(std::max<size_t>(1, max_len));
}

int h2_send_data_callback(nghttp2_session *, nghttp2_frame *frame,
                          const uint8_t *framehd, size_t length,
                          nghttp2_data_source *source, void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  auto *write_source = static_cast<h2_write_source *>(source->ptr);
  if (length != write_source->pending_len) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  std::vector<struct iovec> iov;
  iov.reserve(write_source->cursors.size() - write_source->index + 3);
  iov.push_back({const_cast<uint8_t *>(framehd), kH2FrameHeaderLen});

  unsigned char padlen = 0;
  if (frame->data.padlen > 0) {
    padlen = static_cast<unsigned char>(frame->data.padlen - 1);
    iov.push_back({&padlen, 1});
  }

  size_t remaining = length;
  size_t cursor_index = write_source->index;
  while (remaining > 0 && cursor_index < write_source->cursors.size()) {
    auto &cursor = write_source->cursors[cursor_index];
    size_t chunk = std::min(remaining, cursor.len);
    iov.push_back({const_cast<unsigned char *>(cursor.base), chunk});
    remaining -= chunk;
    ++cursor_index;
  }
  if (remaining != 0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  unsigned char padding[256] = {};
  if (frame->data.padlen > 1) {
    iov.push_back({padding, frame->data.padlen - 1});
  }

  if (h2_write_all_locked(transport, iov.data(), static_cast<int>(iov.size())) <
      0) {
    return NGHTTP2_ERR_CALLBACK_FAILURE;
  }

  remaining = length;
  while (remaining > 0) {
    auto &cursor = write_source->cursors[write_source->index];
    size_t chunk = std::min(remaining, cursor.len);
    cursor.base += chunk;
    cursor.len -= chunk;
    remaining -= chunk;
    if (cursor.len == 0) {
      ++write_source->index;
    }
  }
  write_source->pending_len = 0;
  return 0;
}

int h2_on_data_chunk_recv_callback(nghttp2_session *, uint8_t, int32_t,
                                   const uint8_t *data, size_t len,
                                   void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  queue_bytes(transport->local_out, transport->queued_local_bytes, data, len);
  return 0;
}

nghttp2_nv h2_nv(const char *name, const char *value) {
  return {reinterpret_cast<uint8_t *>(const_cast<char *>(name)),
          reinterpret_cast<uint8_t *>(const_cast<char *>(value)), strlen(name),
          strlen(value),
          NGHTTP2_NV_FLAG_NO_COPY_NAME | NGHTTP2_NV_FLAG_NO_COPY_VALUE};
}

int h2_submit_server_response(h2_transport *transport) {
  nghttp2_nv headers[] = {h2_nv(":status", "200")};
  if (nghttp2_submit_headers(transport->session, NGHTTP2_FLAG_NONE,
                             transport->stream_id, nullptr, headers, 1,
                             nullptr) != 0) {
    return -1;
  }
  transport->response_sent = true;
  return 0;
}

int h2_on_frame_recv_callback(nghttp2_session *, const nghttp2_frame *frame,
                              void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  if (transport->server && frame->hd.type == NGHTTP2_HEADERS &&
      frame->headers.cat == NGHTTP2_HCAT_REQUEST) {
    transport->stream_id = frame->hd.stream_id;
    if (!transport->response_sent && h2_submit_server_response(transport) < 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  return 0;
}

int h2_on_header_callback(nghttp2_session *, const nghttp2_frame *frame,
                          const uint8_t *name, size_t namelen,
                          const uint8_t *value, size_t valuelen, uint8_t,
                          void *user_data) {
  auto *transport = static_cast<h2_transport *>(user_data);
  if (transport->server || frame->hd.type != NGHTTP2_HEADERS ||
      frame->headers.cat != NGHTTP2_HCAT_RESPONSE) {
    return 0;
  }
  if (namelen == 7 && memcmp(name, ":status", 7) == 0) {
    transport->response_status = 0;
    for (size_t i = 0; i < valuelen; ++i) {
      if (value[i] < '0' || value[i] > '9') {
        return 0;
      }
      transport->response_status =
          transport->response_status * 10 + static_cast<int>(value[i] - '0');
    }
    transport->redirect =
        transport->response_status >= 300 && transport->response_status < 400;
  } else if (namelen == 8 && memcmp(name, "location", 8) == 0) {
    transport->redirect_location.assign(reinterpret_cast<const char *>(value),
                                        valuelen);
  }
  return 0;
}

int h2_flush_session(h2_transport *transport) {
  pthread_mutex_lock(&transport->send_mutex);
  int result = nghttp2_session_send(transport->session);
  pthread_mutex_unlock(&transport->send_mutex);
  return result == 0 ? 0 : -1;
}

int h2_read_from_net(h2_transport *transport) {
  unsigned char buffer[64 * 1024];
  ssize_t n = 0;
  do {
    n = read(transport->netfd, buffer, sizeof(buffer));
  } while (n < 0 && errno == EINTR);
  if (n <= 0) {
    return -1;
  }

  size_t offset = 0;
  while (offset < static_cast<size_t>(n)) {
    ssize_t consumed = nghttp2_session_mem_recv(
        transport->session, buffer + offset, static_cast<size_t>(n) - offset);
    if (consumed <= 0) {
      return -1;
    }
    offset += static_cast<size_t>(consumed);
  }
  return h2_flush_session(transport);
}

} // namespace

int rpc_http2_read(conn_t *conn, void *data, size_t size) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  auto *out = static_cast<unsigned char *>(data);
  size_t copied = 0;
  while (copied < size) {
    while (!transport->local_out.empty() && copied < size) {
      h2_buffer &front = transport->local_out.front();
      size_t available = front.data.size() - front.offset;
      size_t chunk = std::min(available, size - copied);
      memcpy(out + copied, front.data.data() + front.offset, chunk);
      front.offset += chunk;
      transport->queued_local_bytes -= chunk;
      copied += chunk;
      if (front.offset == front.data.size()) {
        transport->local_out.pop_front();
      }
    }
    if (copied == size) {
      return static_cast<int>(size);
    }
    if ((transport->response_status != 0 &&
         transport->response_status != 200) ||
        h2_read_from_net(transport) < 0) {
      return -1;
    }
  }
  return static_cast<int>(size);
}

int rpc_http2_writev(conn_t *conn, struct iovec *iov, int iov_count) {
  auto *transport = static_cast<h2_transport *>(conn->http2);
  h2_write_source source;
  source.cursors.reserve(iov_count);
  for (int i = 0; i < iov_count; ++i) {
    if (iov[i].iov_len == 0) {
      continue;
    }
    source.cursors.push_back(
        {static_cast<const unsigned char *>(iov[i].iov_base), iov[i].iov_len});
  }
  if (source.cursors.empty()) {
    return 0;
  }

  nghttp2_data_provider provider = {};
  provider.source.ptr = &source;
  provider.read_callback = h2_data_source_read_callback;
  if (nghttp2_submit_data(transport->session, NGHTTP2_FLAG_NONE,
                          transport->stream_id, &provider) != 0 ||
      h2_flush_session(transport) < 0 || source.remaining() != 0) {
    return -1;
  }
  return 0;
}

static int h2_init_direct(conn_t *conn, bool server) {
  auto *transport = new h2_transport();
  transport->netfd = conn->connfd;
  transport->server = server;

  nghttp2_session_callbacks *callbacks = nullptr;
  if (nghttp2_session_callbacks_new(&callbacks) != 0) {
    delete transport;
    return -1;
  }
  nghttp2_session_callbacks_set_send_callback(callbacks, h2_send_callback);
  nghttp2_session_callbacks_set_send_data_callback(callbacks,
                                                   h2_send_data_callback);
  nghttp2_session_callbacks_set_data_source_read_length_callback(
      callbacks, h2_data_source_read_length_callback);
  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(
      callbacks, h2_on_data_chunk_recv_callback);
  nghttp2_session_callbacks_set_on_frame_recv_callback(
      callbacks, h2_on_frame_recv_callback);
  nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                   h2_on_header_callback);

  int session_result = server ? nghttp2_session_server_new(&transport->session,
                                                           callbacks, transport)
                              : nghttp2_session_client_new(
                                    &transport->session, callbacks, transport);
  nghttp2_session_callbacks_del(callbacks);
  if (session_result != 0) {
    delete transport;
    return -1;
  }

  nghttp2_settings_entry settings[] = {
      {NGHTTP2_SETTINGS_INITIAL_WINDOW_SIZE, kH2InitialWindow},
      {NGHTTP2_SETTINGS_MAX_FRAME_SIZE, kH2MaxFrame},
  };
  if (nghttp2_submit_settings(transport->session, NGHTTP2_FLAG_NONE, settings,
                              2) != 0 ||
      nghttp2_session_set_local_window_size(
          transport->session, NGHTTP2_FLAG_NONE, 0,
          static_cast<int32_t>(kH2InitialWindow)) != 0) {
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }

  if (!server) {
    nghttp2_nv headers[] = {
        h2_nv(":method", "POST"),
        h2_nv(":scheme", "http"),
        h2_nv(":path", "/"),
        h2_nv(":authority", "lupine"),
    };
    int32_t stream_id =
        nghttp2_submit_headers(transport->session, NGHTTP2_FLAG_NONE, -1,
                               nullptr, headers, 4, nullptr);
    if (stream_id < 0) {
      nghttp2_session_del(transport->session);
      delete transport;
      return -1;
    }
    transport->stream_id = stream_id;
  }

  conn->http2 = transport;
  if (h2_flush_session(transport) < 0) {
    conn->http2 = nullptr;
    nghttp2_session_del(transport->session);
    delete transport;
    return -1;
  }
  return 0;
}

int rpc_http2_client_init(conn_t *conn) { return h2_init_direct(conn, false); }

int rpc_http2_server_init(conn_t *conn) { return h2_init_direct(conn, true); }
