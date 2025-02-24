#ifndef RPC_H
#define RPC_H

#include <pthread.h>
#include <sys/uio.h>

typedef struct {
  int connfd;

  int request_id;
  int read_id;
  int write_id;
  int write_op;

  pthread_t read_thread;
  pthread_mutex_t read_mutex, write_mutex;
  pthread_cond_t read_cond;
  struct iovec write_iov[128];
  int write_iov_count = 0;
} conn_t;

extern int rpc_dispatch(conn_t *conn, int parity);
extern int rpc_read_start(conn_t *conn, int write_id);
extern int rpc_read(conn_t *conn, void *data, size_t size);
extern int rpc_read_end(conn_t *conn);

extern int rpc_wait_for_response(conn_t *conn);

extern int rpc_write_start_request(conn_t *conn, const int op);
extern int rpc_write_start_response(conn_t *conn, const int read_id);
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_end(conn_t *conn);

#endif