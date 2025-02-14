#include <stddef.h>

struct conn_t;

extern int rpc_read_start(conn_t *conn, int write_id);
extern int rpc_read(conn_t *conn, void *data, size_t size);
extern int rpc_read_end(conn_t *conn);

extern int rpc_wait_for_response(conn_t *conn);

extern int rpc_write_start_request(conn_t *conn, const unsigned int op);
extern int rpc_write_start_response(conn_t *conn, const int read_id);
extern int rpc_write(conn_t *conn, const void *data, const size_t size);
extern int rpc_write_end(conn_t *conn);