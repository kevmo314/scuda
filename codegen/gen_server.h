#ifndef GEN_SERVER_H
#define GEN_SERVER_H

#include "rpc.h"

typedef int (*RequestHandler)(conn_t *conn);

RequestHandler get_handler(const int op);

#endif