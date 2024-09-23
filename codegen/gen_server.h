#ifndef GEN_SERVER_H
#define GEN_SERVER_H

typedef int (*RequestHandler)(int connfd);

RequestHandler get_handler(const int op);

#endif