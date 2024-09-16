#include <arpa/inet.h>
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
#include <nvml.h>
#include <unistd.h>
#include <pthread.h>
#include <vector>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include "api.h"
#include "handlers.h"

void *dlsym(void *handle, const char *name) __THROW
{
    printf("Resolving function symbol: %s\n", name);

    // get our func by mapping and invoke it if possible
    void *func = getFunctionByName(name);
    if (func != NULL) {
        return func;
    }

    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (real_dlsym == NULL) {
        // avoid calling dlsym recursively; use dlvsym to resolve dlsym itself
        real_dlsym = (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    }

    if (!strcmp(name, "dlsym")) {
        return (void *)dlsym;
    }

    // if func symbol is not found in the handler mappings, return the real dlsym resolution
    return real_dlsym(handle, name);
}
