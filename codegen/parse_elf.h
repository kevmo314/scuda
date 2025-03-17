
#include <cstdint>
#include <vector>
#include "functions.h"

void process_elf(const unsigned char *dt, size_t sz, bool compressed, 
                std::vector<Function> &funcitons,
                uint32_t max_fn_name, uint32_t max_args);
