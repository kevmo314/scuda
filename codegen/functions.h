struct Function {
  char *name;
  void *fat_cubin;        // the fat cubin that this function is a part of.
  const char *host_func;  // if registered, points at the host function.
  int *arg_sizes;
  int arg_count;
};
