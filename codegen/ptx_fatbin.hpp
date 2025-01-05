#ifndef _PTX_FROM_FATBIN_HPP
#define _PTX_FROM_FATBIN_HPP

// UNUSED
#define __cudaFatVERSION 0x00000004
#define __cudaFatMAGIC3 0xba55ed50

#define __cudaFatMAGIC 0x1ee55a01

#define __cudaFatMAGIC2 0x466243b1
#define COMPRESSED_PTX 0x0000000000001000LL

enum FatBin2EntryType { FATBIN_2_PTX = 0x1 };

typedef struct {
  char *gpuProfileName;
  char *ptx;
} __cudaFatPtxEntry;

typedef struct {
  char *gpuProfileName;
  char *cubin;
} __cudaFatCubinEntry;

typedef struct __cudaFatDebugEntryRec {
  char *gpuProfileName;
  char *debug;
  struct __cudaFatDebugEntryRec *next;
  unsigned int size;
} __cudaFatDebugEntry;

typedef struct {
  char *name;
} __cudaFatSymbol;

typedef struct __cudaFatElfEntryRec {
  char *gpuProfileName;
  char *elf;
  struct __cudaFatElfEntryRec *next;
  unsigned int size;
} __cudaFatElfEntry;

typedef struct __cudaFatCudaBinaryRec {
  unsigned long magic;
  unsigned long version;
  unsigned long gpuInfoVersion;
  char *key;
  char *ident;
  char *usageMode;
  __cudaFatPtxEntry *ptx;
  __cudaFatCubinEntry *cubin;
  __cudaFatDebugEntry *debug;
  void *debugInfo;
  unsigned int flags;
  __cudaFatSymbol *exported;
  __cudaFatSymbol *imported;
  struct __cudaFatCudaBinaryRec *dependends;
  unsigned int characteristic;
  __cudaFatElfEntry *elf;
} __cudaFatCudaBinary;

typedef struct __cudaFatCudaBinary2EntryRec {
  unsigned int type;
  unsigned int binary;
  unsigned long long int binarySize;
  unsigned int unknown2;
  unsigned int kindOffset;
  unsigned int unknown3;
  unsigned int unknown4;
  unsigned int name;
  unsigned int nameSize;
  unsigned long long int flags;
  unsigned long long int unknown7;
  unsigned long long int uncompressedBinarySize;
} __cudaFatCudaBinary2Entry;

typedef struct __attribute__((__packed__)) __cudaFatCudaBinary2HeaderRec {
  uint32_t magic;
  uint16_t version;
  uint16_t header_size;
  uint64_t size;
} __cudaFatCudaBinary2Header;

typedef struct __attribute__((__packed__)) __cudaFatCudaBinaryRec2 {
  uint32_t magic;
  uint32_t version;
  uint64_t text; // points to first text section
  uint64_t data; // points to outside of the file
  uint64_t unknown;
  uint64_t text2; // points to second text section
  uint64_t zero;
} __cudaFatCudaBinary2;

void ptx_from_fatbin(const void *cubin_ptr);

#endif