#ifndef _PTX_FROM_FATBIN_HPP
#define _PTX_FROM_FATBIN_HPP

//UNUSED
#define __cudaFatVERSION   0x00000004
#define __cudaFatMAGIC3    0xba55ed50

#define __cudaFatMAGIC     0x1ee55a01

#define __cudaFatMAGIC2    0x466243b1
#define COMPRESSED_PTX     0x0000000000001000LL

enum FatBin2EntryType {
	FATBIN_2_PTX = 0x1
};

typedef struct {
    char* gpuProfileName;            
    char* ptx;
} __cudaFatPtxEntry;

typedef struct {
    char* gpuProfileName;
    char* cubin;
} __cudaFatCubinEntry;

typedef struct __cudaFatDebugEntryRec {
    char* gpuProfileName;            
    char* debug;
    struct __cudaFatDebugEntryRec *next;
    unsigned int size;
} __cudaFatDebugEntry;

typedef struct {
    char* name;
} __cudaFatSymbol;

typedef struct __cudaFatElfEntryRec {
    char* gpuProfileName;            
    char* elf;
    struct __cudaFatElfEntryRec *next;
    unsigned int size;
} __cudaFatElfEntry;

typedef struct __cudaFatCudaBinaryRec {
    unsigned long magic;
    unsigned long version;
    unsigned long gpuInfoVersion;
    char* key;
    char* ident;
    char* usageMode;
    __cudaFatPtxEntry *ptx;
    __cudaFatCubinEntry *cubin;
    __cudaFatDebugEntry *debug;
    void* debugInfo;
    unsigned int flags;
    __cudaFatSymbol *exported;
    __cudaFatSymbol *imported;
    struct __cudaFatCudaBinaryRec *dependends;
    unsigned int characteristic;
    __cudaFatElfEntry *elf;
} __cudaFatCudaBinary;

typedef struct __cudaFatCudaBinary2EntryRec { 
    unsigned int           type;
    unsigned int           binary;
    unsigned long long int binarySize;
    unsigned int           unknown2;
    unsigned int           kindOffset;
    unsigned int           unknown3;
    unsigned int           unknown4;
    unsigned int           name;
    unsigned int           nameSize;
    unsigned long long int flags;
    unsigned long long int unknown7;
    unsigned long long int uncompressedBinarySize;
} __cudaFatCudaBinary2Entry;

typedef struct __cudaFatCudaBinary2HeaderRec { 
    unsigned int            magic;
    unsigned int            version;
    unsigned long long int  length;
} __cudaFatCudaBinary2Header;

typedef struct __cudaFatCudaBinaryRec2 {
    int magic;
    int version;
    const unsigned long long* fatbinData;
    char* f;
} __cudaFatCudaBinary2;

void ptx_from_fatbin(const void *cubin_ptr);

#endif