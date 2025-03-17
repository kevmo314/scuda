#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include "elfio/elfio.hpp"
#include "lz4.h"
#include "functions.h"

#define DBG false
using namespace std;
enum NV_INFO_TYPE {
  EIFMT_NVAL = 01,
  EIFMT_HVAL = 03,
  EIFMT_SVAL = 04,
};

enum NV_INFO_ATTRIBUTE {
  EIATTR_PARAM_CBANK = 0xa,
  EIATTR_FRAME_SIZE = 0x11,
  EIATTR_MIN_STACK_SIZE = 0x12,
  EIATTR_KPARAM_INFO = 0x17,
  EIATTR_CBANK_PARAM_SIZE = 0x19,
  EIATTR_MAXREG_COUNT = 0x1b,
  EIATTR_EXIT_INSTR_OFFSETS = 0x1c,
  EIATTR_CRS_STACK_SIZE = 0x1e,
  EIATTR_REGCOUNT = 0x2f,
  EIATTR_CUDA_API_VERSION = 0x37,
  EIATTR_GEN_ERRBAR_AT_EXIT = 0x53,
};

typedef struct _attrHdr {
  char type;
  char id;
} AttrHdr;

typedef struct __attribute__((packed)) _attrHdrParam {
  uint16_t header_size;
  uint32_t index;
  uint16_t ordinal;
  uint16_t offset;
  uint16_t unk;
  uint16_t size_flag;
 } AttrHdrParam;
std::vector<std::string> split_string(const std::string &str, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(str);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}
void process_nv_info_section(const string &name, const char *data, size_t sz, 
                            vector<Function> &functions,
                            const int max_fn_name, const int max_args) {
  if (DBG) {
    for (int j = 0; j < sz; ++j) {
      if ((j % 32) == 0) cout << endl;
      printf("%02x ", (unsigned char)data[j]);
    }
    cout << endl;
  }
  int no_of_args = 0;
  vector<int> argSizes;
  const char *rdp = data;
  while (rdp < data + sz) {
    AttrHdr *hdr = (AttrHdr *)rdp;
    if (hdr->type == EIFMT_SVAL) {
      if (hdr->id == EIATTR_KPARAM_INFO) {
        AttrHdrParam *param = (AttrHdrParam *)(rdp + 2);
        // printf("header size: %hx Param Idx %x, ord:%hx ofs:%hx sz:%hx\n",
        //        param->header_size, param->index, param->ordinal,
        //        param->offset, param->size_flag >> 2);
        assert(argSizes.size() == param->ordinal);
        argSizes.push_back(param->size_flag >> 2);
      }
      rdp += 4 + *((uint16_t *)(rdp + 2));
    } else {
      rdp += 4;  // skip over these
    }
  }
  printf("   \t%s(", name.c_str());
  for (auto v : argSizes) {
    printf("%d,", v);
  }
  printf(");\n");

  char *fname = new char[max_fn_name];
  int j = 0;
  int i = 0;
  for (; j < max_fn_name - 1 && i < name.size() &&
         (isalnum(name[i]) || name[i] == '_');)
    fname[j++] = name[i++];
  fname[j] = '\0';
  Function *f = nullptr;
  for (auto &function : functions)
    if (strcmp(function.name, fname) == 0) {
      f = &function;
    }
  if (f == nullptr) {
    int *arg_sizes = new int[max_args];
    for (i = 0; i < argSizes.size(); ++i) arg_sizes[i] = argSizes[i];
    functions.push_back(Function{.name = fname,
                                 .fat_cubin = nullptr,
                                 .host_func = nullptr,
                                 .arg_sizes = arg_sizes,
                                 .arg_count = (int)argSizes.size()});
  }
}

void processELFPtx(string &name, const unsigned char *dt, size_t sz,
                   uint16_t sm_version, 
                   vector<Function> &funcitons,
                   uint32_t max_fn_name, uint32_t max_args) {
  ELFIO::elfio reader;
  //   if (!reader.load(fname)) {
  //     cout << "Error ELF loading file: " << fname << endl;
  //     return;
  //   }
  std::vector<unsigned char> buffer;
  buffer.resize(sz);
  for (int i = 0; i < sz; ++i) buffer[i] = *(dt + i);
  std::istringstream stream(std::string(buffer.begin(), buffer.end()));
  reader.load(stream);

  // Print ELF file sections info
  ELFIO::Elf_Half sec_num = reader.sections.size();
  if (DBG)
  std::cout << "Number of sections: " << sec_num << std::endl;
  for (int i = 0; i < sec_num; ++i) {
    const ELFIO::section *psec = reader.sections[i];
    if (DBG)
    std::cout << " [" << i << "] " << psec->get_name() << "\t"
              << psec->get_size() << " Type=" << psec->get_type() << std::endl;
    if (psec->get_type() == ELFIO::SHT_LOPROC) {
      auto splitname = split_string(psec->get_name(), '.');
      if (splitname.size() == 4 && splitname[1] == "nv" &&
          splitname[2] == "info" && splitname[3][0] == '_') {
        // cout << "Found Info Section " << endl;
        process_nv_info_section(splitname[3], reader.sections[i]->get_data(),
                                reader.sections[i]->get_size(), funcitons, max_fn_name, max_args);
      }
    } else if (psec->get_type() == ELFIO::SHT_PROGBITS) {
      // if (psec->get_type) SHT_PROGBITS=1 SHT_LOPROC=0x70000000
      //.nv.constant0._Z9vectorAddPKfS0_Pfi	15c Type=1
      //.text._Z9vectorAddPKfS0_Pfi	440 Typ
      // Access section's data
      auto splitname = split_string(psec->get_name(), '.');
      string fnname = "";
      if (splitname.size() == 4 && splitname[1] == "nv" &&
          splitname[2] == "constant0") {
        // fnname = splitname[3];
        // cout <<"name:"<< fnname <<endl;
      } else if (splitname.size() == 3 && splitname[1] == "text" &&
                 splitname[2][0] == '_') {
        fnname = splitname[2];
        if (DBG) cout << "name:" << fnname << endl;
      }
      if (fnname != "") {
        const char *p = reader.sections[i]->get_data();
        if (DBG) {
          cout << "Section data size:" << reader.sections[i]->get_size()
               << " Section data:" << endl;
          for (int j = 0; j < reader.sections[i]->get_size(); ++j) {
            if ((j % 32) == 0) cout << endl;
            printf("%02x ", (unsigned char)p[j]);
          }
          cout << endl;
        }
        // if (opt_p) {
        //   std::string fname = name + ".cu-" + fnname + ".sm_" +
        //                       std::to_string(sm_version) + "." +
        //                       std::to_string(ptxcnt++) + ".ptx.bin";
        //   printf("Writing to file:%s size %ld with PTX Data\n",
        //   fname.c_str(),
        //          reader.sections[i]->get_size());
        //   std::ofstream OutFile(fname, std::ofstream::binary);
        //   OutFile.write(reader.sections[i]->get_data(),
        //                 reader.sections[i]->get_size());
        //   OutFile.close();
        // }
      }
    }
  }
}

void process_elf(const unsigned char *dt, size_t sz, bool compressed, 
                vector<Function> &funcitons,
                uint32_t max_fn_name, uint32_t max_args) {
  if (compressed) {
    const int dsize = 1024 * 1000;
    unsigned char *dest = new unsigned char[dsize];
    for (int i = 0; i < dsize; ++i) dest[i] = 0;
    int r = LZ4_decompress_safe((const char *)dt, (char *)dest, sz, dsize);
    if (DBG or r < 0) {
      printf("process elf sz:%d comp:%d::", sz, compressed);
      for (int i = 0; i < 32 && i < sz; ++i) {
        printf("%02x ", dt[i]);
      }
      printf("\n");
      cout << "decompress result requested size " << sz << "/" << dsize
           << " decompressed size " << r << endl;
    }
    string st("Test");
    if (r > 0) {
      processELFPtx(st, dest, r, 89, funcitons, max_fn_name, max_args);
    }
  } else {
    string st("Test");
    processELFPtx(st, dt, sz, 89,  funcitons, max_fn_name, max_args);
  }
}
