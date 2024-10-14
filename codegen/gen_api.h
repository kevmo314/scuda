#define RPC_nvmlInit_v2 0
#define RPC_nvmlInitWithFlags 1
#define RPC_nvmlShutdown 2
#define RPC_nvmlSystemGetDriverVersion 3
#define RPC_nvmlSystemGetNVMLVersion 4
#define RPC_nvmlSystemGetCudaDriverVersion 5
#define RPC_nvmlSystemGetCudaDriverVersion_v2 6
#define RPC_nvmlSystemGetProcessName 7
#define RPC_nvmlUnitGetCount 8
#define RPC_nvmlUnitGetHandleByIndex 9
#define RPC_nvmlUnitGetUnitInfo 10
#define RPC_nvmlUnitGetLedState 11
#define RPC_nvmlUnitGetPsuInfo 12
#define RPC_nvmlUnitGetTemperature 13
#define RPC_nvmlUnitGetFanSpeedInfo 14
#define RPC_nvmlUnitGetDevices 15
#define RPC_nvmlSystemGetHicVersion 16
#define RPC_nvmlDeviceGetCount_v2 17
#define RPC_nvmlDeviceGetAttributes_v2 18
#define RPC_nvmlDeviceGetHandleByIndex_v2 19
#define RPC_nvmlDeviceGetHandleBySerial 20
#define RPC_nvmlDeviceGetHandleByUUID 21
#define RPC_nvmlDeviceGetHandleByPciBusId_v2 22
#define RPC_nvmlDeviceGetName 23
#define RPC_nvmlDeviceGetBrand 24
#define RPC_nvmlDeviceGetIndex 25
#define RPC_nvmlDeviceGetSerial 26
#define RPC_nvmlDeviceGetMemoryAffinity 27
#define RPC_nvmlDeviceGetCpuAffinityWithinScope 28
#define RPC_nvmlDeviceGetCpuAffinity 29
#define RPC_nvmlDeviceSetCpuAffinity 30
#define RPC_nvmlDeviceClearCpuAffinity 31
#define RPC_nvmlDeviceGetTopologyCommonAncestor 32
#define RPC_nvmlDeviceGetTopologyNearestGpus 33
#define RPC_nvmlSystemGetTopologyGpuSet 34
#define RPC_nvmlDeviceGetP2PStatus 35
#define RPC_nvmlDeviceGetUUID 36
#define RPC_nvmlVgpuInstanceGetMdevUUID 37
#define RPC_nvmlDeviceGetMinorNumber 38
#define RPC_nvmlDeviceGetBoardPartNumber 39
#define RPC_nvmlDeviceGetInforomVersion 40
#define RPC_nvmlDeviceGetInforomImageVersion 41
#define RPC_nvmlDeviceGetInforomConfigurationChecksum 42
#define RPC_nvmlDeviceValidateInforom 43
#define RPC_nvmlDeviceGetDisplayMode 44
#define RPC_nvmlDeviceGetDisplayActive 45
#define RPC_nvmlDeviceGetPersistenceMode 46
#define RPC_nvmlDeviceGetPciInfo_v3 47
#define RPC_nvmlDeviceGetMaxPcieLinkGeneration 48
#define RPC_nvmlDeviceGetGpuMaxPcieLinkGeneration 49
#define RPC_nvmlDeviceGetMaxPcieLinkWidth 50
#define RPC_nvmlDeviceGetCurrPcieLinkGeneration 51
#define RPC_nvmlDeviceGetCurrPcieLinkWidth 52
#define RPC_nvmlDeviceGetPcieThroughput 53
#define RPC_nvmlDeviceGetPcieReplayCounter 54
#define RPC_nvmlDeviceGetClockInfo 55
#define RPC_nvmlDeviceGetMaxClockInfo 56
#define RPC_nvmlDeviceGetApplicationsClock 57
#define RPC_nvmlDeviceGetDefaultApplicationsClock 58
#define RPC_nvmlDeviceResetApplicationsClocks 59
#define RPC_nvmlDeviceGetClock 60
#define RPC_nvmlDeviceGetMaxCustomerBoostClock 61
#define RPC_nvmlDeviceGetSupportedMemoryClocks 62
#define RPC_nvmlDeviceGetSupportedGraphicsClocks 63
#define RPC_nvmlDeviceGetAutoBoostedClocksEnabled 64
#define RPC_nvmlDeviceSetAutoBoostedClocksEnabled 65
#define RPC_nvmlDeviceSetDefaultAutoBoostedClocksEnabled 66
#define RPC_nvmlDeviceGetFanSpeed 67
#define RPC_nvmlDeviceGetFanSpeed_v2 68
#define RPC_nvmlDeviceGetTargetFanSpeed 69
#define RPC_nvmlDeviceSetDefaultFanSpeed_v2 70
#define RPC_nvmlDeviceGetMinMaxFanSpeed 71
#define RPC_nvmlDeviceGetFanControlPolicy_v2 72
#define RPC_nvmlDeviceSetFanControlPolicy 73
#define RPC_nvmlDeviceGetNumFans 74
#define RPC_nvmlDeviceGetTemperature 75
#define RPC_nvmlDeviceGetTemperatureThreshold 76
#define RPC_nvmlDeviceSetTemperatureThreshold 77
#define RPC_nvmlDeviceGetThermalSettings 78
#define RPC_nvmlDeviceGetPerformanceState 79
#define RPC_nvmlDeviceGetCurrentClocksThrottleReasons 80
#define RPC_nvmlDeviceGetSupportedClocksThrottleReasons 81
#define RPC_nvmlDeviceGetPowerState 82
#define RPC_nvmlDeviceGetPowerManagementMode 83
#define RPC_nvmlDeviceGetPowerManagementLimit 84
#define RPC_nvmlDeviceGetPowerManagementLimitConstraints 85
#define RPC_nvmlDeviceGetPowerManagementDefaultLimit 86
#define RPC_nvmlDeviceGetPowerUsage 87
#define RPC_nvmlDeviceGetTotalEnergyConsumption 88
#define RPC_nvmlDeviceGetEnforcedPowerLimit 89
#define RPC_nvmlDeviceGetGpuOperationMode 90
#define RPC_nvmlDeviceGetMemoryInfo 91
#define RPC_nvmlDeviceGetMemoryInfo_v2 92
#define RPC_nvmlDeviceGetComputeMode 93
#define RPC_nvmlDeviceGetCudaComputeCapability 94
#define RPC_nvmlDeviceGetEccMode 95
#define RPC_nvmlDeviceGetDefaultEccMode 96
#define RPC_nvmlDeviceGetBoardId 97
#define RPC_nvmlDeviceGetMultiGpuBoard 98
#define RPC_nvmlDeviceGetTotalEccErrors 99
#define RPC_nvmlDeviceGetDetailedEccErrors 100
#define RPC_nvmlDeviceGetMemoryErrorCounter 101
#define RPC_nvmlDeviceGetUtilizationRates 102
#define RPC_nvmlDeviceGetEncoderUtilization 103
#define RPC_nvmlDeviceGetEncoderCapacity 104
#define RPC_nvmlDeviceGetEncoderStats 105
#define RPC_nvmlDeviceGetEncoderSessions 106
#define RPC_nvmlDeviceGetDecoderUtilization 107
#define RPC_nvmlDeviceGetFBCStats 108
#define RPC_nvmlDeviceGetFBCSessions 109
#define RPC_nvmlDeviceGetDriverModel 110
#define RPC_nvmlDeviceGetVbiosVersion 111
#define RPC_nvmlDeviceGetBridgeChipInfo 112
#define RPC_nvmlDeviceGetComputeRunningProcesses_v3 113
#define RPC_nvmlDeviceGetGraphicsRunningProcesses_v3 114
#define RPC_nvmlDeviceGetMPSComputeRunningProcesses_v3 115
#define RPC_nvmlDeviceOnSameBoard 116
#define RPC_nvmlDeviceGetAPIRestriction 117
#define RPC_nvmlDeviceGetSamples 118
#define RPC_nvmlDeviceGetBAR1MemoryInfo 119
#define RPC_nvmlDeviceGetViolationStatus 120
#define RPC_nvmlDeviceGetIrqNum 121
#define RPC_nvmlDeviceGetNumGpuCores 122
#define RPC_nvmlDeviceGetPowerSource 123
#define RPC_nvmlDeviceGetMemoryBusWidth 124
#define RPC_nvmlDeviceGetPcieLinkMaxSpeed 125
#define RPC_nvmlDeviceGetPcieSpeed 126
#define RPC_nvmlDeviceGetAdaptiveClockInfoStatus 127
#define RPC_nvmlDeviceGetAccountingMode 128
#define RPC_nvmlDeviceGetAccountingStats 129
#define RPC_nvmlDeviceGetAccountingPids 130
#define RPC_nvmlDeviceGetAccountingBufferSize 131
#define RPC_nvmlDeviceGetRetiredPages 132
#define RPC_nvmlDeviceGetRetiredPages_v2 133
#define RPC_nvmlDeviceGetRetiredPagesPendingStatus 134
#define RPC_nvmlDeviceGetRemappedRows 135
#define RPC_nvmlDeviceGetRowRemapperHistogram 136
#define RPC_nvmlDeviceGetArchitecture 137
#define RPC_nvmlUnitSetLedState 138
#define RPC_nvmlDeviceSetPersistenceMode 139
#define RPC_nvmlDeviceSetComputeMode 140
#define RPC_nvmlDeviceSetEccMode 141
#define RPC_nvmlDeviceClearEccErrorCounts 142
#define RPC_nvmlDeviceSetDriverModel 143
#define RPC_nvmlDeviceSetGpuLockedClocks 144
#define RPC_nvmlDeviceResetGpuLockedClocks 145
#define RPC_nvmlDeviceSetMemoryLockedClocks 146
#define RPC_nvmlDeviceResetMemoryLockedClocks 147
#define RPC_nvmlDeviceSetApplicationsClocks 148
#define RPC_nvmlDeviceGetClkMonStatus 149
#define RPC_nvmlDeviceSetPowerManagementLimit 150
#define RPC_nvmlDeviceSetGpuOperationMode 151
#define RPC_nvmlDeviceSetAPIRestriction 152
#define RPC_nvmlDeviceSetAccountingMode 153
#define RPC_nvmlDeviceClearAccountingPids 154
#define RPC_nvmlDeviceGetNvLinkState 155
#define RPC_nvmlDeviceGetNvLinkVersion 156
#define RPC_nvmlDeviceGetNvLinkCapability 157
#define RPC_nvmlDeviceGetNvLinkRemotePciInfo_v2 158
#define RPC_nvmlDeviceGetNvLinkErrorCounter 159
#define RPC_nvmlDeviceResetNvLinkErrorCounters 160
#define RPC_nvmlDeviceSetNvLinkUtilizationControl 161
#define RPC_nvmlDeviceGetNvLinkUtilizationControl 162
#define RPC_nvmlDeviceGetNvLinkUtilizationCounter 163
#define RPC_nvmlDeviceFreezeNvLinkUtilizationCounter 164
#define RPC_nvmlDeviceResetNvLinkUtilizationCounter 165
#define RPC_nvmlDeviceGetNvLinkRemoteDeviceType 166
#define RPC_nvmlEventSetCreate 167
#define RPC_nvmlDeviceRegisterEvents 168
#define RPC_nvmlDeviceGetSupportedEventTypes 169
#define RPC_nvmlEventSetWait_v2 170
#define RPC_nvmlEventSetFree 171
#define RPC_nvmlDeviceModifyDrainState 172
#define RPC_nvmlDeviceQueryDrainState 173
#define RPC_nvmlDeviceRemoveGpu_v2 174
#define RPC_nvmlDeviceDiscoverGpus 175
#define RPC_nvmlDeviceGetFieldValues 176
#define RPC_nvmlDeviceClearFieldValues 177
#define RPC_nvmlDeviceGetVirtualizationMode 178
#define RPC_nvmlDeviceGetHostVgpuMode 179
#define RPC_nvmlDeviceSetVirtualizationMode 180
#define RPC_nvmlDeviceGetGridLicensableFeatures_v4 181
#define RPC_nvmlDeviceGetProcessUtilization 182
#define RPC_nvmlDeviceGetGspFirmwareVersion 183
#define RPC_nvmlDeviceGetGspFirmwareMode 184
#define RPC_nvmlGetVgpuDriverCapabilities 185
#define RPC_nvmlDeviceGetVgpuCapabilities 186
#define RPC_nvmlDeviceGetSupportedVgpus 187
#define RPC_nvmlDeviceGetCreatableVgpus 188
#define RPC_nvmlVgpuTypeGetClass 189
#define RPC_nvmlVgpuTypeGetName 190
#define RPC_nvmlVgpuTypeGetGpuInstanceProfileId 191
#define RPC_nvmlVgpuTypeGetDeviceID 192
#define RPC_nvmlVgpuTypeGetFramebufferSize 193
#define RPC_nvmlVgpuTypeGetNumDisplayHeads 194
#define RPC_nvmlVgpuTypeGetResolution 195
#define RPC_nvmlVgpuTypeGetLicense 196
#define RPC_nvmlVgpuTypeGetFrameRateLimit 197
#define RPC_nvmlVgpuTypeGetMaxInstances 198
#define RPC_nvmlVgpuTypeGetMaxInstancesPerVm 199
#define RPC_nvmlDeviceGetActiveVgpus 200
#define RPC_nvmlVgpuInstanceGetVmID 201
#define RPC_nvmlVgpuInstanceGetUUID 202
#define RPC_nvmlVgpuInstanceGetVmDriverVersion 203
#define RPC_nvmlVgpuInstanceGetFbUsage 204
#define RPC_nvmlVgpuInstanceGetLicenseStatus 205
#define RPC_nvmlVgpuInstanceGetType 206
#define RPC_nvmlVgpuInstanceGetFrameRateLimit 207
#define RPC_nvmlVgpuInstanceGetEccMode 208
#define RPC_nvmlVgpuInstanceGetEncoderCapacity 209
#define RPC_nvmlVgpuInstanceSetEncoderCapacity 210
#define RPC_nvmlVgpuInstanceGetEncoderStats 211
#define RPC_nvmlVgpuInstanceGetEncoderSessions 212
#define RPC_nvmlVgpuInstanceGetFBCStats 213
#define RPC_nvmlVgpuInstanceGetFBCSessions 214
#define RPC_nvmlVgpuInstanceGetGpuInstanceId 215
#define RPC_nvmlVgpuInstanceGetGpuPciId 216
#define RPC_nvmlVgpuTypeGetCapabilities 217
#define RPC_nvmlVgpuInstanceGetMetadata 218
#define RPC_nvmlDeviceGetVgpuMetadata 219
#define RPC_nvmlGetVgpuCompatibility 220
#define RPC_nvmlDeviceGetPgpuMetadataString 221
#define RPC_nvmlDeviceGetVgpuSchedulerLog 222
#define RPC_nvmlDeviceGetVgpuSchedulerState 223
#define RPC_nvmlDeviceGetVgpuSchedulerCapabilities 224
#define RPC_nvmlGetVgpuVersion 225
#define RPC_nvmlSetVgpuVersion 226
#define RPC_nvmlDeviceGetVgpuUtilization 227
#define RPC_nvmlDeviceGetVgpuProcessUtilization 228
#define RPC_nvmlVgpuInstanceGetAccountingMode 229
#define RPC_nvmlVgpuInstanceGetAccountingPids 230
#define RPC_nvmlVgpuInstanceGetAccountingStats 231
#define RPC_nvmlVgpuInstanceClearAccountingPids 232
#define RPC_nvmlVgpuInstanceGetLicenseInfo_v2 233
#define RPC_nvmlGetExcludedDeviceCount 234
#define RPC_nvmlGetExcludedDeviceInfoByIndex 235
#define RPC_nvmlDeviceSetMigMode 236
#define RPC_nvmlDeviceGetMigMode 237
#define RPC_nvmlDeviceGetGpuInstanceProfileInfo 238
#define RPC_nvmlDeviceGetGpuInstanceProfileInfoV 239
#define RPC_nvmlDeviceGetGpuInstancePossiblePlacements_v2 240
#define RPC_nvmlDeviceGetGpuInstanceRemainingCapacity 241
#define RPC_nvmlDeviceCreateGpuInstance 242
#define RPC_nvmlGpuInstanceDestroy 243
#define RPC_nvmlDeviceGetGpuInstances 244
#define RPC_nvmlDeviceGetGpuInstanceById 245
#define RPC_nvmlGpuInstanceGetInfo 246
#define RPC_nvmlGpuInstanceGetComputeInstanceProfileInfo 247
#define RPC_nvmlGpuInstanceGetComputeInstanceProfileInfoV 248
#define RPC_nvmlGpuInstanceGetComputeInstanceRemainingCapacity 249
#define RPC_nvmlGpuInstanceGetComputeInstancePossiblePlacements 250
#define RPC_nvmlGpuInstanceCreateComputeInstance 251
#define RPC_nvmlComputeInstanceDestroy 252
#define RPC_nvmlGpuInstanceGetComputeInstances 253
#define RPC_nvmlGpuInstanceGetComputeInstanceById 254
#define RPC_nvmlComputeInstanceGetInfo_v2 255
#define RPC_nvmlDeviceIsMigDeviceHandle 256
#define RPC_nvmlDeviceGetGpuInstanceId 257
#define RPC_nvmlDeviceGetComputeInstanceId 258
#define RPC_nvmlDeviceGetMaxMigDeviceCount 259
#define RPC_nvmlDeviceGetMigDeviceHandleByIndex 260
#define RPC_nvmlDeviceGetDeviceHandleFromMigDeviceHandle 261
#define RPC_nvmlDeviceGetBusType 262
#define RPC_nvmlDeviceGetDynamicPstatesInfo 263
#define RPC_nvmlDeviceSetFanSpeed_v2 264
#define RPC_nvmlDeviceGetGpcClkVfOffset 265
#define RPC_nvmlDeviceSetGpcClkVfOffset 266
#define RPC_nvmlDeviceGetMemClkVfOffset 267
#define RPC_nvmlDeviceSetMemClkVfOffset 268
#define RPC_nvmlDeviceGetMinMaxClockOfPState 269
#define RPC_nvmlDeviceGetSupportedPerformanceStates 270
#define RPC_nvmlDeviceGetGpcClkMinMaxVfOffset 271
#define RPC_nvmlDeviceGetMemClkMinMaxVfOffset 272
#define RPC_nvmlDeviceGetGpuFabricInfo 273
#define RPC_nvmlGpmMetricsGet 274
#define RPC_nvmlGpmSampleFree 275
#define RPC_nvmlGpmSampleAlloc 276
#define RPC_nvmlGpmSampleGet 277
#define RPC_nvmlGpmMigSampleGet 278
#define RPC_nvmlGpmQueryDeviceSupport 279
#define RPC_nvmlDeviceSetNvLinkDeviceLowPowerThreshold 280
#define RPC_cuInit 281
#define RPC_cuDriverGetVersion 282
#define RPC_cuDeviceGet 283
#define RPC_cuDeviceGetCount 284
#define RPC_cuDeviceGetName 285
#define RPC_cuDeviceGetUuid 286
#define RPC_cuDeviceGetUuid_v2 287
#define RPC_cuDeviceGetLuid 288
#define RPC_cuDeviceTotalMem_v2 289
#define RPC_cuDeviceGetTexture1DLinearMaxWidth 290
#define RPC_cuDeviceGetAttribute 291
#define RPC_cuDeviceSetMemPool 292
#define RPC_cuDeviceGetMemPool 293
#define RPC_cuDeviceGetDefaultMemPool 294
#define RPC_cuDeviceGetExecAffinitySupport 295
#define RPC_cuFlushGPUDirectRDMAWrites 296
#define RPC_cuDeviceGetProperties 297
#define RPC_cuDeviceComputeCapability 298
#define RPC_cuDevicePrimaryCtxRetain 299
#define RPC_cuDevicePrimaryCtxRelease_v2 300
#define RPC_cuDevicePrimaryCtxSetFlags_v2 301
#define RPC_cuDevicePrimaryCtxGetState 302
#define RPC_cuDevicePrimaryCtxReset_v2 303
#define RPC_cuCtxCreate_v2 304
#define RPC_cuCtxCreate_v3 305
#define RPC_cuCtxDestroy_v2 306
#define RPC_cuCtxPushCurrent_v2 307
#define RPC_cuCtxPopCurrent_v2 308
#define RPC_cuCtxSetCurrent 309
#define RPC_cuCtxGetCurrent 310
#define RPC_cuCtxGetDevice 311
#define RPC_cuCtxGetFlags 312
#define RPC_cuCtxGetId 313
#define RPC_cuCtxSynchronize 314
#define RPC_cuCtxSetLimit 315
#define RPC_cuCtxGetLimit 316
#define RPC_cuCtxGetCacheConfig 317
#define RPC_cuCtxSetCacheConfig 318
#define RPC_cuCtxGetSharedMemConfig 319
#define RPC_cuCtxSetSharedMemConfig 320
#define RPC_cuCtxGetApiVersion 321
#define RPC_cuCtxGetStreamPriorityRange 322
#define RPC_cuCtxResetPersistingL2Cache 323
#define RPC_cuCtxGetExecAffinity 324
#define RPC_cuCtxAttach 325
#define RPC_cuCtxDetach 326
#define RPC_cuModuleLoad 327
#define RPC_cuModuleUnload 328
#define RPC_cuModuleGetLoadingMode 329
#define RPC_cuModuleGetFunction 330
#define RPC_cuModuleGetGlobal_v2 331
#define RPC_cuLinkCreate_v2 332
#define RPC_cuLinkAddData_v2 333
#define RPC_cuLinkAddFile_v2 334
#define RPC_cuLinkComplete 335
#define RPC_cuLinkDestroy 336
#define RPC_cuModuleGetTexRef 337
#define RPC_cuModuleGetSurfRef 338
#define RPC_cuLibraryLoadFromFile 339
#define RPC_cuLibraryUnload 340
#define RPC_cuLibraryGetKernel 341
#define RPC_cuLibraryGetModule 342
#define RPC_cuKernelGetFunction 343
#define RPC_cuLibraryGetGlobal 344
#define RPC_cuLibraryGetManaged 345
#define RPC_cuLibraryGetUnifiedFunction 346
#define RPC_cuKernelGetAttribute 347
#define RPC_cuKernelSetAttribute 348
#define RPC_cuKernelSetCacheConfig 349
#define RPC_cuMemGetInfo_v2 350
#define RPC_cuMemAlloc_v2 351
#define RPC_cuMemAllocPitch_v2 352
#define RPC_cuMemFree_v2 353
#define RPC_cuMemGetAddressRange_v2 354
#define RPC_cuMemAllocHost_v2 355
#define RPC_cuMemFreeHost 356
#define RPC_cuMemHostAlloc 357
#define RPC_cuMemHostGetDevicePointer_v2 358
#define RPC_cuMemHostGetFlags 359
#define RPC_cuMemAllocManaged 360
#define RPC_cuDeviceGetByPCIBusId 361
#define RPC_cuDeviceGetPCIBusId 362
#define RPC_cuIpcGetEventHandle 363
#define RPC_cuIpcOpenEventHandle 364
#define RPC_cuIpcGetMemHandle 365
#define RPC_cuIpcOpenMemHandle_v2 366
#define RPC_cuIpcCloseMemHandle 367
#define RPC_cuMemHostRegister_v2 368
#define RPC_cuMemHostUnregister 369
#define RPC_cuMemcpy 370
#define RPC_cuMemcpyPeer 371
#define RPC_cuMemcpyDtoH_v2 372
#define RPC_cuMemcpyDtoD_v2 373
#define RPC_cuMemcpyDtoA_v2 374
#define RPC_cuMemcpyAtoD_v2 375
#define RPC_cuMemcpyAtoH_v2 376
#define RPC_cuMemcpyAtoA_v2 377
#define RPC_cuMemcpyAsync 378
#define RPC_cuMemcpyPeerAsync 379
#define RPC_cuMemcpyDtoHAsync_v2 380
#define RPC_cuMemcpyDtoDAsync_v2 381
#define RPC_cuMemcpyAtoHAsync_v2 382
#define RPC_cuMemsetD8_v2 383
#define RPC_cuMemsetD16_v2 384
#define RPC_cuMemsetD32_v2 385
#define RPC_cuMemsetD2D8_v2 386
#define RPC_cuMemsetD2D16_v2 387
#define RPC_cuMemsetD2D32_v2 388
#define RPC_cuMemsetD8Async 389
#define RPC_cuMemsetD16Async 390
#define RPC_cuMemsetD32Async 391
#define RPC_cuMemsetD2D8Async 392
#define RPC_cuMemsetD2D16Async 393
#define RPC_cuMemsetD2D32Async 394
#define RPC_cuArrayGetDescriptor_v2 395
#define RPC_cuArrayGetSparseProperties 396
#define RPC_cuMipmappedArrayGetSparseProperties 397
#define RPC_cuArrayGetMemoryRequirements 398
#define RPC_cuMipmappedArrayGetMemoryRequirements 399
#define RPC_cuArrayGetPlane 400
#define RPC_cuArrayDestroy 401
#define RPC_cuArray3DGetDescriptor_v2 402
#define RPC_cuMipmappedArrayGetLevel 403
#define RPC_cuMipmappedArrayDestroy 404
#define RPC_cuMemGetHandleForAddressRange 405
#define RPC_cuMemAddressReserve 406
#define RPC_cuMemAddressFree 407
#define RPC_cuMemRelease 408
#define RPC_cuMemMap 409
#define RPC_cuMemMapArrayAsync 410
#define RPC_cuMemUnmap 411
#define RPC_cuMemExportToShareableHandle 412
#define RPC_cuMemImportFromShareableHandle 413
#define RPC_cuMemGetAllocationPropertiesFromHandle 414
#define RPC_cuMemRetainAllocationHandle 415
#define RPC_cuMemFreeAsync 416
#define RPC_cuMemAllocAsync 417
#define RPC_cuMemPoolTrimTo 418
#define RPC_cuMemPoolSetAttribute 419
#define RPC_cuMemPoolGetAttribute 420
#define RPC_cuMemPoolGetAccess 421
#define RPC_cuMemPoolDestroy 422
#define RPC_cuMemAllocFromPoolAsync 423
#define RPC_cuMemPoolExportToShareableHandle 424
#define RPC_cuMemPoolImportFromShareableHandle 425
#define RPC_cuMemPoolExportPointer 426
#define RPC_cuMemPoolImportPointer 427
#define RPC_cuPointerGetAttribute 428
#define RPC_cuMemPrefetchAsync 429
#define RPC_cuMemAdvise 430
#define RPC_cuMemRangeGetAttribute 431
#define RPC_cuMemRangeGetAttributes 432
#define RPC_cuPointerGetAttributes 433
#define RPC_cuStreamCreate 434
#define RPC_cuStreamCreateWithPriority 435
#define RPC_cuStreamGetPriority 436
#define RPC_cuStreamGetFlags 437
#define RPC_cuStreamGetId 438
#define RPC_cuStreamGetCtx 439
#define RPC_cuStreamWaitEvent 440
#define RPC_cuStreamAddCallback 441
#define RPC_cuStreamBeginCapture_v2 442
#define RPC_cuThreadExchangeStreamCaptureMode 443
#define RPC_cuStreamEndCapture 444
#define RPC_cuStreamIsCapturing 445
#define RPC_cuStreamUpdateCaptureDependencies 446
#define RPC_cuStreamAttachMemAsync 447
#define RPC_cuStreamQuery 448
#define RPC_cuStreamSynchronize 449
#define RPC_cuStreamDestroy_v2 450
#define RPC_cuStreamCopyAttributes 451
#define RPC_cuStreamGetAttribute 452
#define RPC_cuEventCreate 453
#define RPC_cuEventRecord 454
#define RPC_cuEventRecordWithFlags 455
#define RPC_cuEventQuery 456
#define RPC_cuEventSynchronize 457
#define RPC_cuEventDestroy_v2 458
#define RPC_cuEventElapsedTime 459
#define RPC_cuDestroyExternalMemory 460
#define RPC_cuDestroyExternalSemaphore 461
#define RPC_cuStreamWaitValue32_v2 462
#define RPC_cuStreamWaitValue64_v2 463
#define RPC_cuStreamWriteValue32_v2 464
#define RPC_cuStreamWriteValue64_v2 465
#define RPC_cuStreamBatchMemOp_v2 466
#define RPC_cuFuncGetAttribute 467
#define RPC_cuFuncSetAttribute 468
#define RPC_cuFuncSetCacheConfig 469
#define RPC_cuFuncSetSharedMemConfig 470
#define RPC_cuFuncGetModule 471
#define RPC_cuLaunchKernel 472
#define RPC_cuLaunchCooperativeKernel 473
#define RPC_cuLaunchCooperativeKernelMultiDevice 474
#define RPC_cuLaunchHostFunc 475
#define RPC_cuFuncSetBlockShape 476
#define RPC_cuFuncSetSharedSize 477
#define RPC_cuParamSetSize 478
#define RPC_cuParamSeti 479
#define RPC_cuParamSetf 480
#define RPC_cuParamSetv 481
#define RPC_cuLaunch 482
#define RPC_cuLaunchGrid 483
#define RPC_cuLaunchGridAsync 484
#define RPC_cuParamSetTexRef 485
#define RPC_cuGraphCreate 486
#define RPC_cuGraphKernelNodeGetParams_v2 487
#define RPC_cuGraphMemcpyNodeGetParams 488
#define RPC_cuGraphMemsetNodeGetParams 489
#define RPC_cuGraphHostNodeGetParams 490
#define RPC_cuGraphChildGraphNodeGetGraph 491
#define RPC_cuGraphEventRecordNodeGetEvent 492
#define RPC_cuGraphEventRecordNodeSetEvent 493
#define RPC_cuGraphEventWaitNodeGetEvent 494
#define RPC_cuGraphEventWaitNodeSetEvent 495
#define RPC_cuGraphExternalSemaphoresSignalNodeGetParams 496
#define RPC_cuGraphExternalSemaphoresWaitNodeGetParams 497
#define RPC_cuGraphBatchMemOpNodeGetParams 498
#define RPC_cuGraphMemAllocNodeGetParams 499
#define RPC_cuGraphMemFreeNodeGetParams 500
#define RPC_cuDeviceGraphMemTrim 501
#define RPC_cuDeviceGetGraphMemAttribute 502
#define RPC_cuDeviceSetGraphMemAttribute 503
#define RPC_cuGraphClone 504
#define RPC_cuGraphNodeFindInClone 505
#define RPC_cuGraphNodeGetType 506
#define RPC_cuGraphGetNodes 507
#define RPC_cuGraphGetRootNodes 508
#define RPC_cuGraphGetEdges 509
#define RPC_cuGraphNodeGetDependencies 510
#define RPC_cuGraphNodeGetDependentNodes 511
#define RPC_cuGraphDestroyNode 512
#define RPC_cuGraphInstantiateWithFlags 513
#define RPC_cuGraphInstantiateWithParams 514
#define RPC_cuGraphExecGetFlags 515
#define RPC_cuGraphExecChildGraphNodeSetParams 516
#define RPC_cuGraphExecEventRecordNodeSetEvent 517
#define RPC_cuGraphExecEventWaitNodeSetEvent 518
#define RPC_cuGraphNodeSetEnabled 519
#define RPC_cuGraphNodeGetEnabled 520
#define RPC_cuGraphUpload 521
#define RPC_cuGraphLaunch 522
#define RPC_cuGraphExecDestroy 523
#define RPC_cuGraphDestroy 524
#define RPC_cuGraphExecUpdate_v2 525
#define RPC_cuGraphKernelNodeCopyAttributes 526
#define RPC_cuGraphKernelNodeGetAttribute 527
#define RPC_cuUserObjectCreate 528
#define RPC_cuUserObjectRetain 529
#define RPC_cuUserObjectRelease 530
#define RPC_cuGraphRetainUserObject 531
#define RPC_cuGraphReleaseUserObject 532
#define RPC_cuOccupancyMaxActiveBlocksPerMultiprocessor 533
#define RPC_cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags 534
#define RPC_cuOccupancyAvailableDynamicSMemPerBlock 535
#define RPC_cuTexRefSetArray 536
#define RPC_cuTexRefSetMipmappedArray 537
#define RPC_cuTexRefSetAddress_v2 538
#define RPC_cuTexRefSetFormat 539
#define RPC_cuTexRefSetAddressMode 540
#define RPC_cuTexRefSetFilterMode 541
#define RPC_cuTexRefSetMipmapFilterMode 542
#define RPC_cuTexRefSetMipmapLevelBias 543
#define RPC_cuTexRefSetMipmapLevelClamp 544
#define RPC_cuTexRefSetMaxAnisotropy 545
#define RPC_cuTexRefSetBorderColor 546
#define RPC_cuTexRefSetFlags 547
#define RPC_cuTexRefGetAddress_v2 548
#define RPC_cuTexRefGetArray 549
#define RPC_cuTexRefGetMipmappedArray 550
#define RPC_cuTexRefGetAddressMode 551
#define RPC_cuTexRefGetFilterMode 552
#define RPC_cuTexRefGetFormat 553
#define RPC_cuTexRefGetMipmapFilterMode 554
#define RPC_cuTexRefGetMipmapLevelBias 555
#define RPC_cuTexRefGetMipmapLevelClamp 556
#define RPC_cuTexRefGetMaxAnisotropy 557
#define RPC_cuTexRefGetBorderColor 558
#define RPC_cuTexRefGetFlags 559
#define RPC_cuTexRefCreate 560
#define RPC_cuTexRefDestroy 561
#define RPC_cuSurfRefSetArray 562
#define RPC_cuSurfRefGetArray 563
#define RPC_cuTexObjectDestroy 564
#define RPC_cuTexObjectGetResourceDesc 565
#define RPC_cuTexObjectGetTextureDesc 566
#define RPC_cuTexObjectGetResourceViewDesc 567
#define RPC_cuSurfObjectDestroy 568
#define RPC_cuSurfObjectGetResourceDesc 569
#define RPC_cuTensorMapReplaceAddress 570
#define RPC_cuDeviceCanAccessPeer 571
#define RPC_cuCtxEnablePeerAccess 572
#define RPC_cuCtxDisablePeerAccess 573
#define RPC_cuDeviceGetP2PAttribute 574
#define RPC_cuGraphicsUnregisterResource 575
#define RPC_cuGraphicsSubResourceGetMappedArray 576
#define RPC_cuGraphicsResourceGetMappedMipmappedArray 577
#define RPC_cuGraphicsResourceGetMappedPointer_v2 578
#define RPC_cuGraphicsResourceSetMapFlags_v2 579
#define RPC_cuGraphicsMapResources 580
#define RPC_cuGraphicsUnmapResources 581
#define RPC_cudaDeviceReset 582
#define RPC_cudaDeviceSynchronize 583
#define RPC_cudaDeviceSetLimit 584
#define RPC_cudaDeviceGetLimit 585
#define RPC_cudaDeviceGetCacheConfig 586
#define RPC_cudaDeviceGetStreamPriorityRange 587
#define RPC_cudaDeviceSetCacheConfig 588
#define RPC_cudaDeviceGetSharedMemConfig 589
#define RPC_cudaDeviceSetSharedMemConfig 590
#define RPC_cudaDeviceGetPCIBusId 591
#define RPC_cudaIpcGetEventHandle 592
#define RPC_cudaIpcOpenEventHandle 593
#define RPC_cudaIpcGetMemHandle 594
#define RPC_cudaIpcOpenMemHandle 595
#define RPC_cudaIpcCloseMemHandle 596
#define RPC_cudaDeviceFlushGPUDirectRDMAWrites 597
#define RPC_cudaThreadExit 598
#define RPC_cudaThreadSynchronize 599
#define RPC_cudaThreadSetLimit 600
#define RPC_cudaThreadGetLimit 601
#define RPC_cudaThreadGetCacheConfig 602
#define RPC_cudaThreadSetCacheConfig 603
#define RPC_cudaGetLastError 604
#define RPC_cudaPeekAtLastError 605
#define RPC_cudaGetDeviceCount 606
#define RPC_cudaGetDeviceProperties_v2 607
#define RPC_cudaDeviceGetAttribute 608
#define RPC_cudaDeviceGetDefaultMemPool 609
#define RPC_cudaDeviceSetMemPool 610
#define RPC_cudaDeviceGetMemPool 611
#define RPC_cudaDeviceGetNvSciSyncAttributes 612
#define RPC_cudaDeviceGetP2PAttribute 613
#define RPC_cudaInitDevice 614
#define RPC_cudaSetDevice 615
#define RPC_cudaGetDevice 616
#define RPC_cudaSetValidDevices 617
#define RPC_cudaSetDeviceFlags 618
#define RPC_cudaGetDeviceFlags 619
#define RPC_cudaStreamCreate 620
#define RPC_cudaStreamCreateWithFlags 621
#define RPC_cudaStreamCreateWithPriority 622
#define RPC_cudaStreamGetPriority 623
#define RPC_cudaStreamGetFlags 624
#define RPC_cudaStreamGetId 625
#define RPC_cudaCtxResetPersistingL2Cache 626
#define RPC_cudaStreamCopyAttributes 627
#define RPC_cudaStreamGetAttribute 628
#define RPC_cudaStreamDestroy 629
#define RPC_cudaStreamWaitEvent 630
#define RPC_cudaStreamAddCallback 631
#define RPC_cudaStreamSynchronize 632
#define RPC_cudaStreamQuery 633
#define RPC_cudaStreamAttachMemAsync 634
#define RPC_cudaStreamBeginCapture 635
#define RPC_cudaThreadExchangeStreamCaptureMode 636
#define RPC_cudaStreamEndCapture 637
#define RPC_cudaStreamIsCapturing 638
#define RPC_cudaStreamGetCaptureInfo_v2 639
#define RPC_cudaStreamUpdateCaptureDependencies 640
#define RPC_cudaEventCreate 641
#define RPC_cudaEventCreateWithFlags 642
#define RPC_cudaEventRecord 643
#define RPC_cudaEventRecordWithFlags 644
#define RPC_cudaEventQuery 645
#define RPC_cudaEventSynchronize 646
#define RPC_cudaEventDestroy 647
#define RPC_cudaEventElapsedTime 648
#define RPC_cudaDestroyExternalMemory 649
#define RPC_cudaDestroyExternalSemaphore 650
#define RPC_cudaLaunchCooperativeKernelMultiDevice 651
#define RPC_cudaSetDoubleForDevice 652
#define RPC_cudaSetDoubleForHost 653
#define RPC_cudaLaunchHostFunc 654
#define RPC_cudaMallocManaged 655
#define RPC_cudaMalloc 656
#define RPC_cudaMallocHost 657
#define RPC_cudaMallocPitch 658
#define RPC_cudaFree 659
#define RPC_cudaFreeHost 660
#define RPC_cudaFreeArray 661
#define RPC_cudaFreeMipmappedArray 662
#define RPC_cudaHostAlloc 663
#define RPC_cudaHostRegister 664
#define RPC_cudaHostUnregister 665
#define RPC_cudaHostGetDevicePointer 666
#define RPC_cudaHostGetFlags 667
#define RPC_cudaMalloc3D 668
#define RPC_cudaGetMipmappedArrayLevel 669
#define RPC_cudaMemGetInfo 670
#define RPC_cudaArrayGetInfo 671
#define RPC_cudaArrayGetPlane 672
#define RPC_cudaArrayGetMemoryRequirements 673
#define RPC_cudaMipmappedArrayGetMemoryRequirements 674
#define RPC_cudaArrayGetSparseProperties 675
#define RPC_cudaMipmappedArrayGetSparseProperties 676
#define RPC_cudaMemcpy2DFromArray 677
#define RPC_cudaMemcpy2DArrayToArray 678
#define RPC_cudaMemcpy2DFromArrayAsync 679
#define RPC_cudaMemset 680
#define RPC_cudaMemset2D 681
#define RPC_cudaMemset3D 682
#define RPC_cudaMemsetAsync 683
#define RPC_cudaMemset2DAsync 684
#define RPC_cudaMemset3DAsync 685
#define RPC_cudaMemcpyFromArray 686
#define RPC_cudaMemcpyArrayToArray 687
#define RPC_cudaMemcpyFromArrayAsync 688
#define RPC_cudaMallocAsync 689
#define RPC_cudaFreeAsync 690
#define RPC_cudaMemPoolTrimTo 691
#define RPC_cudaMemPoolSetAttribute 692
#define RPC_cudaMemPoolGetAttribute 693
#define RPC_cudaMemPoolGetAccess 694
#define RPC_cudaMemPoolDestroy 695
#define RPC_cudaMallocFromPoolAsync 696
#define RPC_cudaMemPoolExportToShareableHandle 697
#define RPC_cudaMemPoolImportFromShareableHandle 698
#define RPC_cudaMemPoolExportPointer 699
#define RPC_cudaMemPoolImportPointer 700
#define RPC_cudaDeviceCanAccessPeer 701
#define RPC_cudaDeviceEnablePeerAccess 702
#define RPC_cudaDeviceDisablePeerAccess 703
#define RPC_cudaGraphicsUnregisterResource 704
#define RPC_cudaGraphicsResourceSetMapFlags 705
#define RPC_cudaGraphicsMapResources 706
#define RPC_cudaGraphicsUnmapResources 707
#define RPC_cudaGraphicsResourceGetMappedPointer 708
#define RPC_cudaGraphicsSubResourceGetMappedArray 709
#define RPC_cudaGraphicsResourceGetMappedMipmappedArray 710
#define RPC_cudaGetChannelDesc 711
#define RPC_cudaDestroyTextureObject 712
#define RPC_cudaGetTextureObjectResourceDesc 713
#define RPC_cudaGetTextureObjectTextureDesc 714
#define RPC_cudaGetTextureObjectResourceViewDesc 715
#define RPC_cudaDestroySurfaceObject 716
#define RPC_cudaGetSurfaceObjectResourceDesc 717
#define RPC_cudaDriverGetVersion 718
#define RPC_cudaRuntimeGetVersion 719
#define RPC_cudaGraphCreate 720
#define RPC_cudaGraphKernelNodeGetParams 721
#define RPC_cudaGraphKernelNodeCopyAttributes 722
#define RPC_cudaGraphKernelNodeGetAttribute 723
#define RPC_cudaGraphMemcpyNodeGetParams 724
#define RPC_cudaGraphMemsetNodeGetParams 725
#define RPC_cudaGraphHostNodeGetParams 726
#define RPC_cudaGraphChildGraphNodeGetGraph 727
#define RPC_cudaGraphEventRecordNodeGetEvent 728
#define RPC_cudaGraphEventRecordNodeSetEvent 729
#define RPC_cudaGraphEventWaitNodeGetEvent 730
#define RPC_cudaGraphEventWaitNodeSetEvent 731
#define RPC_cudaGraphExternalSemaphoresSignalNodeGetParams 732
#define RPC_cudaGraphExternalSemaphoresWaitNodeGetParams 733
#define RPC_cudaGraphMemAllocNodeGetParams 734
#define RPC_cudaGraphMemFreeNodeGetParams 735
#define RPC_cudaDeviceGraphMemTrim 736
#define RPC_cudaDeviceGetGraphMemAttribute 737
#define RPC_cudaDeviceSetGraphMemAttribute 738
#define RPC_cudaGraphClone 739
#define RPC_cudaGraphNodeFindInClone 740
#define RPC_cudaGraphNodeGetType 741
#define RPC_cudaGraphGetNodes 742
#define RPC_cudaGraphGetRootNodes 743
#define RPC_cudaGraphGetEdges 744
#define RPC_cudaGraphNodeGetDependencies 745
#define RPC_cudaGraphNodeGetDependentNodes 746
#define RPC_cudaGraphDestroyNode 747
#define RPC_cudaGraphInstantiate 748
#define RPC_cudaGraphInstantiateWithFlags 749
#define RPC_cudaGraphInstantiateWithParams 750
#define RPC_cudaGraphExecGetFlags 751
#define RPC_cudaGraphExecChildGraphNodeSetParams 752
#define RPC_cudaGraphExecEventRecordNodeSetEvent 753
#define RPC_cudaGraphExecEventWaitNodeSetEvent 754
#define RPC_cudaGraphNodeSetEnabled 755
#define RPC_cudaGraphNodeGetEnabled 756
#define RPC_cudaGraphExecUpdate 757
#define RPC_cudaGraphUpload 758
#define RPC_cudaGraphLaunch 759
#define RPC_cudaGraphExecDestroy 760
#define RPC_cudaGraphDestroy 761
#define RPC_cudaUserObjectCreate 762
#define RPC_cudaUserObjectRetain 763
#define RPC_cudaUserObjectRelease 764
#define RPC_cudaGraphRetainUserObject 765
#define RPC_cudaGraphReleaseUserObject 766
#define RPC_cudaMemcpy 767
#define RPC_cudaMemcpyAsync 768
#define RPC_cudaLaunchKernel 769
