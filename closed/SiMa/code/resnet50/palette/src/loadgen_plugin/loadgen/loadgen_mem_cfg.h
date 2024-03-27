//**************************************************************************
//||                        SiMa.ai CONFIDENTIAL                          ||
//||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
//**************************************************************************
// NOTICE:  All information contained herein is, and remains the property of
// SiMa.ai. The intellectual and technical concepts contained herein are
// proprietary to SiMa and may be covered by U.S. and Foreign Patents,
// patents in process, and are protected by trade secret or copyright law.
//
// Dissemination of this information or reproduction of this material is
// strictly forbidden unless prior written permission is obtained from
// SiMa.ai.  Access to the source code contained herein is hereby forbidden
// to anyone except current SiMa.ai employees, managers or contractors who
// have executed Confidentiality and Non-disclosure agreements explicitly
// covering such access.
//
// The copyright notice above does not evidence any actual or intended
// publication or disclosure  of  this source code, which includes information
// that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
//
// ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
// DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
// CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
// LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
// CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
// REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
// SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
//
//**************************************************************************

#ifndef LOADGEN_MEM_CFG_H_
#define LOADGEN_MEM_CFG_H_

#include <simaai/simaai_memory.h>

#define MAX_SLOT_NAME 256
#define MAX_MEM_REGIONS 24
#define TOY_MAX_MEM_REGIONS 6
#define MAX_RESERVED_MEM_REGIONS 5

// Set this flag to zero to work with mla-rt without sima-mem-lib
#define MLART_USE_SIMA_MEM_LIB 1

typedef struct loadgen_mem_cfg_s {
    int idx;  /// < An index or unique id for the slot, readable by loadgen
    char slot_name[MAX_SLOT_NAME]; /// < Just a 'name'
    int target_id;   /// < simaai memory target id
    size_t max_size; /// < Maximum size as bytes
    bool is_reserved; /// < Is the memory usable i.e non-reserved.
} loadgen_mem_cfg_t ;

/* 
 * SiMa's SoC has a 16GB ddr, the utilization the DDR is managed by
 * sima-memory driver and sima-memory userspace library.
 * For mlperf we use 7.5GB of DRAM to preload all the sample images to
 * DDR and work with it.
 * Below is the definition of all the slots available from the 4DMA
 * banks addressable to the SiMa MLA
 */

static loadgen_mem_cfg_t mem_cfg_full_mode[MAX_MEM_REGIONS] = {
/* From DMS0 */
    {0, "reservedslot0", SIMAAI_MEM_TARGET_DMS0, 256, true},
    {1, "slot0",         SIMAAI_MEM_TARGET_DMS0, 500, false},
    {2, "slot1",         SIMAAI_MEM_TARGET_DMS0, 500, false},
    {3, "slot2",         SIMAAI_MEM_TARGET_DMS0, 250, false},
/* From DMS1 */
    {4, "slot3",         SIMAAI_MEM_TARGET_DMS1, 256, true},
    {5, "reservedslot1", SIMAAI_MEM_TARGET_DMS1, 512, false},
    {6, "slot4",         SIMAAI_MEM_TARGET_DMS1, 500, false},
    {7, "slot5",         SIMAAI_MEM_TARGET_DMS1, 500, false},
    {8, "slot6",         SIMAAI_MEM_TARGET_DMS1, 250, false},
/* From DMS2 */
    {9, "slot7",        SIMAAI_MEM_TARGET_DMS2, 256, true},
    {10, "slot8",        SIMAAI_MEM_TARGET_DMS2, 512, false},
    {11, "reservedslot2",SIMAAI_MEM_TARGET_DMS2, 512, false},
    {12, "slot9",        SIMAAI_MEM_TARGET_DMS2, 512, false},
    {13, "slot10",       SIMAAI_MEM_TARGET_DMS2, 512, false},
    {14, "slot11",       SIMAAI_MEM_TARGET_DMS2, 256, false},
/* From DMS3 */
    {15, "reservedslot3",SIMAAI_MEM_TARGET_DMS3, 256, true},
    {16, "slot12",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {17, "slot13",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {18, "slot14",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {19, "slot15",       SIMAAI_MEM_TARGET_DMS3, 512, false},
    {20, "slot16",       SIMAAI_MEM_TARGET_DMS3, 256, false},
    {-1, "", -1, 0, false},
};

#endif // LOADGEN_MEM_CFG_H_
