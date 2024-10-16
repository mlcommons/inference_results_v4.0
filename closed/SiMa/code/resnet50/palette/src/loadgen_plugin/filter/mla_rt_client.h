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

#ifndef MLA_RT_CLIENT_HANDLE_
#define MLA_RT_CLIENT_HANDLE_

#include <inttypes.h>
#include <cstdint>

extern "C" {
#include <simaai/gst-api.h>
}
#include <simaai/simaai_memory.h>

#include "gstsimaaimlfilter.h"

typedef int64_t simamm_buffer_id_t;

#ifdef __cplusplus
extern "C" {
#endif

void mla_rt_client_load_model (const char * path);

int mla_rt_client_run_model (simamm_buffer_id_t in_buf_id,
			     simamm_buffer_id_t out_buf_id,
			     uint32_t out_buf_offset);

int32_t
mla_rt_client_write_output_buffer (const void * data, size_t size, int id);

int32_t
mla_rt_client_write_input_buffer (const void * data, size_t size, int id);
     
int mla_rt_client_init (const char * model_path);

simaai_memory_t * mla_rt_client_prepare_outbuf (size_t output_sz);
     
void mla_rt_client_cleanup(void);

int mla_rt_client_batch_run_model (uint64_t in_addr_list[],
                                   simamm_buffer_id_t out_buf_id,
                                   uint32_t out_buf_offset,
                                   size_t batch_size,
                                   size_t in_sz,
                                   size_t out_sz);

void
mla_rt_cleanup(void);
    
#ifdef __cplusplus
}
#endif

#endif // MLA_RT_CLIENT_HANDLE_
