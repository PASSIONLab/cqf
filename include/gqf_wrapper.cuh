/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef GQF_WRAPPER_CUH
#define GQF_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "gqf_int.cuh"
#include "gqf_file.cuh"

QF* g_quotient_filter;
QFi g_quotient_filter_itr;

uint64_t num_locks;

volatile uint64_t * buffer_sizes;

uint64_t num_slots;
uint64_t total_items;
	
uint64_t ** buffers;
	
uint64_t * buffer_backing;


#ifndef NUM_SLOTS_TO_LOCK
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#endif

extern inline uint64_t gqf_xnslots();

extern inline int gqf_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why

	QF temp_device_qf;
	QF host_qf;
	uint64_t nslots = 1ULL << nbits;
	//initialize global counters
	num_slots = nslots;
	total_items = 0;
	//n_inserts = 0;
	qf_malloc(&host_qf, nslots, num_hash_bits, 0, QF_HASH_DEFAULT, false, 0);
	qf_set_auto_resize(&host_qf, false);

	qfruntime* _runtime;
	qfmetadata* _metadata;
	qfblock* _blocks;

	cudaMalloc((void**)&_runtime, sizeof(qfruntime));
	cudaMalloc((void**)&_metadata, sizeof(qfmetadata));
	cudaMalloc((void**)&_blocks, qf_get_total_size_in_bytes(&host_qf));

	cudaMemcpy(_runtime, host_qf.runtimedata, sizeof(qfruntime), cudaMemcpyHostToDevice);
	cudaMemcpy(_metadata, host_qf.metadata, sizeof(qfmetadata), cudaMemcpyHostToDevice);
	cudaMemcpy(_blocks, host_qf.blocks, qf_get_total_size_in_bytes(&host_qf), cudaMemcpyHostToDevice);
	
	temp_device_qf.runtimedata = _runtime;
	temp_device_qf.metadata = _metadata;
	temp_device_qf.blocks = _blocks;

	//this might be buggy
	cudaMalloc((void **)&g_quotient_filter, sizeof(QF));
	cudaMemcpy(g_quotient_filter, &temp_device_qf, sizeof(QF), cudaMemcpyHostToDevice);



	num_locks = gqf_xnslots()/NUM_SLOTS_TO_LOCK+10;

	cudaMalloc((void **) & buffer_sizes, 20*num_locks*sizeof(uint64_t));
	

	cudaMalloc((void **)&buffers, 20*num_locks*sizeof(uint64_t*));

	cudaMalloc((void **)& buffer_backing, buf_size*sizeof(uint64_t));


	return 0;
}

extern inline int gqf_insert(uint64_t val, uint64_t count)
{
	qf_insert(g_quotient_filter, val, 0, count, QF_NO_LOCK);
	return 0;
}

extern inline int gqf_lookup(uint64_t val)
{
	return qf_count_key_value(g_quotient_filter, val, 0, 0);
}

extern inline uint64_t gqf_range()
{
	//fix me im on device
	//have to deep copy

	QF* host_qf;

	cudaMallocHost((void **)&host_qf, sizeof(QF));

	cudaMemcpy(host_qf, g_quotient_filter, sizeof(QF), cudaMemcpyDeviceToHost);

	uint64_t range;
	qfmetadata* _metadata;
	cudaMallocHost((void **)&_metadata, sizeof(qfmetadata));
	cudaMemcpy(_metadata, host_qf->metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);
	range = _metadata->range;

	cudaFreeHost(_metadata);
	cudaFreeHost(host_qf);
	return range;
}

extern inline uint64_t gqf_xnslots()
{
	//fix me im on device
	//have to deep copy

	QF* host_qf;

	cudaMallocHost((void **)&host_qf, sizeof(QF));

	cudaMemcpy(host_qf, g_quotient_filter, sizeof(QF), cudaMemcpyDeviceToHost);

	uint64_t range;
	qfmetadata* _metadata;
	cudaMallocHost((void **)&_metadata, sizeof(qfmetadata));
	cudaMemcpy(_metadata, host_qf->metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);
	range = _metadata->xnslots;

	cudaFreeHost(_metadata);
	cudaFreeHost(host_qf);
	return range;
}

extern inline int gqf_destroy()
{
	//fix me this isn't going to work
	free_buffers_premalloced(g_quotient_filter, buffers, buffer_backing, buffer_sizes, num_locks);
	qf_free_gpu(g_quotient_filter);
	

	return 0;
}

extern inline int gqf_iterator(uint64_t pos)
{
	qf_iterator_from_position(g_quotient_filter, &g_quotient_filter_itr, pos);
	return 0;
}

/* Returns 0 if the iterator is still valid (i.e. has not reached the
 * end of the QF. */
extern inline int gqf_get(uint64_t *key, uint64_t *value, uint64_t *count)
{
	return qfi_get_hash(&g_quotient_filter_itr, key, value, count);
}

/* Advance to next entry.  Returns whether or not another entry is
 * found.  */
extern inline int gqf_next()
{
	return qfi_next(&g_quotient_filter_itr);
}

/* Check to see if the if the end of the QF */
extern inline int gqf_end()
{
	return qfi_end(&g_quotient_filter_itr);
}

extern inline int gqf_bulk_insert(uint64_t * vals, uint64_t count)
{

	//calculate ratios
	//total_items += count;

	//int ratio = num_slots/total_items;

	//no sense in inflating the locks 200x for one insert
	//if (ratio > 15) ratio = 15;
	//printf("Dividing ratio %d\n", ratio);

	//cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t));
	
  //cudaMemset((uint64_t *) buffer_sizes, 0, ratio*num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_one_hash(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  //bulk_insert_bucketing_buffer_provided_timed(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	bulk_insert_no_atomics(g_quotient_filter, vals,0,1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_sizes);

	cudaDeviceSynchronize();
	return 0;
}

extern inline uint64_t gqf_bulk_get(uint64_t * vals, uint64_t count){

  

  return bulk_get_wrapper(g_quotient_filter, vals, count);

}

//replace vals with a cudaMalloced Array for gpu inserts
extern inline uint64_t * gqf_prep_vals(__uint128_t * vals, uint64_t count){


	uint64_t *hostvals;
	//= (uint64_t * ) calloc(count, sizeof(uint64_t));
	cudaMallocManaged((void **)&hostvals, count*sizeof(uint64_t));

	for (uint64_t i=0; i < count; i++){
		hostvals[i] = vals[i];
	}

	return hostvals;
}

#endif
