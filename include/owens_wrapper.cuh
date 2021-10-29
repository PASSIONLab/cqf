/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */

#ifndef OWENS_WRAPPER_CUH
#define OWENS_WRAPPER_CUH

#define INSERT_VERSION_BULK

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "quotientFilter.cuh"
#include <climits>

struct owens_filter::quotient_filter owens_cqf_gpu;

unsigned int * owens_inserts;
unsigned int * owens_returns;




#ifndef NUM_SLOTS_TO_LOCK
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#endif

extern inline uint64_t owens_xnslots();

extern inline int owens_init(uint64_t nbits, uint64_t num_hash_bits, uint64_t buf_size)
{

	//seems that we need to fix something here
	//p qf->metadata->value_bits is 0, idx why
	cudaMalloc((void **)& owens_inserts, sizeof(unsigned int)*buf_size);
	cudaMalloc((void **)& owens_returns, sizeof(int)*buf_size);


	//__host__ void initCQFGPU(struct countingQuotientFilterGPU *cqf, unsigned int q);
	owens_filter::initFilterGPU(&owens_cqf_gpu, nbits, 5);
	//initCQFGPU(&test_cqf_gpu, nbits);

	return 0;
}

extern inline int owens_insert(uint64_t val, uint64_t count)
{
	//qf_insert(g_quotient_filter, val, 0, count, QF_NO_LOCK);
	return 0;
}

extern inline int owens_lookup(uint64_t val)
{

	return 0;
	//return qf_count_key_value(g_quotient_filter, val, 0, 0);
}

extern inline uint64_t owens_range()
{
	//fix me im on device
	//have to deep copy

	
	return 0;
}

extern inline uint64_t owens_xnslots()
{
	//fix me im on device
	//have to deep copy

	return 0;
}

extern inline int owens_destroy()
{
	//since its a struct I don't think we do anything?
	//the orignal code has no ~Filter 
	//I'll write my own if its a problem - this is a memory leak but it may not matter :D

	return 0;
}

extern inline int owens_iterator(uint64_t pos)
{
	//qf_iterator_from_position(g_quotient_filter, &g_quotient_filter_itr, pos);
	return 0;
}

/* Returns 0 if the iterator is still valid (i.e. has not reached the
 * end of the QF. */
extern inline int owens_get(uint64_t *key, uint64_t *value, uint64_t *count)
{
	return 0; //qfi_get_hash(&g_quotient_filter_itr, key, value, count);
}

/* Advance to next entry.  Returns whether or not another entry is
 * found.  */
extern inline int owens_next()
{
	return 0;
}

/* Check to see if the if the end of the QF */
extern inline int owens_end()
{
	return 0;
}


__global__ void owens_downcast(uint64_t nitems, uint64_t * src, unsigned int * dst){


	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= nitems) return;


	dst[tid] = src[tid];


}

extern inline int owens_bulk_insert(uint64_t * vals, uint64_t count)
{

	//calculate ratios
	//total_items += count;

	//int ratio = num_slots/total_items;

	//no sense in inflating the locks 200x for one insert
	//if (ratio > 15) ratio = 15;
	//printf("Dividing ratio %d\n", ratio);

	
	owens_downcast<<<(count-1)/512+1, 512>>>(count, vals, owens_inserts);


  //insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);

	//__host__ float insertGPU(countingQuotientFilterGPU cqf, int numValues, unsigned int* d_insertValues, int* d_returnValues);

	//will this work?


	owens_filter::bulkBuildSegmentedLayouts(owens_cqf_gpu, count, owens_inserts, false);
	//owens_filter::insert(owens_cqf_gpu, count, owens_inserts);
	//insertGPU(owens_cqf_gpu, count, owens_inserts, owens_returns);
	
  //cudaMemset((uint64_t *) buffer_sizes, 0, ratio*num_locks*sizeof(uint64_t));
	//bulk_insert_bucketing_buffer_provided(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_one_hash(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK/ratio, num_locks*ratio, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  //bulk_insert_bucketing_buffer_provided_timed(g_quotient_filter, vals, 0, 1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
	//bulk_insert_no_atomics(g_quotient_filter, vals,0,1, count, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_sizes);

	cudaDeviceSynchronize();
	return 0;
}


__global__ void owens_check(unsigned int * returns, uint64_t count, uint64_t * misses){

	uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid >= count) return;

	if (returns[tid] == UINT_MAX){
		atomicAdd((unsigned long long int *) misses, (unsigned long long int) 1);
	}

}

extern inline uint64_t owens_bulk_get(uint64_t * vals, uint64_t count){


	uint64_t * misses;
  cudaMallocManaged((void **)& misses, sizeof(uint64_t));
  misses[0] = 0;

  owens_downcast<<<(count-1)/512+1, 512>>>(count, vals, owens_inserts);
  //return bulk_get_wrapper(g_quotient_filter, vals, count);

  owens_filter::launchSortedLookups(owens_cqf_gpu, count, owens_inserts, owens_returns);

  //launchLookups(test_cqf_gpu, count, owens_inserts, owens_returns);
  cudaDeviceSynchronize();


  owens_check<<<(count-1)/512+1, 512>>>(owens_returns, count, misses);
  cudaDeviceSynchronize();

  uint64_t toReturn = misses[0];

  cudaFree(misses);

  return toReturn;


}

//replace vals with a cudaMalloced Array for gpu inserts
extern inline uint64_t * owens_prep_vals(__uint128_t * vals, uint64_t count){


	uint64_t *hostvals;
	//= (uint64_t * ) calloc(count, sizeof(uint64_t));
	cudaMallocManaged((void **)&hostvals, count*sizeof(uint64_t));

	for (uint64_t i=0; i < count; i++){
		hostvals[i] = vals[i];
	}

	return hostvals;
}

#endif
