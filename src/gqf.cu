/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>
 *
 * ============================================================================
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

//timing stuff
#include <chrono>
#include <iostream>
#include <cmath>


//how fast is a thrust sort?
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


#include "hashutil.cuh"
#include "gqf.cuh"
#include "gqf_int.cuh"

#include <cuda_profiler_api.h>


/******************************************************************
 * Code for managing the metadata bits and slots w/o interpreting *
 * the content of the slots.
 ******************************************************************/

__device__ uint64_t memmove_cycle_counter = 0;
__device__ uint64_t total_cycle_counter = 0;

#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)                                    \
  ((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#define EXP_BEFORE_FAILURE -15
#define CLUSTER_SIZE (1ULL<<14)
#define METADATA_WORD(qf,field,slot_index)                              \
  (get_block((qf), (slot_index) /   QF_SLOTS_PER_BLOCK)->field[((slot_index)  % QF_SLOTS_PER_BLOCK) / 64])

#define GET_NO_LOCK(flag) (flag & QF_NO_LOCK)
#define GET_TRY_ONCE_LOCK(flag) (flag & QF_TRY_ONCE_LOCK)
#define GET_WAIT_FOR_LOCK(flag) (flag & QF_WAIT_FOR_LOCK)
#define GET_KEY_HASH(flag) (flag & QF_KEY_IS_HASH)

#define NUM_BUFFERS 10
#define MAX_BUFFER_SIZE 100

#define CYCLES_PER_SECOND 1601000000

#define MAX_DEPTH 16
#define SELECT_BOUND 32

#define DISTANCE_FROM_HOME_SLOT_CUTOFF 1000
#define BILLION 1000000000L
#define CUDA_CHECK(ans)                                                                  \
        gpuAssert((ans), __FILE__, __LINE__);
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
}
}

#ifdef DEBUG
#define PRINT_DEBUG 1
#else
#define PRINT_DEBUG 0
#endif

#define DEBUG_CQF(fmt, ...) \
	do { if (PRINT_DEBUG) printf( fmt, __VA_ARGS__); } while (0)

#define DEBUG_DUMP(qf) \
	do { if (PRINT_DEBUG) qf_dump_metadata(qf); } while (0)


#if QF_BITS_PER_SLOT > 0
__host__ __device__ static inline qfblock* get_block(const QF* qf, uint64_t block_index)
{
	return &qf->blocks[block_index];
}
#else
__host__ __device__ static inline qfblock* get_block(const QF* qf, uint64_t block_index)
{
	return (qfblock*)(((char*)qf->blocks)
		+ block_index * (sizeof(qfblock) + QF_SLOTS_PER_BLOCK *
			qf->metadata->bits_per_slot / 8));
}
#endif
/*
__device__ static __inline__ unsigned long long rdtsc(void)
{
	unsigned hi, lo;
	__asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
	return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
*/
/*
__host__ __device__ static void modify_metadata(pc_t *metadata, int cnt)
{
	pc_add(metadata, cnt);
	return;
}
*/
/*changing sizes of register based on https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
l is for "l" = .u64 reg
*/
__host__ __device__ static inline int popcnt(uint64_t val)
{
#ifdef __CUDA_ARCH__
	val = __popcll(val);
#else

	#ifndef __x86_64
		val = __builtin_popcount(val);

	#else
		
		asm("popcnt %[val], %[val]"
			: [val] "+r" (val)
			:
			: "cc");
		
	#endif

#endif
	return val;
}

__device__ static inline int64_t bitscanreverse(uint64_t val)
{
	if (val == 0) {
		return -1;
	} else {

		asm("bsr %[val], %[val]"
			: [val] "+l" (val)
			:
			: );
		return val;
	}
}

__host__ __device__ static inline int popcntv(const uint64_t val, int ignore)
{
	if (ignore % 64)
		return popcnt (val & ~BITMASK(ignore % 64));
	else
		return popcnt(val);
}

// Returns the number of 1s up to (and including) the pos'th bit
// Bits are numbered from 0
__host__ __device__ static inline int bitrank(uint64_t val, int pos) {
	val = val & ((2ULL << pos) - 1);
#ifdef __CUDA_ARCH__
	val = __popcll(val);
#else

	//quick fix for summit

	#ifndef __x86_64

		val = __builtin_popcount(val);

	#else

		
		asm("popcnt %[val], %[val]"
			: [val] "+r" (val)
			:
			: "cc");

	#endif
		


#endif
	return val;
}

//moved dump functions
__host__ __device__ static inline void qf_dump_block(const QF *qf, uint64_t i)
{
	uint64_t j;

	printf("Block %llu Runs from %llu to %llu\n",i,  i*QF_SLOTS_PER_BLOCK, (i+1)*QF_SLOTS_PER_BLOCK);
	printf("Offset: %-192d", get_block(qf, i)->offset);
	printf("\n");

	for (j = 0; j < QF_SLOTS_PER_BLOCK; j++)
		printf("%02lx ", j);
	printf("\n");

	for (j = 0; j < QF_SLOTS_PER_BLOCK; j++)
		printf(" %d ", (get_block(qf, i)->occupieds[j/64] & (1ULL << (j%64))) ? 1 : 0);
	printf("\n");

	for (j = 0; j < QF_SLOTS_PER_BLOCK; j++)
		printf(" %d ", (get_block(qf, i)->runends[j/64] & (1ULL << (j%64))) ? 1 : 0);
	printf("\n");

#if QF_BITS_PER_SLOT == 8 || QF_BITS_PER_SLOT == 16 || QF_BITS_PER_SLOT == 32
	for (j = 0; j < QF_SLOTS_PER_BLOCK; j++)
		printf("%02x ", get_block(qf, i)->slots[j]);
#elif QF_BITS_PER_SLOT == 64
	for (j = 0; j < QF_SLOTS_PER_BLOCK; j++)
		printf("%02lx ", get_block(qf, i)->slots[j]);
#else
	for (j = 0; j < QF_SLOTS_PER_BLOCK * qf->metadata->bits_per_slot / 8; j++)
		printf("%02x ", get_block(qf, i)->slots[j]);
#endif

	printf("\n");

	printf("\n");
}

__host__ __device__ void qf_dump_metadata(const QF *qf) {
	printf("Slots: %lu Occupied: %lu Elements: %lu Distinct: %lu\n",
				 qf->metadata->nslots,
				 qf->metadata->noccupied_slots,
				 qf->metadata->nelts,
				 qf->metadata->ndistinct_elts);
	printf("Key_bits: %lu Value_bits: %lu Remainder_bits: %lu Bits_per_slot: %lu\n",
				 qf->metadata->key_bits,
				 qf->metadata->value_bits,
				 qf->metadata->key_remainder_bits,
				 qf->metadata->bits_per_slot);
}

__host__ __device__ void qf_dump(const QF *qf)
{
	uint64_t i;

	printf("%lu %lu %lu\n",
				 qf->metadata->nblocks,
				 qf->metadata->ndistinct_elts,
				 qf->metadata->nelts);

	for (i = 0; i < qf->metadata->nblocks; i++) {
		qf_dump_block(qf, i);
	}

}

/**
 * Returns the position of the k-th 1 in the 64-bit word x.
 * k is 0-based, so k=0 returns the position of the first 1.
 *
 * Uses the broadword selection algorithm by Vigna [1], improved by Gog
 * and Petri [2] and Vigna [3].
 *
 * [1] Sebastiano Vigna. Broadword Implementation of Rank/Select
 *    Queries. WEA, 2008
 *
 * [2] Simon Gog, Matthias Petri. Optimized succinct data
 * structures for massive data. Softw. Pract. Exper., 2014
 *
 * [3] Sebastiano Vigna. MG4J 5.2.1. http://mg4j.di.unimi.it/
 * The following code is taken from
 * https://github.com/facebook/folly/blob/b28186247104f8b90cfbe094d289c91f9e413317/folly/experimental/Select64.h
 */
 __device__ __constant__ uint8_t gpukSelectInByte[2048] = {
	8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
	1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
	2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
	1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
	3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0,
	1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
	2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
	1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
	4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0,
	1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 8, 8, 1,
	8, 2, 2, 1, 8, 3, 3, 1, 3, 2, 2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2,
	2, 1, 8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1,
	4, 3, 3, 1, 3, 2, 2, 1, 8, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4,
	4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1,
	3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 7, 7, 1, 7, 2,
	2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
	7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3,
	3, 1, 3, 2, 2, 1, 7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1,
	4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2,
	2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 8, 8, 8, 8, 8, 8, 2,
	8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3, 4, 3, 3, 2, 8, 8,
	8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3,
	4, 3, 3, 2, 8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4,
	4, 2, 6, 4, 4, 3, 4, 3, 3, 2, 8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2,
	6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 7, 8, 7, 7, 2, 8, 7,
	7, 3, 7, 3, 3, 2, 8, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2, 8, 7, 7, 5,
	7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3,
	3, 2, 8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2,
	6, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5,
	5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3, 8, 8, 8, 8, 8, 8,
	8, 5, 8, 8, 8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3,
	8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6,
	6, 4, 6, 4, 4, 3, 8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5,
	6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7,
	7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3, 8, 8, 8, 7, 8, 7, 7, 5,
	8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3, 8, 8,
	8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4,
	6, 4, 4, 3, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5,
	5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 4, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6,
	6, 4, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6, 8, 6, 6, 5,
	8, 6, 6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8,
	8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7,
	8, 7, 7, 5, 8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4, 8, 8, 8, 8, 8, 8,
	8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4,
	8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6,
	6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6,
	8, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
	8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 8,
	8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6,
	6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
	8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7
};

 const uint8_t hostkSelectInByte[2048] = {
   8, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0,
   1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0,
   2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0,
   1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0,
   3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 7, 0,
   1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
   2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0,
   1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 6, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0,
   4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 5, 0, 1, 0, 2, 0, 1, 0, 3, 0,
   1, 0, 2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 8, 8, 8, 1,
   8, 2, 2, 1, 8, 3, 3, 1, 3, 2, 2, 1, 8, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2,
   2, 1, 8, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1,
   4, 3, 3, 1, 3, 2, 2, 1, 8, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4,
   4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1,
   3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 7, 7, 1, 7, 2,
   2, 1, 7, 3, 3, 1, 3, 2, 2, 1, 7, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1,
   7, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2, 2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3,
   3, 1, 3, 2, 2, 1, 7, 6, 6, 1, 6, 2, 2, 1, 6, 3, 3, 1, 3, 2, 2, 1, 6, 4, 4, 1,
   4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 6, 5, 5, 1, 5, 2, 2, 1, 5, 3, 3, 1, 3, 2,
   2, 1, 5, 4, 4, 1, 4, 2, 2, 1, 4, 3, 3, 1, 3, 2, 2, 1, 8, 8, 8, 8, 8, 8, 8, 2,
   8, 8, 8, 3, 8, 3, 3, 2, 8, 8, 8, 4, 8, 4, 4, 2, 8, 4, 4, 3, 4, 3, 3, 2, 8, 8,
   8, 5, 8, 5, 5, 2, 8, 5, 5, 3, 5, 3, 3, 2, 8, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3,
   4, 3, 3, 2, 8, 8, 8, 6, 8, 6, 6, 2, 8, 6, 6, 3, 6, 3, 3, 2, 8, 6, 6, 4, 6, 4,
   4, 2, 6, 4, 4, 3, 4, 3, 3, 2, 8, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2,
   6, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 7, 8, 7, 7, 2, 8, 7,
   7, 3, 7, 3, 3, 2, 8, 7, 7, 4, 7, 4, 4, 2, 7, 4, 4, 3, 4, 3, 3, 2, 8, 7, 7, 5,
   7, 5, 5, 2, 7, 5, 5, 3, 5, 3, 3, 2, 7, 5, 5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3,
   3, 2, 8, 7, 7, 6, 7, 6, 6, 2, 7, 6, 6, 3, 6, 3, 3, 2, 7, 6, 6, 4, 6, 4, 4, 2,
   6, 4, 4, 3, 4, 3, 3, 2, 7, 6, 6, 5, 6, 5, 5, 2, 6, 5, 5, 3, 5, 3, 3, 2, 6, 5,
   5, 4, 5, 4, 4, 2, 5, 4, 4, 3, 4, 3, 3, 2, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 3, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 4, 8, 4, 4, 3, 8, 8, 8, 8, 8, 8,
   8, 5, 8, 8, 8, 5, 8, 5, 5, 3, 8, 8, 8, 5, 8, 5, 5, 4, 8, 5, 5, 4, 5, 4, 4, 3,
   8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 3, 8, 8, 8, 6, 8, 6, 6, 4, 8, 6,
   6, 4, 6, 4, 4, 3, 8, 8, 8, 6, 8, 6, 6, 5, 8, 6, 6, 5, 6, 5, 5, 3, 8, 6, 6, 5,
   6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7,
   7, 3, 8, 8, 8, 7, 8, 7, 7, 4, 8, 7, 7, 4, 7, 4, 4, 3, 8, 8, 8, 7, 8, 7, 7, 5,
   8, 7, 7, 5, 7, 5, 5, 3, 8, 7, 7, 5, 7, 5, 5, 4, 7, 5, 5, 4, 5, 4, 4, 3, 8, 8,
   8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 3, 8, 7, 7, 6, 7, 6, 6, 4, 7, 6, 6, 4,
   6, 4, 4, 3, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6, 6, 5, 6, 5, 5, 3, 7, 6, 6, 5, 6, 5,
   5, 4, 6, 5, 5, 4, 5, 4, 4, 3, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 5, 8, 5, 5, 4, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6,
   6, 4, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6, 8, 6, 6, 5, 8, 8, 8, 6, 8, 6, 6, 5,
   8, 6, 6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8,
   8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 4, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7,
   8, 7, 7, 5, 8, 8, 8, 7, 8, 7, 7, 5, 8, 7, 7, 5, 7, 5, 5, 4, 8, 8, 8, 8, 8, 8,
   8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 4,
   8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6, 6, 5, 8, 7, 7, 6, 7, 6, 6, 5, 7, 6,
   6, 5, 6, 5, 5, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 5, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 8, 8, 8, 8, 6, 8, 8, 8, 6,
   8, 6, 6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7,
   8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 8,
   8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8, 8, 7, 8, 7, 7, 6, 8, 7, 7, 6, 7, 6,
   6, 5, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 7, 7, 6, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7
 };
__host__ __device__ static inline uint64_t _select64(uint64_t x, int k)
{
	if (k >= popcnt(x)) { return 64; }

	const uint64_t kOnesStep4  = 0x1111111111111111ULL;
	const uint64_t kOnesStep8  = 0x0101010101010101ULL;
	const uint64_t kMSBsStep8  = 0x80ULL * kOnesStep8;

	uint64_t s = x;
	s = s - ((s & 0xA * kOnesStep4) >> 1);
	s = (s & 0x3 * kOnesStep4) + ((s >> 2) & 0x3 * kOnesStep4);
	s = (s + (s >> 4)) & 0xF * kOnesStep8;
	uint64_t byteSums = s * kOnesStep8;

	uint64_t kStep8 = k * kOnesStep8;
	uint64_t geqKStep8 = (((kStep8 | kMSBsStep8) - byteSums) & kMSBsStep8);
	uint64_t place = popcnt(geqKStep8) * 8;
	uint64_t byteRank = k - (((byteSums << 8) >> place) & (uint64_t)(0xFF));
#ifdef __CUDA_ARCH__
	return place + gpukSelectInByte[((x >> place) & 0xFF) | (byteRank << 8)];
#else
	return place + hostkSelectInByte[((x >> place) & 0xFF) | (byteRank << 8)];
#endif // __CUDA_ARCH__


}

// Returns the position of the rank'th 1.  (rank = 0 returns the 1st 1)
// Returns 64 if there are fewer than rank+1 1s.
__host__ __device__ static inline uint64_t bitselect(uint64_t val, int rank) {
#ifdef __SSE4_2_
	uint64_t i = 1ULL << rank;
	asm("pdep %[val], %[mask], %[val]"
			: [val] "+r" (val)
			: [mask] "r" (i));
	asm("tzcnt %[bit], %[index]"
			: [index] "=r" (i)
			: [bit] "g" (val)
			: "cc");
	return i;
#endif
	return _select64(val, rank);
}

__host__ __device__ static inline uint64_t bitselectv(const uint64_t val, int ignore, int rank)
{
	return bitselect(val & ~BITMASK(ignore % 64), rank);
}

__host__ __device__ static inline int is_runend(const QF *qf, uint64_t index)
{
	return (METADATA_WORD(qf, runends, index) >> ((index % QF_SLOTS_PER_BLOCK) %
																								64)) & 1ULL;
}

__host__ __device__ static inline int is_occupied(const QF *qf, uint64_t index)
{
	return (METADATA_WORD(qf, occupieds, index) >> ((index % QF_SLOTS_PER_BLOCK) %
																									64)) & 1ULL;
}

#if QF_BITS_PER_SLOT == 8 || QF_BITS_PER_SLOT == 16 || QF_BITS_PER_SLOT == 32 || QF_BITS_PER_SLOT == 64

__host__ __device__ static inline uint64_t get_slot(const QF *qf, uint64_t index)
{
  //ERR: Index passed in is incorrect
	//printf("slots %lu, index %lu\n", qf->metadata->nslots, index);
	assert(index < qf->metadata->xnslots);
	return get_block(qf, index / QF_SLOTS_PER_BLOCK)->slots[index % QF_SLOTS_PER_BLOCK];
}

__host__ __device__ static inline void set_slot(const QF *qf, uint64_t index, uint64_t value)
{
	assert(index < qf->metadata->xnslots);
	get_block(qf, index / QF_SLOTS_PER_BLOCK)->slots[index % QF_SLOTS_PER_BLOCK] =
		value & BITMASK(qf->metadata->bits_per_slot);
}

#elif QF_BITS_PER_SLOT > 0

/* Little-endian code ....  Big-endian is TODO */

__host__ __device__ static inline uint64_t get_slot(const QF *qf, uint64_t index)
{
	/* Should use __uint128_t to support up to 64-bit remainders, but gcc seems
	 * to generate buggy code.  :/  */
  //printf("Other get slot: slots %lu, index %lu\n", qf->metadata->nslots, index);
	assert(index < qf->metadata->xnslots);
	uint64_t *p = (uint64_t *)&get_block(qf, index /
																			 QF_SLOTS_PER_BLOCK)->slots[(index %
																																QF_SLOTS_PER_BLOCK)
																			 * QF_BITS_PER_SLOT / 8];
	return (uint64_t)(((*p) >> (((index % QF_SLOTS_PER_BLOCK) * QF_BITS_PER_SLOT) %
															8)) & BITMASK(QF_BITS_PER_SLOT));
}

__host__ __device__ static inline void set_slot(const QF *qf, uint64_t index, uint64_t value)
{
	/* Should use __uint128_t to support up to 64-bit remainders, but gcc seems
	 * to generate buggy code.  :/  */
	assert(index < qf->metadata->xnslots);
	uint64_t *p = (uint64_t *)&get_block(qf, index /
																			 QF_SLOTS_PER_BLOCK)->slots[(index %
																																QF_SLOTS_PER_BLOCK)
																			 * QF_BITS_PER_SLOT / 8];
	uint64_t t = *p;
	uint64_t mask = BITMASK(QF_BITS_PER_SLOT);
	uint64_t v = value;
	int shift = ((index % QF_SLOTS_PER_BLOCK) * QF_BITS_PER_SLOT) % 8;
	mask <<= shift;
	v <<= shift;
	t &= ~mask;
	t |= v;
	*p = t;
}

#else

/* Little-endian code ....  Big-endian is TODO */

__host__ __device__ static inline uint64_t get_slot(const QF *qf, uint64_t index)
{
  //rintf("Third get slot?!? slots %lu, index %lu\n", qf->metadata->nslots, index);
	assert(index < qf->metadata->xnslots);
	/* Should use __uint128_t to support up to 64-bit remainders, but gcc seems
	 * to generate buggy code.  :/  */
	uint64_t *p = (uint64_t *)&get_block(qf, index / QF_SLOTS_PER_BLOCK)->slots[(index %QF_SLOTS_PER_BLOCK)* qf->metadata->bits_per_slot / 8];
	return (uint64_t)(((*p) >> (((index % QF_SLOTS_PER_BLOCK) *qf->metadata->bits_per_slot) % 8)) & BITMASK(qf->metadata->bits_per_slot));
}

__host__ __device__ static inline void set_slot(const QF *qf, uint64_t index, uint64_t value)
{
	assert(index < qf->metadata->xnslots);
	/* Should use __uint128_t to support up to 64-bit remainders, but gcc seems
	 * to generate buggy code.  :/  */
	uint64_t *p = (uint64_t *)&get_block(qf, index /QF_SLOTS_PER_BLOCK)->slots[(index %QF_SLOTS_PER_BLOCK)* qf->metadata->bits_per_slot / 8];
	uint64_t t = *p;
	uint64_t mask = BITMASK(qf->metadata->bits_per_slot);
	uint64_t v = value;
	int shift = ((index % QF_SLOTS_PER_BLOCK) * qf->metadata->bits_per_slot) % 8;
	mask <<= shift;
	v <<= shift;
	t &= ~mask;
	t |= v;
	*p = t;
}

#endif

__host__ __device__ static inline uint64_t run_end(const QF *qf, uint64_t hash_bucket_index);

__host__ __device__ static inline uint64_t block_offset(const QF *qf, uint64_t blockidx)
{
	/* If we have extended counters and a 16-bit (or larger) offset
		 field, then we can safely ignore the possibility of overflowing
		 that field. */
	if (sizeof(qf->blocks[0].offset) > 1 ||
			get_block(qf, blockidx)->offset < BITMASK(8*sizeof(qf->blocks[0].offset)))
		return get_block(qf, blockidx)->offset;

	return run_end(qf, QF_SLOTS_PER_BLOCK * blockidx - 1) - QF_SLOTS_PER_BLOCK *
		blockidx + 1;
}

__host__ __device__ static inline uint64_t run_end(const QF *qf, uint64_t hash_bucket_index)
{
	uint64_t bucket_block_index       = hash_bucket_index / QF_SLOTS_PER_BLOCK;
	uint64_t bucket_intrablock_offset = hash_bucket_index % QF_SLOTS_PER_BLOCK;
	uint64_t bucket_blocks_offset = block_offset(qf, bucket_block_index);

	uint64_t bucket_intrablock_rank   = bitrank(get_block(qf, bucket_block_index)->occupieds[0], bucket_intrablock_offset);

	if (bucket_intrablock_rank == 0) {
		if (bucket_blocks_offset <= bucket_intrablock_offset)
			return hash_bucket_index;
		else
			return QF_SLOTS_PER_BLOCK * bucket_block_index + bucket_blocks_offset - 1;
	}

	uint64_t runend_block_index  = bucket_block_index + bucket_blocks_offset /
		QF_SLOTS_PER_BLOCK;
	uint64_t runend_ignore_bits  = bucket_blocks_offset % QF_SLOTS_PER_BLOCK;
	uint64_t runend_rank         = bucket_intrablock_rank - 1;
	uint64_t runend_block_offset = bitselectv(get_block(qf,
																						runend_block_index)->runends[0],
																						runend_ignore_bits, runend_rank);
	if (runend_block_offset == QF_SLOTS_PER_BLOCK) {
		if (bucket_blocks_offset == 0 && bucket_intrablock_rank == 0) {
			/* The block begins in empty space, and this bucket is in that region of
			 * empty space */
			return hash_bucket_index;
		} else {
			do {
				runend_rank        -= popcntv(get_block(qf,
																								runend_block_index)->runends[0],
																			runend_ignore_bits);
				runend_block_index++;
				runend_ignore_bits  = 0;
				runend_block_offset = bitselectv(get_block(qf,
																									 runend_block_index)->runends[0],
																				 runend_ignore_bits, runend_rank);
			} while (runend_block_offset == QF_SLOTS_PER_BLOCK);
		}
	}

	uint64_t runend_index = QF_SLOTS_PER_BLOCK * runend_block_index +
		runend_block_offset;
	if (runend_index < hash_bucket_index)
		return hash_bucket_index;
	else
		return runend_index;
}

__host__ __device__ static inline int offset_lower_bound(const QF *qf, uint64_t slot_index)
{
	const qfblock * b = get_block(qf, slot_index / QF_SLOTS_PER_BLOCK);
	const uint64_t slot_offset = slot_index % QF_SLOTS_PER_BLOCK;
	const uint64_t boffset = b->offset;
	const uint64_t occupieds = b->occupieds[0] & BITMASK(slot_offset+1);

	//printf("slot %llu, slot_offset %02lx, block offset %llu, occupieds: %d ", slot_index, slot_offset, boffset, popcnt(occupieds));
	assert(QF_SLOTS_PER_BLOCK == 64);

	//if (boffset < slot_offset) {
	if (boffset <= slot_offset) {
		const uint64_t runends = (b->runends[0] & BITMASK(slot_offset)) >> boffset;
		//printf(" runends %d\n", popcnt(runends));
		//printf("boffset < slot_offset, runends %llu, popcnt(occupieds) %d, popcnt(runends) %d\n", runends, popcnt(occupieds), popcnt(runends));
		//printf("returning %d\n", popcnt(occupieds)-popcnt(runends));
		return popcnt(occupieds) - popcnt(runends);

	}
	//printf("\n");
	//printf("boffset > slot_offset, boffset-slotoffset %llu, popcnt(occupieds) %d\n", boffset-slot_offset, popcnt(occupieds));
	//printf("returning %d\n", boffset-slot_offset+popcnt(occupieds));
	return boffset - slot_offset + popcnt(occupieds);
}

__host__ __device__ static inline int offset_lower_bound_verbose(const QF *qf, uint64_t slot_index)
{
	const qfblock * b = get_block(qf, slot_index / QF_SLOTS_PER_BLOCK);
	const uint64_t slot_offset = slot_index % QF_SLOTS_PER_BLOCK;
	const uint64_t boffset = b->offset;
	const uint64_t occupieds = b->occupieds[0] & BITMASK(slot_offset+1);

	printf("slot %llu, slot_offset %02lx, block offset %llu, occupieds: %d ", slot_index, slot_offset, boffset, popcnt(occupieds));
	assert(QF_SLOTS_PER_BLOCK == 64);
	if (boffset <= slot_offset) {
		const uint64_t runends = (b->runends[0] & BITMASK(slot_offset)) >> boffset;
		printf(" runends %d\n", popcnt(runends));
		//printf("boffset < slot_offset, runends %llu, popcnt(occupieds) %d, popcnt(runends) %d\n", runends, popcnt(occupieds), popcnt(runends));
		printf("returning %d\n", popcnt(occupieds)-popcnt(runends));
		return popcnt(occupieds) - popcnt(runends);
	}
	printf("\n");
	//printf("boffset > slot_offset, boffset-slotoffset %llu, popcnt(occupieds) %d\n", boffset-slot_offset, popcnt(occupieds));
	printf("returning %d\n", boffset-slot_offset+popcnt(occupieds));
	return boffset - slot_offset + popcnt(occupieds);
}

__host__ __device__ static inline int is_empty(const QF *qf, uint64_t slot_index)
{
	return offset_lower_bound(qf, slot_index) == 0;
}

__host__ __device__ static inline int might_be_empty(const QF *qf, uint64_t slot_index)
{
	return !is_occupied(qf, slot_index)
		&& !is_runend(qf, slot_index);
}

__device__ static inline int probably_is_empty(const QF *qf, uint64_t slot_index)
{
	return get_slot(qf, slot_index) == 0
		&& !is_occupied(qf, slot_index)
		&& !is_runend(qf, slot_index);
}


__host__ __device__ static inline uint64_t find_first_empty_slot_verbose(QF *qf, uint64_t from)
{

	printf("Starting find first - this will terminate in -1\n");
	qf_dump_block(qf, from/QF_SLOTS_PER_BLOCK);
	do {
		int t = offset_lower_bound_verbose(qf, from);
    //get block of from

    if (t < 0){
    	
      printf("Finding first empty slot. T: %d, from: %llu\n - block %llu", t, from, from/QF_SLOTS_PER_BLOCK);
      qf_dump(qf);
    }
		assert(t>=0);
		if (t == 0)
			break;
		from = from + t;
	} while(1);
	printf("Next empty slot: %llu", from);
	return from;
}

__host__ __device__ static inline uint64_t find_first_empty_slot(QF *qf, uint64_t from)
{

	uint64_t start_from = from;

	do {
		int t = offset_lower_bound(qf, from);
    //get block of from

    // if (t < 0){

    // 	//this implies a failure in the code - you are going to 
    // 	find_first_empty_slot_verbose(qf, start_from);
  
    // }


		assert(t>=0);
		if (t == 0)
			break;
		from = from + t;
	} while(1);


	uint64_t bucket_start_from = start_from/NUM_SLOTS_TO_LOCK;
	uint64_t end_start_from = from/NUM_SLOTS_TO_LOCK;

	//testing without this gate to check if we see speed improvements
	if (end_start_from>bucket_start_from+1){
		printf("Find first empty ran over a bucket: %llu\n", end_start_from-bucket_start_from);
	}

	return from;
}



__host__ __device__ static inline uint64_t shift_into_b(const uint64_t a, const uint64_t b,
																		const int bstart, const int bend,
																		const int amount)
{
	const uint64_t a_component = bstart == 0 ? (a >> (64 - amount)) : 0;
	const uint64_t b_shifted_mask = BITMASK(bend - bstart) << bstart;
	const uint64_t b_shifted = ((b_shifted_mask & b) << amount) & b_shifted_mask;
	const uint64_t b_mask = ~b_shifted_mask;
	return a_component | b_shifted | (b & b_mask);
}

// __device__ void* gpu_memmove(void* dst, const void* src, size_t n)
// {
// 	//printf("Launching memmove\n");
// 	//todo: allocate space per thread for this buffer before launching the kernel
// 	void* temp_buffer = malloc(n);
// 	//maybe stack allocation?
// 	//void* temp_buffer = void* char[n];
// 	// cudaMemcpyAsync(temp_buffer, src, n, cudaMemcpyDeviceToDevice);
// 	// cudaMemcpyAsync(dst, temp_buffer, n, cudaMemcpyDeviceToDevice);
// 	// //cudaFree(temp_buffer);
// 	// return dst;
//   memcpy(temp_buffer, src, n);
//   memcpy(dst, temp_buffer, n);

//   free(temp_buffer);

// }


//a variant of memmove that compares the two pointers
__device__ void* gpu_memmove(void* dst, const void* src, size_t n)
{
	//printf("Launching memmove\n");
	//todo: allocate space per thread for this buffer before launching the kernel

	uint64_t start = clock64();

	char * char_dst = (char *) dst;
	char * char_src = (char *) src;

  //double check this,
  //think it is just > since dst+n does not get copied
  if (char_src+n > char_dst){

  	//copy backwards 
  	for (int i =n-1; i >= 0; i--){



  		char_dst[i] = char_src[i];

  	}

  } else {

  	//copy regular
  	for (int i =0; i<n; i++){
  		char_dst[i] = char_src[i];
  	}


  }

  uint64_t end = clock64();

  memmove_cycle_counter += end-start;

  //free(temp_buffer);

}


#if QF_BITS_PER_SLOT == 8 || QF_BITS_PER_SLOT == 16 || QF_BITS_PER_SLOT == 32 || QF_BITS_PER_SLOT == 64

__host__ __device__ static inline void shift_remainders(QF *qf, uint64_t start_index, uint64_t
																		empty_index)
{
	uint64_t start_block  = start_index / QF_SLOTS_PER_BLOCK;
	uint64_t start_offset = start_index % QF_SLOTS_PER_BLOCK;
	uint64_t empty_block  = empty_index / QF_SLOTS_PER_BLOCK;
	uint64_t empty_offset = empty_index % QF_SLOTS_PER_BLOCK;

	assert (start_index <= empty_index);
  assert (empty_index < qf->metadata->xnslots);

	while (start_block < empty_block) {
#ifdef __CUDA_ARCH__
		gpu_memmove(&get_block(qf, empty_block)->slots[1],
			&get_block(qf, empty_block)->slots[0],
			empty_offset * sizeof(qf->blocks[0].slots[0]));
#else
		memmove(&get_block(qf, empty_block)->slots[1],
			&get_block(qf, empty_block)->slots[0],
			empty_offset * sizeof(qf->blocks[0].slots[0]));
#endif

		get_block(qf, empty_block)->slots[0] = get_block(qf,
																			empty_block-1)->slots[QF_SLOTS_PER_BLOCK-1];
		empty_block--;
		empty_offset = QF_SLOTS_PER_BLOCK-1;
	}
#ifdef __CUDA_ARCH__
	gpu_memmove(&get_block(qf, empty_block)->slots[start_offset + 1],
		&get_block(qf, empty_block)->slots[start_offset],
		(empty_offset - start_offset) * sizeof(qf->blocks[0].slots[0]));
#else
	memmove(&get_block(qf, empty_block)->slots[start_offset+1],
					&get_block(qf, empty_block)->slots[start_offset],
					(empty_offset - start_offset) * sizeof(qf->blocks[0].slots[0]));
#endif
}

#else

#define REMAINDER_WORD(qf, i) ((uint64_t *)&(get_block(qf, (i)/qf->metadata->bits_per_slot)->slots[8 * ((i) % qf->metadata->bits_per_slot)]))

__host__ __device__ static inline void shift_remainders(QF *qf, const uint64_t start_index, const
																		uint64_t empty_index)
{
	uint64_t last_word = (empty_index + 1) * qf->metadata->bits_per_slot / 64;
	const uint64_t first_word = start_index * qf->metadata->bits_per_slot / 64;
	int bend = ((empty_index + 1) * qf->metadata->bits_per_slot) % 64;
	const int bstart = (start_index * qf->metadata->bits_per_slot) % 64;

	while (last_word != first_word) {
		*REMAINDER_WORD(qf, last_word) = shift_into_b(*REMAINDER_WORD(qf, last_word-1),
																									*REMAINDER_WORD(qf, last_word),
																									0, bend, qf->metadata->bits_per_slot);
		last_word--;
		bend = 64;
	}
	*REMAINDER_WORD(qf, last_word) = shift_into_b(0, *REMAINDER_WORD(qf,
																																	 last_word),
																								bstart, bend,
																								qf->metadata->bits_per_slot);
}

#endif



__host__ __device__ static inline void find_next_n_empty_slots(QF *qf, uint64_t from, uint64_t n,
																					 uint64_t *indices)
{
	while (n) {
		indices[--n] = find_first_empty_slot(qf, from);
		from = indices[n] + 1;
	}
}

__host__ __device__ static inline void shift_slots(QF *qf, int64_t first, uint64_t last, uint64_t
															 distance)
{
	int64_t i;
	if (distance == 1)
		shift_remainders(qf, first, last+1);
	else
		for (i = last; i >= first; i--)
			set_slot(qf, i + distance, get_slot(qf, i));
}

__host__ __device__ static inline void shift_runends(QF *qf, int64_t first, uint64_t last,
																 uint64_t distance)
{
	assert(last < qf->metadata->xnslots && distance < 64);
	uint64_t first_word = first / 64;
	uint64_t bstart = first % 64;
	uint64_t last_word = (last + distance + 1) / 64;
	uint64_t bend = (last + distance + 1) % 64;

	if (last_word != first_word) {
		METADATA_WORD(qf, runends, 64*last_word) = shift_into_b(METADATA_WORD(qf, runends, 64*(last_word-1)),
																														METADATA_WORD(qf, runends, 64*last_word),
																														0, bend, distance);
		bend = 64;
		last_word--;
		while (last_word != first_word) {
			METADATA_WORD(qf, runends, 64*last_word) = shift_into_b(METADATA_WORD(qf, runends, 64*(last_word-1)),
																															METADATA_WORD(qf, runends, 64*last_word),
																															0, bend, distance);
			last_word--;
		}
	}
	METADATA_WORD(qf, runends, 64*last_word) = shift_into_b(0, METADATA_WORD(qf,
																																					 runends,
																																					 64*last_word),
																													bstart, bend, distance);

}

__host__ __device__ static inline bool insert_replace_slots_and_shift_remainders_and_runends_and_offsets(QF		*qf,
																																										 int		 operation,
																																										 uint64_t		 bucket_index,
																																										 uint64_t		 overwrite_index,
																																										 const uint64_t	*remainders,
																																										 uint64_t		 total_remainders,
																																										 uint64_t		 noverwrites)
{
	uint64_t empties[67];
	uint64_t i;
	int64_t j;
	int64_t ninserts = total_remainders - noverwrites;
	uint64_t insert_index = overwrite_index + noverwrites;

	if (ninserts > 0) {
		/* First, shift things to create n empty spaces where we need them. */
		find_next_n_empty_slots(qf, insert_index, ninserts, empties);
		if (empties[0] >= qf->metadata->xnslots) {
			return false;
		}
		for (j = 0; j < ninserts - 1; j++)
			shift_slots(qf, empties[j+1] + 1, empties[j] - 1, j + 1);
		shift_slots(qf, insert_index, empties[ninserts - 1] - 1, ninserts);

		for (j = 0; j < ninserts - 1; j++)
			shift_runends(qf, empties[j+1] + 1, empties[j] - 1, j + 1);
		shift_runends(qf, insert_index, empties[ninserts - 1] - 1, ninserts);

		for (i = noverwrites; i < total_remainders - 1; i++)
			METADATA_WORD(qf, runends, overwrite_index + i) &= ~(1ULL <<
																													 (((overwrite_index
																															+ i) %
																														 QF_SLOTS_PER_BLOCK)
																														% 64));

		switch (operation) {
			case 0: /* insert into empty bucket */
				assert (noverwrites == 0);
				METADATA_WORD(qf, runends, overwrite_index + total_remainders - 1) |=
					1ULL << (((overwrite_index + total_remainders - 1) %
										QF_SLOTS_PER_BLOCK) % 64);
				break;
			case 1: /* append to bucket */
				METADATA_WORD(qf, runends, overwrite_index + noverwrites - 1)      &=
					~(1ULL << (((overwrite_index + noverwrites - 1) % QF_SLOTS_PER_BLOCK) %
										 64));
				METADATA_WORD(qf, runends, overwrite_index + total_remainders - 1) |=
					1ULL << (((overwrite_index + total_remainders - 1) %
										QF_SLOTS_PER_BLOCK) % 64);
				break;
			case 2: /* insert into bucket */
				METADATA_WORD(qf, runends, overwrite_index + total_remainders - 1) &=
					~(1ULL << (((overwrite_index + total_remainders - 1) %
											QF_SLOTS_PER_BLOCK) % 64));
				break;
			default:
				printf("Invalid operation %d\n", operation);
#ifdef __CUDA_ARCH__
				__threadfence();         // ensure store issued before trap
				asm("trap;");
#else
				abort();
#endif
		}

		uint64_t npreceding_empties = 0;
		for (i = bucket_index / QF_SLOTS_PER_BLOCK + 1; i <= empties[0]/QF_SLOTS_PER_BLOCK; i++) {
			while ((int64_t)npreceding_empties < ninserts &&
						 empties[ninserts - 1 - npreceding_empties]  / QF_SLOTS_PER_BLOCK < i)
				npreceding_empties++;

			if (get_block(qf, i)->offset + ninserts - npreceding_empties < BITMASK(8*sizeof(qf->blocks[0].offset)))
				get_block(qf, i)->offset += ninserts - npreceding_empties;
			else
				get_block(qf, i)->offset = (uint8_t) BITMASK(8*sizeof(qf->blocks[0].offset));
		}
	}

	for (i = 0; i < total_remainders; i++)
		set_slot(qf, overwrite_index + i, remainders[i]);

	//modify_metadata(&qf->runtimedata->pc_noccupied_slots, ninserts);

	return true;
}

__host__ __device__ static inline int remove_replace_slots_and_shift_remainders_and_runends_and_offsets(QF *qf,
																																										 int		 operation,
																																										 uint64_t		 bucket_index,
																																										 uint64_t		 overwrite_index,
																																										 const uint64_t	*remainders,
																																										 uint64_t		 total_remainders,
																																										 uint64_t		 old_length)
{
	uint64_t i;

	// Update the slots
	for (i = 0; i < total_remainders; i++)
		set_slot(qf, overwrite_index + i, remainders[i]);

	// If this is the last thing in its run, then we may need to set a new runend bit
	if (is_runend(qf, overwrite_index + old_length - 1)) {
	  if (total_remainders > 0) {
	    // If we're not deleting this entry entirely, then it will still the last entry in this run
	    METADATA_WORD(qf, runends, overwrite_index + total_remainders - 1) |= 1ULL << ((overwrite_index + total_remainders - 1) % 64);
	  } else if (overwrite_index > bucket_index &&
		     !is_runend(qf, overwrite_index - 1)) {
	    // If we're deleting this entry entirely, but it is not the first entry in this run,
	    // then set the preceding entry to be the runend
	    METADATA_WORD(qf, runends, overwrite_index - 1) |= 1ULL << ((overwrite_index - 1) % 64);
	  }
	}

	// shift slots back one run at a time
	uint64_t original_bucket = bucket_index;
	uint64_t current_bucket = bucket_index;
	uint64_t current_slot = overwrite_index + total_remainders;
	uint64_t current_distance = old_length - total_remainders;
	int ret_current_distance = current_distance;

	while (current_distance > 0) {
		if (is_runend(qf, current_slot + current_distance - 1)) {
			do {
				current_bucket++;
			} while (current_bucket < current_slot + current_distance &&
							 !is_occupied(qf, current_bucket));
		}

		if (current_bucket <= current_slot) {
			set_slot(qf, current_slot, get_slot(qf, current_slot + current_distance));
			if (is_runend(qf, current_slot) !=
					is_runend(qf, current_slot + current_distance))
				METADATA_WORD(qf, runends, current_slot) ^= 1ULL << (current_slot % 64);
			current_slot++;

		} else if (current_bucket <= current_slot + current_distance) {
			uint64_t i;
			for (i = current_slot; i < current_slot + current_distance; i++) {
				set_slot(qf, i, 0);
				METADATA_WORD(qf, runends, i) &= ~(1ULL << (i % 64));
			}

			current_distance = current_slot + current_distance - current_bucket;
			current_slot = current_bucket;
		} else {
			current_distance = 0;
		}
	}

	// reset the occupied bit of the hash bucket index if the hash is the
	// only item in the run and is removed completely.
	if (operation && !total_remainders)
		METADATA_WORD(qf, occupieds, bucket_index) &= ~(1ULL << (bucket_index % 64));

	// update the offset bits.
	// find the number of occupied slots in the original_bucket block.
	// Then find the runend slot corresponding to the last run in the
	// original_bucket block.
	// Update the offset of the block to which it belongs.
	uint64_t original_block = original_bucket / QF_SLOTS_PER_BLOCK;
	if (old_length > total_remainders) {	// we only update offsets if we shift/delete anything
		while (1) {
			uint64_t last_occupieds_hash_index = QF_SLOTS_PER_BLOCK * original_block + (QF_SLOTS_PER_BLOCK - 1);
			uint64_t runend_index = run_end(qf, last_occupieds_hash_index);
			// runend spans across the block
			// update the offset of the next block
			if (runend_index / QF_SLOTS_PER_BLOCK == original_block) { // if the run ends in the same block
				if (get_block(qf, original_block + 1)->offset == 0)
					break;
				get_block(qf, original_block + 1)->offset = 0;
			} else { // if the last run spans across the block
				if (get_block(qf, original_block + 1)->offset == (runend_index - last_occupieds_hash_index))
					break;
				get_block(qf, original_block + 1)->offset = (runend_index - last_occupieds_hash_index);
			}
			original_block++;
		}
	}

	int num_slots_freed = old_length - total_remainders;
	//modify_metadata(&qf->runtimedata->pc_noccupied_slots, -num_slots_freed);
	/*qf->metadata->noccupied_slots -= (old_length - total_remainders);*/
	if (!total_remainders) {
		//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, -1);
		/*qf->metadata->ndistinct_elts--;*/
	}

	return ret_current_distance;
}

/*****************************************************************************
 * Code that uses the above to implement a QF with keys and inline counters. *
 *****************************************************************************/

/*
	 Counter format:
	 0 xs:    <empty string>
	 1 x:     x
	 2 xs:    xx
	 3 0s:    000
	 >2 xs:   xbc...cx  for x != 0, b < x, c != 0, x
	 >3 0s:   0c...c00  for c != 0
	 */
__host__ __device__ static inline uint64_t *encode_counter(QF *qf, uint64_t remainder, uint64_t
																			 counter, uint64_t *slots)
{
	uint64_t digit = remainder;
	uint64_t base = (1ULL << qf->metadata->bits_per_slot) - 1;
	uint64_t *p = slots;

	if (counter == 0)
		return p;

	*--p = remainder;

	if (counter == 1)
		return p;

	if (counter == 2) {
		*--p = remainder;
		return p;
	}

	if (counter == 3 && remainder == 0) {
		*--p = remainder;
		*--p = remainder;
		return p;
	}

	if (counter == 3 && remainder > 0) {
		*--p = 0;
		*--p = remainder;
		return p;
	}

	if (remainder == 0)
		*--p = remainder;
	else
		base--;

	if (remainder)
		counter -= 3;
	else
		counter -= 4;
	do {
		digit = counter % base;
		digit++; /* Zero not allowed */
		if (remainder && digit >= remainder)
			digit++; /* Cannot overflow since digit is mod 2^r-2 */
		*--p = digit;
		counter /= base;
	} while (counter);

	if (remainder && digit >= remainder)
		*--p = 0;

	*--p = remainder;

	return p;
}

/* Returns the length of the encoding.
REQUIRES: index points to first slot of a counter. */
__host__ __device__ static inline uint64_t decode_counter(const QF *qf, uint64_t index, uint64_t *remainder, uint64_t *count)
{
	uint64_t base;
	uint64_t rem;
	uint64_t cnt;
	uint64_t digit;
	uint64_t end;

	*remainder = rem = get_slot(qf, index);

	if (is_runend(qf, index)) { /* Entire run is "0" */
		*count = 1;
		return index;
	}

	digit = get_slot(qf, index + 1);

	if (is_runend(qf, index + 1)) {
		*count = digit == rem ? 2 : 1;
		return index + (digit == rem ? 1 : 0);
	}

	if (rem > 0 && digit >= rem) {
		*count = digit == rem ? 2 : 1;
		return index + (digit == rem ? 1 : 0);
	}

	if (rem > 0 && digit == 0 && get_slot(qf, index + 2) == rem) {
		*count = 3;
		return index + 2;
	}

	if (rem == 0 && digit == 0) {
		if (get_slot(qf, index + 2) == 0) {
			*count = 3;
			return index + 2;
		} else {
			*count = 2;
			return index + 1;
		}
	}

	cnt = 0;
	base = (1ULL << qf->metadata->bits_per_slot) - (rem ? 2 : 1);

	end = index + 1;
	while (digit != rem && !is_runend(qf, end)) {
		if (digit > rem)
			digit--;
		if (digit && rem)
			digit--;
		cnt = cnt * base + digit;

		end++;
		digit = get_slot(qf, end);
	}

	if (rem) {
		*count = cnt + 3;
		return end;
	}

	if (is_runend(qf, end) || get_slot(qf, end + 1) != 0) {
		*count = 1;
		return index;
	}

	*count = cnt + 4;
	return end + 1;
}

/* return the next slot which corresponds to a
 * different element
 * */
__device__ static inline uint64_t next_slot(QF *qf, uint64_t current)
{
	uint64_t rem = get_slot(qf, current);
	current++;

	while (get_slot(qf, current) == rem && current <= qf->metadata->nslots) {
		current++;
	}
	return current;
}

__host__ __device__ static inline int insert1(QF *qf, __uint64_t hash, uint8_t runtime_lock)
{
	int ret_distance = 0;
	uint64_t hash_remainder           = hash & BITMASK(qf->metadata->bits_per_slot);
	uint64_t hash_bucket_index        = hash >> qf->metadata->bits_per_slot;
	uint64_t hash_bucket_block_offset = hash_bucket_index % QF_SLOTS_PER_BLOCK;
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		if (!qf_lock(qf, hash_bucket_index,  true, runtime_lock))
			return QF_COULDNT_LOCK;
	}
	*/
  //printf("In insert1, Index is %llu, block_offset is %llu, remainder is %llu \n", hash_bucket_index, hash_bucket_block_offset, hash_remainder);

	if (is_empty(qf, hash_bucket_index) /* might_be_empty(qf, hash_bucket_index) && runend_index == hash_bucket_index */) {
		METADATA_WORD(qf, runends, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);
		set_slot(qf, hash_bucket_index, hash_remainder);
		METADATA_WORD(qf, occupieds, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);

		ret_distance = 0;
		//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
		//modify_metadata(&qf->runtimedata->pc_noccupied_slots, 1);
		//modify_metadata(&qf->runtimedata->pc_nelts, 1);
	} else {
		uint64_t runend_index       = run_end(qf, hash_bucket_index);
		int operation = 0; /* Insert into empty bucket */
		uint64_t insert_index = runend_index + 1;
		uint64_t new_value = hash_remainder;

		/* printf("RUNSTART: %02lx RUNEND: %02lx\n", runstart_index, runend_index); */

		uint64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf, hash_bucket_index- 1) + 1;

		if (is_occupied(qf, hash_bucket_index)) {

			/* Find the counter for this remainder if it exists. */
			uint64_t current_remainder = get_slot(qf, runstart_index);
			uint64_t zero_terminator = runstart_index;

			/* The counter for 0 is special. */
			if (current_remainder == 0) {
				uint64_t t = runstart_index + 1;
				while (t < runend_index && get_slot(qf, t) != 0)
					t++;
				if (t < runend_index && get_slot(qf, t+1) == 0)
					zero_terminator = t+1; /* Three or more 0s */
				else if (runstart_index < runend_index && get_slot(qf, runstart_index
																													 + 1) == 0)
					zero_terminator = runstart_index + 1; /* Exactly two 0s */
				/* Otherwise, exactly one 0 (i.e. zero_terminator == runstart_index) */

				/* May read past end of run, but that's OK because loop below
					 can handle that */
				if (hash_remainder != 0) {
					runstart_index = zero_terminator + 1;
					current_remainder = get_slot(qf, runstart_index);
				}
			}

			/* Skip over counters for other remainders. */
			while (current_remainder < hash_remainder && runstart_index <=
						 runend_index) {
				/* If this remainder has an extended counter, skip over it. */
				if (runstart_index < runend_index &&
						get_slot(qf, runstart_index + 1) < current_remainder) {
					runstart_index = runstart_index + 2;
					while (runstart_index < runend_index &&
								 get_slot(qf, runstart_index) != current_remainder)
						runstart_index++;
					runstart_index++;

					/* This remainder has a simple counter. */
				} else {
					runstart_index++;
				}

				/* This may read past the end of the run, but the while loop
					 condition will prevent us from using the invalid result in
					 that case. */
				current_remainder = get_slot(qf, runstart_index);
			}

			/* If this is the first time we've inserted the new remainder,
				 and it is larger than any remainder in the run. */
			if (runstart_index > runend_index) {
				operation = 1;
				insert_index = runstart_index;
				new_value = hash_remainder;
				//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);

				/* This is the first time we're inserting this remainder, but
					 there are larger remainders already in the run. */
			} else if (current_remainder != hash_remainder) {
				operation = 2; /* Inserting */
				insert_index = runstart_index;
				new_value = hash_remainder;
				//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);

				/* Cases below here: we're incrementing the (simple or
					 extended) counter for this remainder. */

				/* If there's exactly one instance of this remainder. */
			} else if (runstart_index == runend_index ||
								 (hash_remainder > 0 && get_slot(qf, runstart_index + 1) >
									hash_remainder) ||
								 (hash_remainder == 0 && zero_terminator == runstart_index)) {
				operation = 2; /* Insert */
				insert_index = runstart_index;
				new_value = hash_remainder;

				/* If there are exactly two instances of this remainder. */
			} else if ((hash_remainder > 0 && get_slot(qf, runstart_index + 1) ==
									hash_remainder) ||
								 (hash_remainder == 0 && zero_terminator == runstart_index + 1)) {
				operation = 2; /* Insert */
				insert_index = runstart_index + 1;
				new_value = 0;

				/* Special case for three 0s */
			} else if (hash_remainder == 0 && zero_terminator == runstart_index + 2) {
				operation = 2; /* Insert */
				insert_index = runstart_index + 1;
				new_value = 1;

				/* There is an extended counter for this remainder. */
			} else {

				/* Move to the LSD of the counter. */
				insert_index = runstart_index + 1;
				while (get_slot(qf, insert_index+1) != hash_remainder)
					insert_index++;

				/* Increment the counter. */
				uint64_t digit, carry;
				do {
					carry = 0;
					digit = get_slot(qf, insert_index);
					// Convert a leading 0 (which is special) to a normal encoded digit
					if (digit == 0) {
						digit++;
						if (digit == current_remainder)
							digit++;
					}

					// Increment the digit
					digit = (digit + 1) & BITMASK(qf->metadata->bits_per_slot);

					// Ensure digit meets our encoding requirements
					if (digit == 0) {
						digit++;
						carry = 1;
					}
					if (digit == current_remainder)
						digit = (digit + 1) & BITMASK(qf->metadata->bits_per_slot);
					if (digit == 0) {
						digit++;
						carry = 1;
					}

					set_slot(qf, insert_index, digit);
					insert_index--;
				} while(insert_index > runstart_index && carry);

				/* If the counter needs to be expanded. */
				if (insert_index == runstart_index && (carry > 0 || (current_remainder
																														 != 0 && digit >=
																														 current_remainder)))
				{
					operation = 2; /* insert */
					insert_index = runstart_index + 1;
					if (!carry)						/* To prepend a 0 before the counter if the MSD is greater than the rem */
						new_value = 0;
					else if (carry) {			/* Increment the new value because we don't use 0 to encode counters */
						new_value = 2;
						/* If the rem is greater than or equal to the new_value then fail*/
						if (current_remainder > 0)
							assert(new_value < current_remainder);
					}
				} else {
					operation = -1;
				}
			}
		} //else {
			//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
		//}

		if (operation >= 0) {
			uint64_t empty_slot_index = find_first_empty_slot(qf, runend_index+1);
			if (empty_slot_index >= qf->metadata->xnslots) {
				printf("Ran out of space. Total xnslots is %lu, first empty slot is %lu\n", qf->metadata->xnslots, empty_slot_index);
				return QF_NO_SPACE;
			}
			shift_remainders(qf, insert_index, empty_slot_index);

			set_slot(qf, insert_index, new_value);
			ret_distance = insert_index - hash_bucket_index;

			shift_runends(qf, insert_index, empty_slot_index-1, 1);
			switch (operation) {
				case 0:
					METADATA_WORD(qf, runends, insert_index)   |= 1ULL << ((insert_index%QF_SLOTS_PER_BLOCK) % 64);
					break;
				case 1:
					METADATA_WORD(qf, runends, insert_index-1) &= ~(1ULL <<	(((insert_index-1) %QF_SLOTS_PER_BLOCK) %64));
					METADATA_WORD(qf, runends, insert_index)   |= 1ULL << ((insert_index%QF_SLOTS_PER_BLOCK)% 64);
					break;
				case 2:
					METADATA_WORD(qf, runends, insert_index)   &= ~(1ULL <<((insert_index %QF_SLOTS_PER_BLOCK) %64));
					break;
				default:
					printf("Invalid operation %d\n", operation);
#ifdef __CUDA_ARCH__
					__threadfence();         // ensure store issued before trap
					asm("trap;");
#else
					abort();
#endif
			}
			/*
			 * Increment the offset for each block between the hash bucket index
			 * and block of the empty slot
			 * */
			uint64_t i;
			for (i = hash_bucket_index / QF_SLOTS_PER_BLOCK + 1; i <=
					 empty_slot_index/QF_SLOTS_PER_BLOCK; i++) {
				if (get_block(qf, i)->offset < BITMASK(8*sizeof(qf->blocks[0].offset)))
					get_block(qf, i)->offset++;
				assert(get_block(qf, i)->offset != 0);
			}
			//modify_metadata(&qf->runtimedata->pc_noccupied_slots, 1);
		}
		//modify_metadata(&qf->runtimedata->pc_nelts, 1);
		METADATA_WORD(qf, occupieds, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);
	}
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		qf_unlock(qf, hash_bucket_index, true);
	}
	*/
	return ret_distance;
}


//Things that need to be check for lock acquisition:
//runend_index
//find first empty slot
//hash_bucket_index
__host__ __device__ static inline int insert1_external_lock(QF *qf, __uint64_t hash, uint8_t runtime_lock)
{
	int ret_distance = 0;
	uint64_t hash_remainder           = hash & BITMASK(qf->metadata->bits_per_slot);
	uint64_t hash_bucket_index        = hash >> qf->metadata->bits_per_slot;
	uint64_t hash_bucket_block_offset = hash_bucket_index % QF_SLOTS_PER_BLOCK;
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		if (!qf_lock(qf, hash_bucket_index,  true, runtime_lock))
			return QF_COULDNT_LOCK;
	}
	*/
  //printf("In insert1, Index is %llu, block_offset is %llu, remainder is %llu \n", hash_bucket_index, hash_bucket_block_offset, hash_remainder);

	if (is_empty(qf, hash_bucket_index) /* might_be_empty(qf, hash_bucket_index) && runend_index == hash_bucket_index */) {
		METADATA_WORD(qf, runends, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);
		set_slot(qf, hash_bucket_index, hash_remainder);
		METADATA_WORD(qf, occupieds, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);

		ret_distance = 0;
		//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
		//modify_metadata(&qf->runtimedata->pc_noccupied_slots, 1);
		//modify_metadata(&qf->runtimedata->pc_nelts, 1);
	} else {
		uint64_t runend_index       = run_end(qf, hash_bucket_index);
		int operation = 0; /* Insert into empty bucket */
		uint64_t insert_index = runend_index + 1;
		uint64_t new_value = hash_remainder;

		/* printf("RUNSTART: %02lx RUNEND: %02lx\n", runstart_index, runend_index); */

		uint64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf, hash_bucket_index- 1) + 1;

		if (is_occupied(qf, hash_bucket_index)) {

			/* Find the counter for this remainder if it exists. */
			uint64_t current_remainder = get_slot(qf, runstart_index);
			uint64_t zero_terminator = runstart_index;

			/* The counter for 0 is special. */
			if (current_remainder == 0) {
				uint64_t t = runstart_index + 1;
				while (t < runend_index && get_slot(qf, t) != 0)
					t++;
				if (t < runend_index && get_slot(qf, t+1) == 0)
					zero_terminator = t+1; /* Three or more 0s */
				else if (runstart_index < runend_index && get_slot(qf, runstart_index
																													 + 1) == 0)
					zero_terminator = runstart_index + 1; /* Exactly two 0s */
				/* Otherwise, exactly one 0 (i.e. zero_terminator == runstart_index) */

				/* May read past end of run, but that's OK because loop below
					 can handle that */
				if (hash_remainder != 0) {
					runstart_index = zero_terminator + 1;
					current_remainder = get_slot(qf, runstart_index);
				}
			}

			/* Skip over counters for other remainders. */
			while (current_remainder < hash_remainder && runstart_index <=
						 runend_index) {
				/* If this remainder has an extended counter, skip over it. */
				if (runstart_index < runend_index &&
						get_slot(qf, runstart_index + 1) < current_remainder) {
					runstart_index = runstart_index + 2;
					while (runstart_index < runend_index &&
								 get_slot(qf, runstart_index) != current_remainder)
						runstart_index++;
					runstart_index++;

					/* This remainder has a simple counter. */
				} else {
					runstart_index++;
				}

				/* This may read past the end of the run, but the while loop
					 condition will prevent us from using the invalid result in
					 that case. */
				current_remainder = get_slot(qf, runstart_index);
			}

			/* If this is the first time we've inserted the new remainder,
				 and it is larger than any remainder in the run. */
			if (runstart_index > runend_index) {
				operation = 1;
				insert_index = runstart_index;
				new_value = hash_remainder;
				//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);

				/* This is the first time we're inserting this remainder, but
					 there are larger remainders already in the run. */
			} else if (current_remainder != hash_remainder) {
				operation = 2; /* Inserting */
				insert_index = runstart_index;
				new_value = hash_remainder;
				//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);

				/* Cases below here: we're incrementing the (simple or
					 extended) counter for this remainder. */

				/* If there's exactly one instance of this remainder. */
			} else if (runstart_index == runend_index ||
								 (hash_remainder > 0 && get_slot(qf, runstart_index + 1) >
									hash_remainder) ||
								 (hash_remainder == 0 && zero_terminator == runstart_index)) {
				operation = 2; /* Insert */
				insert_index = runstart_index;
				new_value = hash_remainder;

				/* If there are exactly two instances of this remainder. */
			} else if ((hash_remainder > 0 && get_slot(qf, runstart_index + 1) ==
									hash_remainder) ||
								 (hash_remainder == 0 && zero_terminator == runstart_index + 1)) {
				operation = 2; /* Insert */
				insert_index = runstart_index + 1;
				new_value = 0;

				/* Special case for three 0s */
			} else if (hash_remainder == 0 && zero_terminator == runstart_index + 2) {
				operation = 2; /* Insert */
				insert_index = runstart_index + 1;
				new_value = 1;

				/* There is an extended counter for this remainder. */
			} else {

				/* Move to the LSD of the counter. */
				insert_index = runstart_index + 1;
				while (get_slot(qf, insert_index+1) != hash_remainder)
					insert_index++;

				/* Increment the counter. */
				uint64_t digit, carry;
				do {
					carry = 0;
					digit = get_slot(qf, insert_index);
					// Convert a leading 0 (which is special) to a normal encoded digit
					if (digit == 0) {
						digit++;
						if (digit == current_remainder)
							digit++;
					}

					// Increment the digit
					digit = (digit + 1) & BITMASK(qf->metadata->bits_per_slot);

					// Ensure digit meets our encoding requirements
					if (digit == 0) {
						digit++;
						carry = 1;
					}
					if (digit == current_remainder)
						digit = (digit + 1) & BITMASK(qf->metadata->bits_per_slot);
					if (digit == 0) {
						digit++;
						carry = 1;
					}

					set_slot(qf, insert_index, digit);
					insert_index--;
				} while(insert_index > runstart_index && carry);

				/* If the counter needs to be expanded. */
				if (insert_index == runstart_index && (carry > 0 || (current_remainder
																														 != 0 && digit >=
																														 current_remainder)))
				{
					operation = 2; /* insert */
					insert_index = runstart_index + 1;
					if (!carry)						/* To prepend a 0 before the counter if the MSD is greater than the rem */
						new_value = 0;
					else if (carry) {			/* Increment the new value because we don't use 0 to encode counters */
						new_value = 2;
						/* If the rem is greater than or equal to the new_value then fail*/
						if (current_remainder > 0)
							assert(new_value < current_remainder);
					}
				} else {
					operation = -1;
				}
			}
		} //else {
			//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
		//}

		if (operation >= 0) {
			uint64_t empty_slot_index = find_first_empty_slot(qf, runend_index+1);
			if (empty_slot_index >= qf->metadata->xnslots) {
				printf("Ran out of space. Total xnslots is %lu, first empty slot is %lu\n", qf->metadata->xnslots, empty_slot_index);
				return QF_NO_SPACE;
			}
			shift_remainders(qf, insert_index, empty_slot_index);

			set_slot(qf, insert_index, new_value);
			ret_distance = insert_index - hash_bucket_index;

			shift_runends(qf, insert_index, empty_slot_index-1, 1);
			switch (operation) {
				case 0:
					METADATA_WORD(qf, runends, insert_index)   |= 1ULL << ((insert_index%QF_SLOTS_PER_BLOCK) % 64);
					break;
				case 1:
					METADATA_WORD(qf, runends, insert_index-1) &= ~(1ULL <<	(((insert_index-1) %QF_SLOTS_PER_BLOCK) %64));
					METADATA_WORD(qf, runends, insert_index)   |= 1ULL << ((insert_index%QF_SLOTS_PER_BLOCK)% 64);
					break;
				case 2:
					METADATA_WORD(qf, runends, insert_index)   &= ~(1ULL <<((insert_index %QF_SLOTS_PER_BLOCK) %64));
					break;
				default:
					printf("Invalid operation %d\n", operation);
#ifdef __CUDA_ARCH__
					__threadfence();         // ensure store issued before trap
					asm("trap;");
#else
					abort();
#endif
			}
			/*
			 * Increment the offset for each block between the hash bucket index
			 * and block of the empty slot
			 * */
			uint64_t i;
			for (i = hash_bucket_index / QF_SLOTS_PER_BLOCK + 1; i <=
					 empty_slot_index/QF_SLOTS_PER_BLOCK; i++) {
				if (get_block(qf, i)->offset < BITMASK(8*sizeof(qf->blocks[0].offset)))
					get_block(qf, i)->offset++;
				assert(get_block(qf, i)->offset != 0);
			}
			//modify_metadata(&qf->runtimedata->pc_noccupied_slots, 1);
		}
		//modify_metadata(&qf->runtimedata->pc_nelts, 1);
		METADATA_WORD(qf, occupieds, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);
	}
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		qf_unlock(qf, hash_bucket_index, true);
	}
	*/
	return ret_distance;
}

__host__ __device__ static inline int insert(QF *qf, __uint64_t hash, uint64_t count, uint8_t
												 runtime_lock)
{
	int ret_distance = 0;
	uint64_t hash_remainder           = hash & BITMASK(qf->metadata->bits_per_slot);
	uint64_t hash_bucket_index        = hash >> qf->metadata->bits_per_slot;
	uint64_t hash_bucket_block_offset = hash_bucket_index % QF_SLOTS_PER_BLOCK;
	/*uint64_t hash_bucket_lock_offset  = hash_bucket_index % NUM_SLOTS_TO_LOCK;*/
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		if (!qf_lock(qf, hash_bucket_index,  false, runtime_lock))
			return QF_COULDNT_LOCK;
	}
	*/
	uint64_t runend_index = run_end(qf, hash_bucket_index);

	/* Empty slot */
	if (might_be_empty(qf, hash_bucket_index) && runend_index ==
			hash_bucket_index) {
		METADATA_WORD(qf, runends, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);
		set_slot(qf, hash_bucket_index, hash_remainder);
		METADATA_WORD(qf, occupieds, hash_bucket_index) |= 1ULL <<
			(hash_bucket_block_offset % 64);

		//ERIC TODO: see if this metadata is needed--probably isn't compatible with GPU
		//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
		//modify_metadata(&qf->runtimedata->pc_noccupied_slots, 1);
		//modify_metadata(&qf->runtimedata->pc_nelts, 1);
		/* This trick will, I hope, keep the fast case fast. */
		if (count > 1) {
			insert(qf, hash, count - 1, QF_NO_LOCK);
		}
	} else { /* Non-empty slot */
		uint64_t new_values[67];
		int64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf,hash_bucket_index- 1) + 1;

		bool ret;
		if (!is_occupied(qf, hash_bucket_index)) { /* Empty bucket, but its slot is occupied. */
			uint64_t *p = encode_counter(qf, hash_remainder, count, &new_values[67]);
			ret = insert_replace_slots_and_shift_remainders_and_runends_and_offsets(qf, 0, hash_bucket_index, runstart_index, p, &new_values[67] - p, 0);
			if (!ret)
				return QF_NO_SPACE;
			//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
			ret_distance = runstart_index - hash_bucket_index;
		} else { /* Non-empty bucket */

			uint64_t current_remainder, current_count, current_end;

			/* Find the counter for this remainder, if one exists. */
			current_end = decode_counter(qf, runstart_index, &current_remainder,&current_count);
			while (current_remainder < hash_remainder && !is_runend(qf, current_end)) {
				runstart_index = current_end + 1;
				current_end = decode_counter(qf, runstart_index, &current_remainder,
																		 &current_count);
			}

			/* If we reached the end of the run w/o finding a counter for this remainder,
				 then append a counter for this remainder to the run. */
			if (current_remainder < hash_remainder) {
				uint64_t *p = encode_counter(qf, hash_remainder, count, &new_values[67]);
				ret = insert_replace_slots_and_shift_remainders_and_runends_and_offsets(qf, 1, /* Append to bucket */hash_bucket_index, current_end + 1, p, &new_values[67] - p, 0);
				if (!ret)
					return QF_NO_SPACE;
				//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
				ret_distance = (current_end + 1) - hash_bucket_index;
				/* Found a counter for this remainder.  Add in the new count. */
			} else if (current_remainder == hash_remainder) {
				uint64_t *p = encode_counter(qf, hash_remainder, current_count + count, &new_values[67]);
				ret = insert_replace_slots_and_shift_remainders_and_runends_and_offsets(qf,
																																					is_runend(qf, current_end) ? 1 : 2,
																																					hash_bucket_index,
																																					runstart_index,
																																					p,
																																					&new_values[67] - p,
																																					current_end - runstart_index + 1);
			if (!ret)
				return QF_NO_SPACE;
			ret_distance = runstart_index - hash_bucket_index;
				/* No counter for this remainder, but there are larger
					 remainders, so we're not appending to the bucket. */
			} else {
				uint64_t *p = encode_counter(qf, hash_remainder, count, &new_values[67]);
				ret = insert_replace_slots_and_shift_remainders_and_runends_and_offsets(qf,
																																								2, /* Insert to bucket */
																																								hash_bucket_index,
																																								runstart_index,
																																								p,
																																								&new_values[67] - p,
																																								0);
				if (!ret)
					return QF_NO_SPACE;
				//modify_metadata(&qf->runtimedata->pc_ndistinct_elts, 1);
			ret_distance = runstart_index - hash_bucket_index;
			}
		}
		METADATA_WORD(qf, occupieds, hash_bucket_index) |= 1ULL << (hash_bucket_block_offset % 64);

		//modify_metadata(&qf->runtimedata->pc_nelts, count);
	}
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		qf_unlock(qf, hash_bucket_index,  false);
	}
	*/
	return ret_distance;
}

__host__ __device__ inline static int _remove(QF *qf, __uint64_t hash, uint64_t count, uint8_t
													runtime_lock)
{
	int ret_numfreedslots = 0;
	uint64_t hash_remainder           = hash & BITMASK(qf->metadata->bits_per_slot);
	uint64_t hash_bucket_index        = hash >> qf->metadata->bits_per_slot;
	uint64_t current_remainder, current_count, current_end;
	uint64_t new_values[67];
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		if (!qf_lock(qf, hash_bucket_index,  false, runtime_lock))
			return -2;
	}
	*/

	/* Empty bucket */
	if (!is_occupied(qf, hash_bucket_index))
		return -1;

	uint64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf, hash_bucket_index - 1) + 1;
	uint64_t original_runstart_index = runstart_index;
	int only_item_in_the_run = 0;

	/*Find the counter for this remainder, if one exists.*/
	current_end = decode_counter(qf, runstart_index, &current_remainder, &current_count);
	while (current_remainder < hash_remainder && !is_runend(qf, current_end)) {
		runstart_index = current_end + 1;
		current_end = decode_counter(qf, runstart_index, &current_remainder, &current_count);
	}
	/* remainder not found in the given run */
	if (current_remainder != hash_remainder)
		return -1;

	if (original_runstart_index == runstart_index && is_runend(qf, current_end))
		only_item_in_the_run = 1;

	/* endode the new counter */
	uint64_t *p = encode_counter(qf, hash_remainder,
															 count > current_count ? 0 : current_count - count,
															 &new_values[67]);
	ret_numfreedslots = remove_replace_slots_and_shift_remainders_and_runends_and_offsets(qf,
																																		only_item_in_the_run,
																																		hash_bucket_index,
																																		runstart_index,
																																		p,
																																		&new_values[67] - p,
																																		current_end - runstart_index + 1);

	// update the nelements.
	//modify_metadata(&qf->runtimedata->pc_nelts, -count);
	/*qf->metadata->nelts -= count;*/
	/*
	if (GET_NO_LOCK(runtime_lock) != QF_NO_LOCK) {
		qf_unlock(qf, hash_bucket_index, false);
	}
	*/
	return ret_numfreedslots;
}

/***********************************************************************
 * Code that uses the above to implement key-value-counter operations. *
 ***********************************************************************/

__host__ uint64_t qf_init(QF *qf, uint64_t nslots, uint64_t key_bits, uint64_t value_bits,
								 enum qf_hashmode hash, uint32_t seed, void* buffer, uint64_t
								 buffer_len)
{
	uint64_t num_slots, xnslots, nblocks;
	uint64_t key_remainder_bits, bits_per_slot;
	uint64_t size;
	uint64_t total_num_bytes;

	assert(popcnt(nslots) == 1); /* nslots must be a power of 2 */
	num_slots = nslots;
	xnslots = nslots + 10*sqrt((double)nslots);
	nblocks = (xnslots + QF_SLOTS_PER_BLOCK - 1) / QF_SLOTS_PER_BLOCK;
	key_remainder_bits = key_bits;
	while (nslots > 1 && key_remainder_bits > 0) {
		key_remainder_bits--;
		nslots >>= 1;
	}
	assert(key_remainder_bits >= 2);

	bits_per_slot = key_remainder_bits + value_bits;
	assert (QF_BITS_PER_SLOT == 0 || QF_BITS_PER_SLOT == bits_per_slot);
	assert(bits_per_slot > 1);
#if QF_BITS_PER_SLOT == 8 || QF_BITS_PER_SLOT == 16 || QF_BITS_PER_SLOT == 32 || QF_BITS_PER_SLOT == 64
	size = nblocks * sizeof(qfblock);
#else
	size = nblocks * (sizeof(qfblock) + QF_SLOTS_PER_BLOCK * bits_per_slot / 8);
#endif

	total_num_bytes = sizeof(qfmetadata) + size;
	if (buffer == NULL || total_num_bytes > buffer_len)
		return total_num_bytes;

	// memset(buffer, 0, total_num_bytes);
	qf->metadata = (qfmetadata *)(buffer);
	qf->blocks = (qfblock *)(qf->metadata + 1);

	qf->metadata->magic_endian_number = MAGIC_NUMBER;
	qf->metadata->reserved = 0;
	qf->metadata->hash_mode = hash;
	qf->metadata->total_size_in_bytes = size;
	qf->metadata->seed = seed;
	qf->metadata->nslots = num_slots;
	qf->metadata->xnslots = xnslots;
	qf->metadata->key_bits = key_bits;
	qf->metadata->value_bits = value_bits;
	qf->metadata->key_remainder_bits = key_remainder_bits;
	qf->metadata->bits_per_slot = bits_per_slot;

	qf->metadata->range = qf->metadata->nslots;
	qf->metadata->range <<= qf->metadata->key_remainder_bits;
	qf->metadata->nblocks = (qf->metadata->xnslots + QF_SLOTS_PER_BLOCK - 1) /
		QF_SLOTS_PER_BLOCK;
	qf->metadata->nelts = 0;
	qf->metadata->ndistinct_elts = 0;
	qf->metadata->noccupied_slots = 0;

	qf->runtimedata->num_locks = (qf->metadata->xnslots/NUM_SLOTS_TO_LOCK)+2;

	pc_init(&qf->runtimedata->pc_nelts, (int64_t*)&qf->metadata->nelts, 8, 100);
	pc_init(&qf->runtimedata->pc_ndistinct_elts, (int64_t*)&qf->metadata->ndistinct_elts, 8, 100);
	pc_init(&qf->runtimedata->pc_noccupied_slots, (int64_t*)&qf->metadata->noccupied_slots, 8, 100);
	/* initialize container resize */
	qf->runtimedata->auto_resize = 0;
	qf->runtimedata->container_resize = qf_resize_malloc;
	/* initialize all the locks to 0 */
	qf->runtimedata->metadata_lock = 0;
	//etodo: copy this to GPU
	qf->runtimedata->locks = (volatile int *)calloc(qf->runtimedata->num_locks, sizeof(volatile int));
	if (qf->runtimedata->locks == NULL) {
		perror("Couldn't allocate memory for runtime locks.");
		exit(EXIT_FAILURE);
	}
#ifdef LOG_WAIT_TIME
	qf->runtimedata->wait_times = (wait_time_data*
																 )calloc(qf->runtimedata->num_locks+1,
																				 sizeof(wait_time_data));
	if (qf->runtimedata->wait_times == NULL) {
		perror("Couldn't allocate memory for runtime wait_times.");
		exit(EXIT_FAILURE);
	}
#endif

	return total_num_bytes;
}

__host__ uint64_t qf_use(QF* qf, void* buffer, uint64_t buffer_len)
{
	qf->metadata = (qfmetadata *)(buffer);
	if (qf->metadata->total_size_in_bytes + sizeof(qfmetadata) > buffer_len) {
		return qf->metadata->total_size_in_bytes + sizeof(qfmetadata);
	}
	qf->blocks = (qfblock *)(qf->metadata + 1);

	qf->runtimedata = (qfruntime *)calloc(sizeof(qfruntime), 1);
	if (qf->runtimedata == NULL) {
		perror("Couldn't allocate memory for runtime data.");
		exit(EXIT_FAILURE);
	}
	/* initialize all the locks to 0 */
	qf->runtimedata->metadata_lock = 0;
	qf->runtimedata->locks = (volatile int *)calloc(qf->runtimedata->num_locks,
																					sizeof(volatile int));
	if (qf->runtimedata->locks == NULL) {
		perror("Couldn't allocate memory for runtime locks.");
		exit(EXIT_FAILURE);
	}
#ifdef LOG_WAIT_TIME
	qf->runtimedata->wait_times = (wait_time_data*
																 )calloc(qf->runtimedata->num_locks+1,
																				 sizeof(wait_time_data));
	if (qf->runtimedata->wait_times == NULL) {
		perror("Couldn't allocate memory for runtime wait_times.");
		exit(EXIT_FAILURE);
	}
#endif

	return sizeof(qfmetadata) + qf->metadata->total_size_in_bytes;
}

__host__ void *qf_destroy(QF *qf)
{
	assert(qf->runtimedata != NULL);
	if (qf->runtimedata->locks != NULL)
		free((void*)qf->runtimedata->locks);
	if (qf->runtimedata->wait_times != NULL)
		free(qf->runtimedata->wait_times);
	if (qf->runtimedata->f_info.filepath != NULL)
		free(qf->runtimedata->f_info.filepath);
	free(qf->runtimedata);

	return (void*)qf->metadata;
}

__host__ bool qf_malloc(QF *qf, uint64_t nslots, uint64_t key_bits, uint64_t
							 value_bits, enum qf_hashmode hash, bool on_device, uint32_t seed)
{
	uint64_t total_num_bytes = qf_init(qf, nslots, key_bits, value_bits,
																	 hash, seed, NULL, 0);

  //buffer malloc bad?
	void* buffer = malloc(total_num_bytes);
  memset(buffer, 0, total_num_bytes);


	if (buffer == NULL) {
		perror("Couldn't allocate memory for the CQF.");
		exit(EXIT_FAILURE);
	}

	qf->runtimedata = (qfruntime*)calloc(sizeof(qfruntime), 1);


	if (qf->runtimedata == NULL) {
		perror("Couldn't allocate memory for runtime data.");
		exit(EXIT_FAILURE);
	}

	uint64_t init_size = qf_init(qf, nslots, key_bits, value_bits, hash, seed,
															 buffer, total_num_bytes);

	if (init_size == total_num_bytes)
		return total_num_bytes;
	else
		return -1;
}



__host__ bool qf_free(QF *qf)
{
	assert(qf->metadata != NULL);
	void *buffer = qf_destroy(qf);
	if (buffer != NULL) {
		free(buffer);
		return true;
	}

	return false;
}

__host__ void qf_free_gpu(QF * qf){

	QF hostQF;

	//cudaMallocHost((void **)&hostQF, sizeof(QF));

	cudaMemcpy(&hostQF, qf, sizeof(QF), cudaMemcpyDeviceToHost);

	cudaFree(hostQF.runtimedata);
	cudaFree(hostQF.metadata);
	cudaFree(hostQF.blocks);

	cudaFree(qf);
	
}

__host__ void qf_copy(QF *dest, const QF *src)
{
	DEBUG_CQF("%s\n","Source CQF");
	DEBUG_DUMP(src);
	memcpy(dest->runtimedata, src->runtimedata, sizeof(qfruntime));
	memcpy(dest->metadata, src->metadata, sizeof(qfmetadata));
	memcpy(dest->blocks, src->blocks, src->metadata->total_size_in_bytes);
	DEBUG_CQF("%s\n","Destination CQF after copy.");
	DEBUG_DUMP(dest);
}

__host__ void qf_reset(QF *qf)
{
	qf->metadata->nelts = 0;
	qf->metadata->ndistinct_elts = 0;
	qf->metadata->noccupied_slots = 0;

#ifdef LOG_WAIT_TIME
	memset(qf->wait_times, 0,
				 (qf->runtimedata->num_locks+1)*sizeof(wait_time_data));
#endif
#if QF_BITS_PER_SLOT == 8 || QF_BITS_PER_SLOT == 16 || QF_BITS_PER_SLOT == 32 || QF_BITS_PER_SLOT == 64
	memset(qf->blocks, 0, qf->metadata->nblocks* sizeof(qfblock));
#else
	memset(qf->blocks, 0, qf->metadata->nblocks*(sizeof(qfblock) + QF_SLOTS_PER_BLOCK *
																		 qf->metadata->bits_per_slot / 8));
#endif
}

__host__ int64_t qf_resize_malloc(QF *qf, uint64_t nslots)
{
	QF new_qf;
	if (!qf_malloc(&new_qf, nslots, qf->metadata->key_bits,
								 qf->metadata->value_bits, qf->metadata->hash_mode,
								 false, qf->metadata->seed))
		return -1;
	if (qf->runtimedata->auto_resize) qf_set_auto_resize(&new_qf, true);

	// copy keys from qf into new_qf
	QFi qfi;
	qf_iterator_from_position(qf, &qfi, 0);
	int64_t ret_numkeys = 0;
	do {
		uint64_t key, value, count;
		qfi_get_hash(&qfi, &key, &value, &count);
		qfi_next(&qfi);
		int ret = qf_insert(&new_qf, key, value, count, QF_NO_LOCK | QF_KEY_IS_HASH);
		if (ret < 0) {
			printf("Failed to insert key: %ld into the new CQF.\n", key);
			return ret;
		}
		ret_numkeys++;
	} while(!qfi_end(&qfi));

	qf_free(qf);
	memcpy(qf, &new_qf, sizeof(QF));

	return ret_numkeys;
}

uint64_t qf_resize(QF* qf, uint64_t nslots, void* buffer, uint64_t buffer_len)
{

  printf("QF attempting resize - This will fail\n");
	QF new_qf;
	new_qf.runtimedata = (qfruntime *)calloc(sizeof(qfruntime), 1);
	if (new_qf.runtimedata == NULL) {
		perror("Couldn't allocate memory for runtime data.\n");
		exit(EXIT_FAILURE);
	}

	uint64_t init_size = qf_init(&new_qf, nslots, qf->metadata->key_bits,
															 qf->metadata->value_bits,
															 qf->metadata->hash_mode, qf->metadata->seed,
															 buffer, buffer_len);

	if (init_size > buffer_len)
		return init_size;

	if (qf->runtimedata->auto_resize)
		qf_set_auto_resize(&new_qf, true);

	// copy keys from qf into new_qf
	QFi qfi;
	qf_iterator_from_position(qf, &qfi, 0);
	do {
		uint64_t key, value, count;
		qfi_get_hash(&qfi, &key, &value, &count);
		qfi_next(&qfi);
		int ret = qf_insert(&new_qf, key, value, count, QF_NO_LOCK | QF_KEY_IS_HASH);
		if (ret < 0) {
			printf("Failed to insert key: %ld into the new CQF.\n", key);
			abort();          // kill kernel with error
		}
	} while(!qfi_end(&qfi));

	qf_free(qf);
	memcpy(qf, &new_qf, sizeof(QF));

	return init_size;
}

__host__  void qf_set_auto_resize(QF* qf, bool enabled)
{
	if (enabled)
		qf->runtimedata->auto_resize = 1;
	else
		qf->runtimedata->auto_resize = 0;
}



__host__ __device__ int qf_insert(QF *qf, uint64_t key, uint64_t value, uint64_t count, uint8_t
							flags)
{
	// We fill up the CQF up to 95% load factor.
	// This is a very conservative check.

  //TODO: GPU resizing
	/*
	if (qf_get_num_occupied_slots(qf) >= qf->metadata->nslots * 0.95) {
		if (qf->runtimedata->auto_resize) {
			fprintf(stdout, "Resizing the CQF.\n");
			if (qf->runtimedata->container_resize(qf, qf->metadata->nslots * 2) < 0)
			{
				fprintf(stderr, "Resizing the failed.\n");
				return QF_NO_SPACE;
			}
		} else
			return QF_NO_SPACE;
	}
	*/
	if (count == 0)
		return 0;

	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}

	uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));
  //printf("Inside insert, new hash is recorded as %llu\n", hash);
	int ret;
	if (count == 1)
		ret = insert1(qf, hash, flags);
	//for now count is always 1
	//else
		//ret = insert(qf, hash, count, flags);

	// check for fullness based on the distance from the home slot to the slot
	// in which the key is inserted
	/*
	if (ret == QF_NO_SPACE || ret > DISTANCE_FROM_HOME_SLOT_CUTOFF) {
		float load_factor = qf_get_num_occupied_slots(qf) /
			(float)qf->metadata->nslots;
		fprintf(stdout, "Load factor: %lf\n", load_factor);
		if (qf->runtimedata->auto_resize) {
			fprintf(stdout, "Resizing the CQF.\n");
			if (qf->runtimedata->container_resize(qf, qf->metadata->nslots * 2) > 0)
			{
				if (ret == QF_NO_SPACE) {
					if (count == 1)
						ret = insert1(qf, hash, flags);
					else
						ret = insert(qf, hash, count, flags);
				}
				fprintf(stderr, "Resize finished.\n");
			} else {
				fprintf(stderr, "Resize failed\n");
				ret = QF_NO_SPACE;
			}
		} else {
			fprintf(stderr, "The CQF is filling up.\n");
			ret = QF_NO_SPACE;
		}
	}
	*/
	return ret;
}


//We can reduce the size of the locking zones if we support functionality for grabbing extra locks
__host__ __device__ int qf_insert_external_locks(QF *qf, uint64_t key, uint64_t value, uint64_t count, uint8_t
							flags, unsigned short int * locks, int * my_indices, int locks_gathered)
{
	// We fill up the CQF up to 95% load factor.
	// This is a very conservative check.

  //TODO: GPU resizing
	/*
	if (qf_get_num_occupied_slots(qf) >= qf->metadata->nslots * 0.95) {
		if (qf->runtimedata->auto_resize) {
			fprintf(stdout, "Resizing the CQF.\n");
			if (qf->runtimedata->container_resize(qf, qf->metadata->nslots * 2) < 0)
			{
				fprintf(stderr, "Resizing the failed.\n");
				return QF_NO_SPACE;
			}
		} else
			return QF_NO_SPACE;
	}
	*/
	if (count == 0)
		return 0;

	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}

	uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));
  //printf("Inside insert, new hash is recorded as %llu\n", hash);
	int ret;
	if (count == 1)
		ret = insert1(qf, hash, flags);
	//for now count is always 1
	//else
		//ret = insert(qf, hash, count, flags);

	// check for fullness based on the distance from the home slot to the slot
	// in which the key is inserted
	/*
	if (ret == QF_NO_SPACE || ret > DISTANCE_FROM_HOME_SLOT_CUTOFF) {
		float load_factor = qf_get_num_occupied_slots(qf) /
			(float)qf->metadata->nslots;
		fprintf(stdout, "Load factor: %lf\n", load_factor);
		if (qf->runtimedata->auto_resize) {
			fprintf(stdout, "Resizing the CQF.\n");
			if (qf->runtimedata->container_resize(qf, qf->metadata->nslots * 2) > 0)
			{
				if (ret == QF_NO_SPACE) {
					if (count == 1)
						ret = insert1(qf, hash, flags);
					else
						ret = insert(qf, hash, count, flags);
				}
				fprintf(stderr, "Resize finished.\n");
			} else {
				fprintf(stderr, "Resize failed\n");
				ret = QF_NO_SPACE;
			}
		} else {
			fprintf(stderr, "The CQF is filling up.\n");
			ret = QF_NO_SPACE;
		}
	}
	*/
	return ret;
}



/*------------------------
GPU Modifications
--------------------------*/


//TODO: it might expect a short int instead of uint16_t
//TODO: needs to be 32 bits (whoops)
__device__ uint16_t get_lock(volatile uint32_t* lock, int index) {
	//set lock to 1 to claim
	//returns 0 if success
	uint32_t zero = 0;
	uint32_t one = 1;
	return atomicCAS((uint32_t *) &lock[index], zero, one);
}

//synchronous lock so that we can acquire multiple locks

// __device__ uint16_t get_lock_wait(uint32_t * locks, int index){

//   uint16_t result = 1;

//   do {

//     result = get_lock(locks, index);

//   } while (result !=0);

//   return result;

// }

__device__ uint16_t unlock(volatile uint32_t* lock, int index) {
	//set lock to 0 to release
	uint32_t zero = 0;
	uint32_t one = 1;
	//TODO: might need a __threadfence();
	lock[index] = 0;
}


__device__ __forceinline__ void exchange(uint64_t * arr, uint64_t i, uint64_t j){


	uint64_t temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;

	//maybe synchthreads?

}

__device__ __forceinline__ void compare(uint64_t * arr, uint64_t i, uint64_t j, bool dir){

	if (dir == (arr[i] > arr[j])){

		exchange(arr, i, j);

	}

}


//return the biggest int of a uint64
__device__ __forceinline__ int biggest_bit(uint64_t n){

	return 63 - __clzll((unsigned long long int) n);

}


__device__ __forceinline__ uint64_t biggest_pow_2(uint64_t n){


	return 1UL<<biggest_bit(n)-1;

}



//provide merge of a section of the array
__device__ void __bitonic_merge(uint64_t * array, uint64_t low, uint64_t n, uint64_t idx, bool dir){


	if (n > 1){

		uint64_t m = biggest_pow_2(n);

		if (idx >= low && idx < low+n-m){

			compare(array, idx, idx+m, dir);

		}
	//what spot right spot?
	__syncthreads();

	if (idx < m){
		__bitonic_merge(array, low, m, idx, dir);
	} else {

		//need to verify this section
		__bitonic_merge(array, low+m, n-m, idx-m, dir);
	}
	
	}


}

//__device__ void __bitonic_sort(uint64_t * array, uint64_t low, uint64_t n, uint64_t idx, bool dir);


__device__ void __bitonic_sort(uint64_t * array, uint64_t low, uint64_t n, uint64_t idx, bool dir){

	if (n > 1){

		//get power of two to separate on
		uint64_t m = biggest_pow_2(n);


		if (idx < m){
			__bitonic_sort(array, low, m, idx, !dir);
		} else {
			__bitonic_sort(array, low+m, n-m, idx-m, dir);
		}
		//and perform final merge
		__bitonic_merge(array, low, n, idx, dir);
		
	}


}

//in place bitonic sort
__global__ void sort(uint64_t * array, uint64_t n){

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	__bitonic_sort(array, 0, n, idx, true);

}


__device__ void swap(uint64_t * array, uint64_t first, uint64_t second){
    
    uint64_t temp = array[first];
    array[first] = array[second];
    array[second] = temp;
}

//give index to partition over
__device__ uint64_t partition(uint64_t * array, uint64_t low, uint64_t high){
    
    
    uint64_t index = low;
    uint64_t pivot = high;
    
    for (uint64_t i = low; i < high; i++){
        
        if (array[i] < array[pivot]){
            swap(array, i, index);
            index++;
        }
    }
    
    swap(array, index, pivot);
    
    return index;
    
    
}

__device__ void selection_sort( uint64_t *data, uint64_t left, uint64_t right )
{
  for( int i = left ; i <= right ; ++i )
  {
    uint64_t min_val = data[i];
    int min_idx = i;

    // Find the smallest value in the range [left, right].
    for( int j = i+1 ; j <= right ; ++j )
    {
      uint64_t val_j = data[j];
      if( val_j < min_val )
      {
        min_idx = j;
        min_val = val_j;
      }
    }

    // Swap the values.
    if( i != min_idx )
    {
      data[min_idx] = data[i];
      data[i] = min_val;
    }
  }
}


//cpp code i wrote cause cori is down - this works in C
__device__ bool assert_sorted(uint64_t * array, uint64_t nitems){
    
    uint64_t start = array[0];
    
    for (uint64_t i =1; i < nitems; i++){
        
        if (start > array[i]) return false;
        start = array[i];
    }
    
    return true;
    
}

void print_arr(uint64_t * array, uint64_t nitems){
    
    for (uint64_t i =0; i < nitems; i++){
        
        printf("%llu ",array[i]);
    }
    printf("\n");
}

__device__ void quick_sort(uint64_t * array, uint64_t low, uint64_t high, uint64_t depth){
    
    if (low >= high) return;


    if (depth >= MAX_DEPTH || high-low <= SELECT_BOUND){

    	selection_sort(array, low, high);
    	return;

    }
    
    uint64_t pivot = partition(array, low, high);
 
    
    //print_arr(array+low, high+1);
    
    //don't go below bounds
    
    
    if (pivot != low) quick_sort(array, low, pivot-1, depth+1);
    if (pivot != high) quick_sort(array, pivot+1, high, depth+1);
    
}

__device__ void assert_sorted(uint64_t * array, uint64_t low, uint64_t high){

	uint64_t smallest_key = array[low];

	for (uint64_t i = low; i < high; i++){

		assert(smallest_key <= array[i]);
		smallest_key = array[i];

	}

}


//end of cpp copy paste

__global__ void bulk_quick_sort(uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts){


	int idx = threadIdx.x + blockDim.x * blockIdx.x;



	if (idx >= num_buffers) return;


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	if (buffer_counts[idx] > 0) {

		quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

		//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	}
}



__global__ void hash_all(QF* qf, uint64_t* vals, uint64_t* hashes, uint64_t nvals, uint64_t value, uint8_t flags) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= nvals){
		return;
	}

  uint64_t key = vals[idx];

  if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}

	uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));
  
  hashes[idx] = hash;

	return;
}



//Get counts of which bucket a given item would like to move to
//these are used to set the starting pointers of the next phase
__global__ void count_off(QF * qf, uint64_t num_keys, uint64_t slots_per_lock, uint64_t * keys, uint64_t num_buffers, volatile uint64_t * buffer_counts, uint64_t value, uint8_t flags){

		int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= num_keys) return;

		//printf("idx %llu\n", idx);

		uint64_t key = keys[idx];

    //adding back in hashing here - this is inefficient
    if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
  		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
  			key = MurmurHash64A(((void *)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
  		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
  			key = hash_64(key, BITMASK(qf->metadata->key_bits));
  	}

		uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));

    //printf("%d insert %d, key: %llu, hash: %llu \n", idx, i, key, hash);
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);
		uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		uint64_t lock_index = hash_bucket_index / slots_per_lock;

		//counts are volatile, so atomic update
		//this isn't as fast as a reduction, but should be fine for the moment.
		atomicAdd((unsigned long long int *) buffer_counts+lock_index, 1);
		//buffer_counts[lock_index] += 1;



}

//A variant of count off that takes in prehashed keys
__global__ void count_off_hashed(QF * qf, uint64_t num_keys, uint64_t slots_per_lock, uint64_t * keys, uint64_t num_buffers, volatile uint64_t * buffer_counts, uint64_t value, uint8_t flags){

		int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= num_keys) return;

		//printf("idx %llu\n", idx);

		uint64_t hash = keys[idx];

    //printf("%d insert %d, key: %llu, hash: %llu \n", idx, i, key, hash);
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);
		uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		uint64_t lock_index = hash_bucket_index / slots_per_lock;

		//counts are volatile, so atomic update
		//this isn't as fast as a reduction, but should be fine for the moment.
		atomicAdd((unsigned long long int *) buffer_counts+lock_index, 1);
		//buffer_counts[lock_index] += 1;


}

// sanity check to assert that the buffers are identical
__global__ void compare_buffer_lens(uint64_t num_locks, uint64_t * buf1, uint64_t * buf2){

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= num_locks) return;

	assert(buf1[idx] == buf2[idx]);
}

// sanity check to assert that the buffers are identical
__global__ void compare_buffers(uint64_t num_locks, uint64_t ** buf1, uint64_t ** buf2){

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= num_locks) return;

	assert(buf1[idx] == buf2[idx]);
}

//this can maybe be rolled into set_buffers_binary
//it performs an identical set of operations that are O(1) here
// O(log n) there, but maybe amortized
__global__ void set_buffer_lens(QF * qf, uint64_t num_keys, uint64_t * keys, uint64_t num_buffers, uint64_t * buffer_sizes, uint64_t ** buffers){


	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	if (idx >= num_buffers) return;


	//only 1 thread will diverge - should be fine - any cost already exists because of tail
	if (idx != num_buffers-1){

		//this should work? not 100% convinced but it seems ok
		buffer_sizes[idx] = buffers[idx+1] - buffers[idx];
	} else {

		buffer_sizes[idx] = num_keys - (buffers[idx] - keys);

	}

	return;


}


//a variant of count_off that uses binary search instead
//this would avoid atomics and should reduce memory usage - myaybe faster :o
//requires the keys to be sorted hashes - otherwise the results are just junk
//something a la two passes may be necessary to save space / communication
// threads set their boundaries - and then set their sizes

//revised work pipeline
// 1) Set all offsets to keys here based on relative offset + keys - skips the launch call later - TODO: double check that (keys + offset) - keys == offset. -- cpp says this works
// 2) subtract sets of keys from each other to get the relative offsets - these will give offsets, last key needs to subtract from origin pointer
// this means that the keys here are set to point to the START of their bucket
__global__ void set_buffers_binary(QF * qf, uint64_t num_keys, uint64_t slots_per_lock, uint64_t * keys, uint64_t num_buffers, uint64_t ** buffers, uint64_t value, uint8_t flags){

		int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= num_buffers) return;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		uint64_t boundary = (slots_per_lock*idx); //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;


		uint64_t lower = 0;
		uint64_t upper = num_keys;
		uint64_t index = upper-lower;

		//upper is non inclusive bound


		//if we exceed bounds that's our index
		while (upper != lower){


			index = lower + (upper - lower)/2;

			if ((keys[index] >> qf->metadata->bits_per_slot) < boundary){

				//false - the list before this point can be removed
				lower = index+1;

				//jump to a new midpoint
				


			} else if (index==0){

				//will this fix? otherwise need to patch via round up
				upper = index;

			} else if ((keys[index-1] >> qf->metadata->bits_per_slot) < boundary) {

				//set index! this is the first instance where I am valid and the next isnt
				//buffers[idx] = keys+index;
				break;

			} else {

				//we are too far right, all keys to the right do not matter
				upper = index;


			}

		}

		//we either exited or have an edge condition:
		//upper == lower iff 0 or max key
		index = lower + (upper - lower)/2;


		buffers[idx] = keys + index;
		


}



//revised work pipeline
// 1) Set all offsets to keys here based on relative offset + keys - skips the launch call later - TODO: double check that (keys + offset) - keys == offset. -- cpp says this works
// 2) subtract sets of keys from each other to get the relative offsets - these will give offsets, last key needs to subtract from origin pointer
// this means that the keys here are set to point to the START of their bucket
__global__ void set_buffers_per_thread(QF * qf, uint64_t num_keys, uint64_t * keys, uint64_t nthreads, uint64_t ** buffers, uint64_t value, uint8_t flags){

		int idx = threadIdx.x + blockDim.x * blockIdx.x;

		if (idx >= nthreads) return;

		//since we are finding all boundaries, we only need

		//printf("idx %llu\n", idx);

		//this sounds right? - they divide to go back so I think this is fine
		//this seems to break for reasons we do not fully understand
		uint64_t boundary_idx = (num_keys*idx)/nthreads; //<< qf->metadata->bits_per_slot;


		//This is the code I'm stealing that assumption from
		//uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);	
		//uint64_t lock_index = hash_bucket_index / slots_per_lock;
		uint64_t boundary_slot = keys[boundary_idx] >> qf->metadata->bits_per_slot;

		while (boundary_idx != 0){

			//loop and find the next slot
			if (boundary_slot != (keys[boundary_idx-1] >> qf->metadata->bits_per_slot)){

				//this is the start of a bound, return here 
				
				break;
			}

			boundary_idx--;

		}

		buffers[idx] = keys + boundary_idx;

		//upper is non inclusive bound


}


//given the allocated buffers, start inserting
//may make sense to allocate the buffers as one contiguous array with uint64** directing to each section - ask Kathy/Prashant
__global__ void count_insert(QF * qf, uint64_t num_keys, uint64_t slots_per_lock, uint64_t * keys, uint64_t num_buffers, uint64_t ** buffers, volatile uint64_t * buffer_counts, uint64_t value, uint8_t flags){


	int idx = threadIdx.x + blockDim.x * blockIdx.x;



	if (idx >= num_keys) return;



	uint64_t key = keys[idx];

    //adding back in hashing here - this is inefficient
    if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
  		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
  			key = MurmurHash64A(((void *)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
  		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
  			key = hash_64(key, BITMASK(qf->metadata->key_bits));
  	}

	uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));

	uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;

	uint64_t bucket_index = hash_bucket_index / slots_per_lock;

	uint64_t my_slot = atomicAdd( (long long unsigned int *) (buffer_counts + bucket_index), (long long unsigned int) 1);


	uint64_t to_insert = keys[idx];

	buffers[bucket_index][my_slot] = to_insert;



}

//a variant of count_insert that takes in hashed keys

__global__ void count_insert_hashed(QF * qf, uint64_t num_keys, uint64_t slots_per_lock, uint64_t * keys, uint64_t num_buffers, uint64_t ** buffers, volatile uint64_t * buffer_counts, uint64_t value, uint8_t flags){


	int idx = threadIdx.x + blockDim.x * blockIdx.x;



	if (idx >= num_keys) return;



	uint64_t hash = keys[idx];

	uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;

	uint64_t bucket_index = hash_bucket_index / slots_per_lock;

	uint64_t my_slot = atomicAdd( (long long unsigned int *) (buffer_counts + bucket_index), (long long unsigned int) 1);


	//uint64_t to_insert = keys[idx];

	buffers[bucket_index][my_slot] = hash;



}

//assign one thread per buffer, empty the buffer into the cqf
//this will get called twice, once for even buffers and once for odd
// so that locking isn't necessary
__global__ void insert_from_buffers(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, uint64_t evenness){


	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;



	if (idx >= num_buffers) return;


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	// if (buffer_counts[idx] > 0) {

	// 	quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

	// 	//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	// }
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK);

		//internal threadfence. Bad? actually seems to be fine
		//__threadfence();

	}

	__threadfence();




}

//insert from buffers using prehashed_data
__global__ void insert_from_buffers_hashed_onepass(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts){


	int idx = threadIdx.x + blockDim.x * blockIdx.x;



	if (idx >= num_buffers) return;


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	// if (buffer_counts[idx] > 0) {

	// 	quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

	// 	//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	// }
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK | QF_KEY_IS_HASH);

		//internal threadfence. Bad? actually seems to be fine
		//__threadfence();

	}

	__threadfence();




}

//insert from buffers using prehashed_data
__global__ void insert_from_buffers_hashed(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, uint64_t evenness){


	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;



	if (idx >= num_buffers) return;


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	// if (buffer_counts[idx] > 0) {

	// 	quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

	// 	//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	// }
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK | QF_KEY_IS_HASH);

		//internal threadfence. Bad? actually seems to be fine
		//costs a little bit of performance, not worth for 2 pass
		//__threadfence();

	}

	__threadfence();




}

//insert from buffers using prehashed_data
//and write to the global time counters
__global__ void insert_from_buffers_cycles(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, uint64_t evenness){


	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;



	if (idx >= num_buffers) return;

	uint64_t start = clock64();


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	// if (buffer_counts[idx] > 0) {

	// 	quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

	// 	//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	// }
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK | QF_KEY_IS_HASH);

		//internal threadfence. Bad? actually seems to be fine
		//costs a little bit of performance, not worth for 2 pass
		//__threadfence();

	}

	__threadfence();


	uint64_t end = clock64();

	total_cycle_counter += end-start;


}

__global__ void insert_from_buffers_locked(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, unsigned short int * locks, uint64_t evenness){


	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;



	if (idx >= num_buffers) return;


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	// if (buffer_counts[idx] > 0) {

	// 	quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

	// 	//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	// }

	while (true){

		//if 0 was empty
		if (atomicCAS(locks + idx, (unsigned short int) 0, (unsigned short int) 1) == 0){

			if (atomicCAS(locks + idx + 1, (unsigned short int) 0, (unsigned short int) 1) == 0){

				//grabbed both locks
				break;

			} else {

				assert(atomicCAS(locks+idx, (unsigned short int) 1, (unsigned short int) 0) == 1);

			}
		}

	}
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK | QF_KEY_IS_HASH);

		//internal threadfence. Bad? actually seems to be fine
		//costs a little bit of performance, not worth for 2 pass
		//__threadfence();

	}

	__threadfence();

	//we're done, but make sure we're done
	assert(atomicCAS(locks+idx+1, (unsigned short int) 1, (unsigned short int) 0) == 1);
	assert(atomicCAS(locks+idx, (unsigned short int) 1,(unsigned short int) 0) == 1);



}


//one pass version - with locking we can safely remove
__global__ void insert_from_buffers_locked_onepass(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, unsigned short int * locks, uint64_t evenness){


	int idx = (threadIdx.x + blockDim.x * blockIdx.x);



	if (idx >= num_buffers) return;


	//at the start, we sort
	//we are exceeding bounds by 1
	//quick_sort(buffers[idx], 0, buffer_counts[idx]-1,0);
	//no need to sort if empty - this will cause overflow as 0-1 == max_uint
	// if (buffer_counts[idx] > 0) {

	// 	quick_sort(buffers[idx], 0, buffer_counts[idx]-1, 0);

	// 	//assert(assert_sorted(buffers[idx], buffer_counts[idx]));

	// }

	while (true){

		//if 0 was empty
		if (atomicCAS(locks + idx, (unsigned short int) 0, (unsigned short int) 1) == 0){

			if (atomicCAS(locks + idx + 1, (unsigned short int) 0, (unsigned short int) 1) == 0){

				//grabbed both locks
				break;

			} else {

				assert(atomicCAS(locks+idx, (unsigned short int) 1, (unsigned short int) 0) == 1);

			}
		}

	}
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK | QF_KEY_IS_HASH);

		//internal threadfence. Bad? actually seems to be fine
		//costs a little bit of performance, not worth for 2 pass
		//__threadfence();

	}

	__threadfence();

	//we're done, but make sure we're done
	assert(atomicCAS(locks+idx+1, (unsigned short int) 1, (unsigned short int) 0) == 1);
	assert(atomicCAS(locks+idx, (unsigned short int) 1,(unsigned short int) 0) == 1);



}

//assign one thread per buffer, empty the buffer into the cqf
//this will get called twice, once for even buffers and once for odd
// so that locking isn't necessary
__global__ void insert_from_buffers_timed(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, uint64_t evenness, uint64_t * min, uint64_t * max, uint64_t * average, uint64_t * count){


	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;



	if (idx >= num_buffers) return;

	uint64_t start = clock64();
	

	uint64_t my_count = buffer_counts[idx];

	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK);

		//internal threadfence. Bad? actually seems to be fine
		__threadfence();

	}


	uint64_t end = clock64();

	uint64_t duration = end-start;

	atomicAdd((long long unsigned int *) average, (long long unsigned int) duration);
	atomicAdd((long long unsigned int *) count, (long long unsigned int) 1);
	atomicMin((long long unsigned int *) min, (long long unsigned int) duration);
	atomicMax((long long unsigned int *) max, (long long unsigned int) duration);
	//__threadfence();




}

__global__ void insert_from_buffers_utilization(QF* qf, uint64_t num_buffers, uint64_t** buffers, volatile uint64_t * buffer_counts, uint64_t evenness, uint64_t * insert_cycles, uint64_t * fence_cycles, uint64_t * sort_cycles){


	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;



	if (idx >= num_buffers) return;

	uint64_t start = clock64();
	

	

	uint64_t my_count = buffer_counts[idx];

	//quick_sort(buffers[idx], 0, my_count, 0);

	//assert_sorted(buffers[idx], 0, my_count);

	uint64_t end_sort = clock64();


	for (uint64_t i =0; i < my_count; i++){

		int ret = qf_insert(qf, buffers[idx][i], 0, 1, QF_NO_LOCK);

		//internal threadfence. Bad? actually seems to be fine
		

	}

	uint64_t end_insert = clock64();

	__threadfence();

	uint64_t end_fence = clock64();

	uint64_t duration_sort = end_sort - start;
	uint64_t duration_insert = end_insert-end_sort;
	uint64_t duration_fence = end_fence - end_insert;

	atomicAdd((long long unsigned int *) insert_cycles, (long long unsigned int) duration_insert);
	atomicAdd((long long unsigned int *) fence_cycles, (long long unsigned int) duration_fence);
	atomicAdd((long long unsigned int *) sort_cycles, (long long unsigned int) duration_sort);
	//atomicMin((long long unsigned int *) min, (long long unsigned int) duration);
	//atomicMax((long long unsigned int *) max, (long long unsigned int) duration);
	//__threadfence();




}

__global__ void print_counts(uint64_t num_locks, volatile uint64_t * buffer_counts){

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx != 0) return;

	for (uint64_t i =0; i < num_locks; i++){
		printf("%*llu", 8, i);
	}
	printf("\n");

	for (uint64_t i =0; i < num_locks; i++){
		printf("%*llu", 8, buffer_counts[i]);
	}
	printf("\n");


}

__host__ void create_buffers(QF * qf, uint64_t** buffers, volatile uint64_t * buffer_sizes, uint64_t num_buffers){

	uint64_t**buffers_host;

	CUDA_CHECK(cudaMallocHost((void**)&buffers_host, num_buffers*sizeof(uint64_t *)));

	uint64_t* buffer_sizes_host;

	CUDA_CHECK(cudaMallocHost((void **)&buffer_sizes_host, num_buffers*sizeof(uint64_t)));

	CUDA_CHECK(cudaMemcpy(buffer_sizes_host, (uint64_t *) buffer_sizes, num_buffers*sizeof(uint64_t), cudaMemcpyDeviceToHost));

	for (uint64_t i =0; i < num_buffers; i++){

		uint64_t * temp_buffer;
		CUDA_CHECK(cudaMalloc((void **)&temp_buffer, buffer_sizes_host[i]*sizeof(uint64_t) ));

		buffers_host[i] = temp_buffer;

	}


	CUDA_CHECK(cudaMemcpy(buffers, buffers_host, num_buffers*sizeof(uint64_t *), cudaMemcpyHostToDevice));

	CUDA_CHECK(cudaFreeHost(buffers_host));
	CUDA_CHECK(cudaFreeHost(buffer_sizes_host));


}

//save on malloc call and device_syncronize
__host__ void create_buffers_premalloced(QF*qf, uint64_t** buffers, uint64_t * buffer_backing, volatile uint64_t * buffer_sizes, uint64_t num_buffers){

	uint64_t**buffers_host;

	CUDA_CHECK(cudaMallocHost((void**)&buffers_host, num_buffers*sizeof(uint64_t *)));

	uint64_t* buffer_sizes_host;

	CUDA_CHECK(cudaMallocHost((void **)&buffer_sizes_host, num_buffers*sizeof(uint64_t)));

	CUDA_CHECK(cudaMemcpy(buffer_sizes_host, (uint64_t *) buffer_sizes, num_buffers*sizeof(uint64_t), cudaMemcpyDeviceToHost));


	uint64_t counter = 0;

	for (uint64_t i =0; i < num_buffers; i++){

		//uint64_t * temp_buffer;
		//CUDA_CHECK(cudaMalloc((void **)&temp_buffer, buffer_sizes_host[i]*sizeof(uint64_t) ));

		buffers_host[i] = buffer_backing+counter;
		counter += buffer_sizes_host[i];

	}

	cudaMemcpy(buffers, buffers_host, num_buffers*sizeof(uint64_t *), cudaMemcpyHostToDevice);
	CUDA_CHECK(cudaFreeHost(buffers_host));
	CUDA_CHECK(cudaFreeHost(buffer_sizes_host));


}

__host__ void free_buffers_premalloced(QF *qf, uint64_t**buffers, uint64_t * buffer_backing, volatile uint64_t*buffer_sizes, uint64_t num_buffers){



	//free main buffers and sizes
	CUDA_CHECK(cudaFree(buffers));
	CUDA_CHECK(cudaFree((uint64_t *) buffer_sizes));
	cudaFree(buffer_backing);

	//CUDA_CHECK(cudaMemcpy(buffers, buffers_host, num_buffers*sizeof(uint64_t *), cudaMemcpyHostToDevice));





}

__host__ void free_buffers_nobacking(QF *qf, uint64_t**buffers, volatile uint64_t*buffer_sizes, uint64_t num_buffers){



	//free main buffers and sizes
	CUDA_CHECK(cudaFree(buffers));
	CUDA_CHECK(cudaFree((uint64_t *) buffer_sizes));
	//cudaFree(buffer_backing);

	//CUDA_CHECK(cudaMemcpy(buffers, buffers_host, num_buffers*sizeof(uint64_t *), cudaMemcpyHostToDevice));





}

__host__ void free_buffers(QF *qf, uint64_t**buffers, volatile uint64_t*buffer_sizes, uint64_t num_buffers){


	uint64_t**buffers_host;

	CUDA_CHECK(cudaMallocHost((void**)&buffers_host, num_buffers*sizeof(uint64_t *)));

	CUDA_CHECK(cudaMemcpy(buffers_host, buffers, num_buffers*sizeof(uint64_t), cudaMemcpyDeviceToHost));

	for (uint64_t i =0; i < num_buffers; i++){

		CUDA_CHECK(cudaFree(buffers_host[i]));

	}

	//free main buffers and sizes
	CUDA_CHECK(cudaFree(buffers));
	CUDA_CHECK(cudaFree((uint64_t *) buffer_sizes));

	//CUDA_CHECK(cudaMemcpy(buffers, buffers_host, num_buffers*sizeof(uint64_t *), cudaMemcpyHostToDevice));



	CUDA_CHECK(cudaFreeHost(buffers_host));


}

__host__ void bulk_insert_bucketing_premalloc(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags) {

	uint64_t key_block_size = 24;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts

	volatile uint64_t * buffer_sizes;
	CUDA_CHECK(cudaMalloc((void **) & buffer_sizes, num_locks*sizeof(uint64_t)));
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t)));
	uint64_t ** buffers;
	CUDA_CHECK(cudaMalloc((void **)&buffers, num_locks*sizeof(uint64_t*)));
	uint64_t * buffer_backing;
	cudaMalloc((void **)& buffer_backing, nvals*sizeof(uint64_t));

	//printf("Number of items to be inserted: %llu\n", nvals);


	//sort first
	//sort<<<key_block, key_block_size>>>(keys, nvals);


	count_off<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	
	cudaDeviceSynchronize();

	create_buffers_premalloced(qf, buffers, buffer_backing, buffer_sizes, num_locks);
	

	//reset sizes
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	count_insert<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);




	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);

	cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

  bulk_quick_sort<<<key_block, key_block_size>>>(num_locks, buffers, buffer_sizes);

	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end-start;

  std::cout << "Sorted " << nvals << " in " << diff.count() << " seconds\n";

  printf("Items Sorted per second: %f\n", nvals/diff.count());




	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	evenness = 1;

	insert_from_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);


	//free materials;
	//TODO:
	free_buffers_premalloced(qf, buffers, buffer_backing,  buffer_sizes, num_locks);

}

__host__ void bulk_insert_bucketing_buffer_provided(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, uint64_t * buffer_backing, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts


	
	//auto start_setup = std::chrono::high_resolution_clock::now();


	count_off<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	
	//cudaDeviceSynchronize();


	create_buffers_premalloced(qf, buffers, buffer_backing, buffer_sizes, num_locks);
	//cudaDeviceSynchronize();

	//reset sizes
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	count_insert<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);


	//these can go at the end
	//cudaDeviceSynchronize();

	//auto end_setup = std::chrono::high_resolution_clock::now();

	//std::chrono::duration<double> diff = end_setup-start_setup;

	//printf("Num items: %llu, num_locks: %llu\n", nvals, num_locks);

  //std::cout << "Setup in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);





	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	//cudaDeviceSynchronize();
	//auto first = std::chrono::high_resolution_clock::now();

	//diff = first-end_setup;

  //std::cout << "First finished in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	evenness = 1;

	insert_from_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);


	//free materials;
	//cudaDeviceSynchronize();
	//auto second = std::chrono::high_resolution_clock::now();

	//diff = second-first;

  //std::cout << "Second finished in " << diff.count() << " seconds\n";

}

//adding timers back in for profiling
__host__ void bulk_insert_bucketing_buffer_provided_timed(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, uint64_t * buffer_backing, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts

	printf("Per Item num blocks, block dim: (%llu, %llu): %llu threads\n", key_block, key_block_size, key_block*key_block_size);
	printf("Locking num blocks, block dim: (%llu, %llu): %llu threads\n", (num_locks-1)/key_block_size+1, key_block_size, ((num_locks-1)/key_block_size+1)*key_block_size);

	uint64_t * insert_timer;
	uint64_t * fence_timer;
	uint64_t * sort_timer;

	//set these timers to 0
	cudaMallocManaged((void**)&insert_timer, sizeof(uint64_t));
	cudaMallocManaged((void**)&fence_timer, sizeof(uint64_t));
	cudaMallocManaged((void**)&sort_timer, sizeof(uint64_t));

	insert_timer[0] = 0;
	fence_timer[0] = 0;
	sort_timer[0] = 0;


	
	auto start_setup = std::chrono::high_resolution_clock::now();

	//sort!

	thrust::sort(thrust::device, keys, keys+nvals);
	//thrust::sort(thrust::host, A, A + N, thrust::greater<int>());



	
	count_off<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	
	//cudaDeviceSynchronize();


	create_buffers_premalloced(qf, buffers, buffer_backing, buffer_sizes, num_locks);
	//cudaDeviceSynchronize();

	//reset sizes
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	count_insert<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);


	//these can go at the end
	cudaDeviceSynchronize();

	auto end_setup = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end_setup-start_setup;

	//printf("Num items: %llu, num_locks: %llu\n", nvals, num_locks);

  std::cout << "Setup in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);





	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers_utilization<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness, insert_timer, fence_timer, sort_timer);
	

	cudaDeviceSynchronize();
	auto first = std::chrono::high_resolution_clock::now();

	diff = first-end_setup;

  std::cout << "First finished in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	evenness = 1;

	insert_from_buffers_utilization<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness, insert_timer, fence_timer, sort_timer);


	//free materials;
	cudaDeviceSynchronize();
	auto second = std::chrono::high_resolution_clock::now();

	diff = second-first;

  std::cout << "Second finished in " << diff.count() << " seconds\n";

  diff = second - start_setup;

  std::cout << "inserts per second: " << nvals/diff.count() << "\n";

  std::cout << "Summed cycles across " << num_locks << " theads\n";
  std::cout << std::fixed << "cycles to sort: " << (*sort_timer)  << " cycles\n";
  std::cout << std::fixed << "cycles to insert: " << (*insert_timer) << " cycles\n";
  std::cout << std::fixed << "cycles to __threadfence(): " << (*fence_timer)  << " cycles\n";

  cudaFree(insert_timer);
  cudaFree(fence_timer);
  cudaFree(sort_timer);

}

__global__ void reset_global_counters(){

	uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx !=0) return;

	total_cycle_counter = 0;
	memmove_cycle_counter = 0;
}

__global__ void read_global_counters(){

	uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx !=0) return;

	printf("Total cycles used in insert: %llu\n", total_cycle_counter);
	printf("Memmove cycles used in insert: %llu\n", memmove_cycle_counter);
}


//kernel to assert that a device-side list is sorted
__global__ void assert_sorted_kernel(uint64_t * vals, uint64_t low, uint64_t high){

	uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= high-1) return;

	uint64_t my_val = vals[idx];
	uint64_t next_val = vals[idx+1];

	assert(my_val <= next_val);
}

//modified version of buffers_provided - performs an initial bulk hash, should save work over other versions
//note: this DOES modify the given buffer - fine for all versions now
//This variant performs an ititial sort that allows us to save time overall
//as we avoid the atomic count-off and any sort of cross-thread communication
__host__ void bulk_insert_no_atomics_timed(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts


	
	auto start_hash = std::chrono::high_resolution_clock::now();

	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);

	cudaDeviceSynchronize();


	auto end_hash = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end_hash-start_hash;


	std::cout << "hashed in " << diff.count() << " seconds\n";


	thrust::sort(thrust::device, keys, keys+nvals);


	cudaDeviceSynchronize();

	auto end_sort = std::chrono::high_resolution_clock::now();

	diff = end_sort-end_hash;


	std::cout << "sorted in " << diff.count() << " seconds\n";


	set_buffers_binary<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, value, flags);

	cudaDeviceSynchronize();

	auto end_set = std::chrono::high_resolution_clock::now();

	diff = end_set - end_sort;


	std::cout << "set buffers in " << diff.count() << " seconds\n";



	set_buffer_lens<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, num_locks, (uint64_t *) buffer_sizes, buffers);


	cudaDeviceSynchronize();


	auto end_set_lens = std::chrono::high_resolution_clock::now();

	diff = end_set_lens - end_set;

	std::cout << "set buf lens in " << diff.count() << " seconds\n";





	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers_cycles<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	//cudaDeviceSynchronize();
	//auto first = std::chrono::high_resolution_clock::now();

	//diff = first-end_setup;

  //std::cout << "First finished in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

  cudaDeviceSynchronize();

	auto first_insert = std::chrono::high_resolution_clock::now();

	diff = first_insert - end_set_lens;


	std::cout << "even buffers dumped in " << diff.count() << " seconds\n";

	evenness = 1;

	insert_from_buffers_cycles<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);

	cudaDeviceSynchronize();

	auto second_insert = std::chrono::high_resolution_clock::now();

	diff = second_insert - first_insert;


	std::cout << "odd buffers dumped in " << diff.count() << " seconds\n";

	evenness = 1;
	//free materials;
	//cudaDeviceSynchronize();
	//auto second = std::chrono::high_resolution_clock::now();

	//diff = second-first;

  //std::cout << "Second finished in " << diff.count() << " seconds\n";

}

//modified version of buffers_provided - performs an initial bulk hash, should save work over other versions
//note: this DOES modify the given buffer - fine for all versions now
//This variant performs an ititial sort that allows us to save time overall
//as we avoid the atomic count-off and any sort of cross-thread communication
__host__ void bulk_insert_no_atomics(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts



	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);


	thrust::sort(thrust::device, keys, keys+nvals);


	set_buffers_binary<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, value, flags);

	
	set_buffer_lens<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, num_locks, (uint64_t *) buffer_sizes, buffers);


	//insert_from_buffers_hashed_onepass<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes);
	
	//return;

	uint64_t evenness = 0;

	insert_from_buffers_hashed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	evenness = 1;

	insert_from_buffers_hashed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);


}

//modified version of buffers_provided - this one uses locking to increase the # threads that can work
__host__ void bulk_insert_locking(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes, unsigned short int * locks) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts



	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);


	thrust::sort(thrust::device, keys, keys+nvals);


	set_buffers_binary<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, value, flags);

	
	set_buffer_lens<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, num_locks, (uint64_t *) buffer_sizes, buffers);


	//insert_from_buffers_hashed_onepass<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes);
	
	//return;

	uint64_t evenness = 0;

	insert_from_buffers_locked_onepass<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, locks, evenness);
	

	//evenness = 1;

	//insert_from_buffers_locked<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, locks, evenness);


}

//modified version of buffers_provided - performs an initial bulk hash, should save work over other versions
//note: this DOES modify the given buffer - fine for all versions now
//This variant performs an ititial sort that allows us to save time overall
//as we avoid the atomic count-off and any sort of cross-thread communication
__host__ void bulk_insert_per_thread(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t nthreads, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts



	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);


	thrust::sort(thrust::device, keys, keys+nvals);


	set_buffers_per_thread<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, nthreads, buffers, value, flags);


	set_buffer_lens<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, nthreads, (uint64_t *) buffer_sizes, buffers);


	//insert_from_buffers_hashed_onepass<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes);
	
	//return;

	uint64_t evenness = 0;

	insert_from_buffers_hashed<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nthreads, buffers, buffer_sizes, evenness);
	

	evenness = 1;

	insert_from_buffers_hashed<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nthreads, buffers, buffer_sizes, evenness);


}


//modified version of buffers_provided - performs an initial bulk hash, should save work over other versions
//note: this DOES modify the given buffer - fine for all versions now
//This variant performs an ititial sort that allows us to save time overall
//as we avoid the atomic count-off and any sort of cross-thread communication
__host__ void bulk_insert_per_thread_timed(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t nthreads, uint8_t flags, uint64_t ** buffers, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts

	cudaDeviceSynchronize();

	auto start_hash = std::chrono::high_resolution_clock::now();

	
	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);

	cudaDeviceSynchronize();

	auto end_hash = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end_hash-start_hash;


	std::cout << "hashed in " << diff.count() << " seconds\n";


	thrust::sort(thrust::device, keys, keys+nvals);


	cudaDeviceSynchronize();

	auto end_sort = std::chrono::high_resolution_clock::now();

	diff = end_sort - end_hash;


	std::cout << "sorted in " << diff.count() << " seconds\n";

	set_buffers_per_thread<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, nthreads, buffers, value, flags);

	cudaDeviceSynchronize();

	auto end_set_buf = std::chrono::high_resolution_clock::now();

	diff = end_set_buf - end_sort;


	std::cout << "set_bufs in " << diff.count() << " seconds\n";

	set_buffer_lens<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, nthreads, (uint64_t *) buffer_sizes, buffers);


	cudaDeviceSynchronize();

	auto end_set_lens = std::chrono::high_resolution_clock::now();

	diff = end_set_lens - end_set_buf;


	std::cout << "set_lens in " << diff.count() << " seconds\n";


	//insert_from_buffers_hashed_onepass<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes);
	
	//return;

	uint64_t evenness = 0;

	insert_from_buffers_hashed<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nthreads, buffers, buffer_sizes, evenness);

	cudaDeviceSynchronize();

	auto end_one = std::chrono::high_resolution_clock::now();

	diff = end_one - end_set_lens;


	std::cout << "insert 1 in " << diff.count() << " seconds\n";
	

	evenness = 1;

	insert_from_buffers_hashed<<<(nthreads-1)/key_block_size+1, key_block_size>>>(qf, nthreads, buffers, buffer_sizes, evenness);

	cudaDeviceSynchronize();

	auto end_all = std::chrono::high_resolution_clock::now();

	diff = end_all - end_one;


	std::cout << "insert 2 in " << diff.count() << " seconds\n";


}

//modified version of buffers_provided - performs an initial bulk hash, should save work over other versions
//note: this DOES modify the given buffer
__host__ void bulk_insert_one_hash_timed(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, uint64_t * buffer_backing, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts


	
	auto start_hash = std::chrono::high_resolution_clock::now();

	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);

	cudaDeviceSynchronize();


	auto end_hash = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end_hash-start_hash;


	std::cout << "hashed in " << diff.count() << " seconds\n";


	thrust::sort(thrust::device, keys, keys+nvals);


	cudaDeviceSynchronize();

	//assert_sorted_kernel<<<key_block, key_block_size>>>(keys, 0, nvals);

	cudaDeviceSynchronize();


	auto end_sort = std::chrono::high_resolution_clock::now();

	diff = end_sort-end_hash;


	std::cout << "sorted in " << diff.count() << " seconds\n";



	//reversal of work here to be more efficient: we set buffers THEN set sizes
	count_off_hashed<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	
	cudaDeviceSynchronize();

	auto end_count_off = std::chrono::high_resolution_clock::now();

	diff = end_count_off - end_sort;


	std::cout << "count off in " << diff.count() << " seconds\n";


	//NOTE: this call to premalloced uses nvals instead of buffer_backing: because the sort happens in place, we actually don't need a buffer for this func
	create_buffers_premalloced(qf, buffers, keys, buffer_sizes, num_locks);
	//cudaDeviceSynchronize();

	cudaDeviceSynchronize();

	auto end_buf = std::chrono::high_resolution_clock::now();

	diff = end_buf - end_count_off;


	std::cout << "set buffer sizes in " << diff.count() << " seconds\n";

	//NOTE: This version doesn't need it, as the items are already in their buckets
	//reset sizes
	//CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	cudaDeviceSynchronize();

	auto end_memset = std::chrono::high_resolution_clock::now();

	diff = end_memset - end_buf;


	std::cout << "reset buffers in " << diff.count() << " seconds\n";

	//NOTE: This version doesn't need this either, as the items are already in their buckets with correct sizes.
	//I think this step can be avoided if we sort 
	//do a count off
	//count_insert_hashed<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);


	cudaDeviceSynchronize();

	auto end_count_insert = std::chrono::high_resolution_clock::now();

	diff = end_count_insert - end_memset;


	std::cout << "filled buffers in " << diff.count() << " seconds\n";


	//we're set up, lets do a check

	uint64_t ** buffers_binary;
	uint64_t * buffer_sizes_binary;
	CUDA_CHECK(cudaMalloc((void **)&buffers_binary, num_locks*sizeof(uint64_t *)));
	CUDA_CHECK(cudaMalloc((void **)& buffer_sizes_binary, num_locks*sizeof(uint64_t)));

  //count_off_hashed<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);

	set_buffers_binary<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers_binary, value, flags);

	cudaDeviceSynchronize();

	compare_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(num_locks, buffers, buffers_binary);

	cudaDeviceSynchronize();


	set_buffer_lens<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, nvals, keys, num_locks, buffer_sizes_binary, buffers_binary);


	cudaDeviceSynchronize();

	compare_buffer_lens<<<(num_locks-1)/key_block_size+1, key_block_size>>>(num_locks, (uint64_t *) buffer_sizes, buffer_sizes_binary);

	cudaDeviceSynchronize();

	cudaFree(buffer_sizes_binary);
	cudaFree(buffers_binary);

	//these can go at the end
	//cudaDeviceSynchronize();

	//auto end_setup = std::chrono::high_resolution_clock::now();

	//std::chrono::duration<double> diff = end_setup-start_setup;

	//printf("Num items: %llu, num_locks: %llu\n", nvals, num_locks);

  //std::cout << "Setup in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);



	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers_hashed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	//cudaDeviceSynchronize();
	//auto first = std::chrono::high_resolution_clock::now();

	//diff = first-end_setup;

  //std::cout << "First finished in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

  cudaDeviceSynchronize();

	auto first_insert = std::chrono::high_resolution_clock::now();

	diff = first_insert - end_count_insert;


	std::cout << "even buffers dumped in " << diff.count() << " seconds\n";

	evenness = 1;

	insert_from_buffers_hashed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);

	cudaDeviceSynchronize();

	auto second_insert = std::chrono::high_resolution_clock::now();

	diff = second_insert - first_insert;


	std::cout << "odd buffers dumped in " << diff.count() << " seconds\n";

	evenness = 1;
	//free materials;
	//cudaDeviceSynchronize();
	//auto second = std::chrono::high_resolution_clock::now();

	//diff = second-first;

  //std::cout << "Second finished in " << diff.count() << " seconds\n";

}

__host__ void bulk_insert_one_hash(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags, uint64_t ** buffers, uint64_t * buffer_backing, volatile uint64_t * buffer_sizes) {

	uint64_t key_block_size = 32;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts



	//keys are hashed, now need to treat them as hashed in all further functions
	hash_all<<<key_block, key_block_size>>>(qf, keys, keys, nvals, value, flags);

	


	thrust::sort(thrust::device, keys, keys+nvals);



	count_off_hashed<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	
	//cudaDeviceSynchronize();


	create_buffers_premalloced(qf, buffers, buffer_backing, buffer_sizes, num_locks);
	//cudaDeviceSynchronize();

	//reset sizes
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	//I think this step can be avoided if we hash
	count_insert_hashed<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);


	//these can go at the end
	//cudaDeviceSynchronize();

	//auto end_setup = std::chrono::high_resolution_clock::now();

	//std::chrono::duration<double> diff = end_setup-start_setup;

	//printf("Num items: %llu, num_locks: %llu\n", nvals, num_locks);

  //std::cout << "Setup in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);





	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers_hashed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	//cudaDeviceSynchronize();
	//auto first = std::chrono::high_resolution_clock::now();

	//diff = first-end_setup;

  //std::cout << "First finished in " << diff.count() << " seconds\n";

  //printf("Items Sorted per second: %f\n", nvals/diff.count());

	evenness = 1;

	insert_from_buffers_hashed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);


	//free materials;
	//cudaDeviceSynchronize();
	//auto second = std::chrono::high_resolution_clock::now();

	//diff = second-first;

  //std::cout << "Second finished in " << diff.count() << " seconds\n";

}


__host__ void bulk_insert_bucketing(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags) {

	uint64_t key_block_size = 24;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts

	volatile uint64_t * buffer_sizes;
	CUDA_CHECK(cudaMalloc((void **) & buffer_sizes, num_locks*sizeof(uint64_t)));
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t)));

	printf("Number of items to be inserted: %llu\n", nvals);

	auto bucketing_start = std::chrono::high_resolution_clock::now();

	count_off<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);
	auto count_timer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = count_timer-bucketing_start;

  std::cout << "Counted in " << diff.count() << " seconds\n";

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	uint64_t ** buffers;
	CUDA_CHECK(cudaMalloc((void **)&buffers, num_locks*sizeof(uint64_t*)));

	create_buffers(qf, buffers, buffer_sizes, num_locks);
	cudaDeviceSynchronize();

	//reset sizes
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	auto malloced = std::chrono::high_resolution_clock::now();

  diff = malloced-count_timer;

  std::cout << "Malloced in " << diff.count() << " seconds\n";

	count_insert<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);




	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);
	cudaDeviceSynchronize();


	auto fill= std::chrono::high_resolution_clock::now();

  diff = fill-malloced;

  std::cout << "Filled Buffers in " << diff.count() << " seconds\n";

	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);
	

	evenness = 1;

	insert_from_buffers<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness);


	cudaDeviceSynchronize();

	auto insert= std::chrono::high_resolution_clock::now();

  diff = insert-fill;

  std::cout << "Inserted in CQF in " << diff.count() << " seconds\n";

	//free materials;
	//TODO:
	free_buffers(qf, buffers, buffer_sizes, num_locks);

	cudaDeviceSynchronize();

	auto free= std::chrono::high_resolution_clock::now();

  diff = free-insert;

  std::cout << "Freed Buffers in " << diff.count() << " seconds\n";

  diff = free-bucketing_start;

  std::cout << "Total time is " << diff.count() << " seconds\n";


}

__host__ void bulk_insert_bucketing_timed(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t slots_per_lock, uint64_t num_locks, uint8_t flags) {

	uint64_t key_block_size = 24;
	uint64_t key_block = (nvals -1)/key_block_size + 1;
	//start with num_locks, get counts

	volatile uint64_t * buffer_sizes;
	CUDA_CHECK(cudaMalloc((void **) & buffer_sizes, num_locks*sizeof(uint64_t)));
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t)));

	printf("Number of items to be inserted: %llu\n", nvals);

	auto bucketing_start = std::chrono::high_resolution_clock::now();

	count_off<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffer_sizes, value, flags);
	cudaDeviceSynchronize();
	//(QF * qf, uint64_t num_keys, uint64_t * keys, uint64_t num_buffers, volatile uint64_t * buffer_counts, uint64_t value, uint8_t flags){
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);
	cudaDeviceSynchronize();
	fflush(stdout);

	auto count_timer = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> diff = count_timer-bucketing_start;

  std::cout << "Counted in " << diff.count() << " seconds\n";

	//counts look good!
	//copy over buffer to __host__, and malloc buffers
	uint64_t ** buffers;
	CUDA_CHECK(cudaMalloc((void **)&buffers, num_locks*sizeof(uint64_t*)));

	create_buffers(qf, buffers, buffer_sizes, num_locks);
	cudaDeviceSynchronize();

	//reset sizes
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0 ,num_locks*sizeof(uint64_t)));

	auto malloced = std::chrono::high_resolution_clock::now();

  diff = malloced-count_timer;

  std::cout << "Malloced in " << diff.count() << " seconds\n";

	count_insert<<<key_block, key_block_size>>>(qf, nvals, slots_per_lock, keys, num_locks, buffers, buffer_sizes, value, flags);




	//and launch
	//print_counts<<<1,1>>>(num_locks, buffer_sizes);
	cudaDeviceSynchronize();

	//lets malloc some buffers
	uint64_t * min;
	uint64_t * max;
	uint64_t * average;
	uint64_t * sum_avg;

	cudaMallocManaged((void**)&min, sizeof(uint64_t));
	cudaMallocManaged((void**)&max, sizeof(uint64_t));
	cudaMallocManaged((void**)&average, sizeof(uint64_t));
	cudaMallocManaged((void**)&sum_avg, sizeof(uint64_t));

	min[0] = 1LL<<60;
	max[0] = 0;
	average[0] = 0;
	sum_avg[0] = 0;


	auto fill= std::chrono::high_resolution_clock::now();

  diff = fill-malloced;

  std::cout << "Filled Buffers in " << diff.count() << " seconds\n";



	//time to insert
	uint64_t evenness = 0;

	insert_from_buffers_timed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness, min, max, average, sum_avg);
	
	cudaDeviceSynchronize();

	printf("Evens: Min: %llu, Max: %llu, average: %f\n", *min, *max, 1.0*average[0]/sum_avg[0]);
	printf("Min: %f, Max: %f, Avg: %f\n", (1.0)*(*min)/CYCLES_PER_SECOND, (1.0)*(*max)/CYCLES_PER_SECOND, 1.0*(1.0*average[0]/sum_avg[0])/CYCLES_PER_SECOND);
	min[0] = 1LL<<60;
	max[0] = 0;
	average[0] = 0;
	sum_avg[0] = 0;

	cudaDeviceSynchronize();

	evenness = 1;

	insert_from_buffers_timed<<<(num_locks-1)/key_block_size+1, key_block_size>>>(qf, num_locks, buffers, buffer_sizes, evenness, min, max, average, sum_avg);


	cudaDeviceSynchronize();

	printf("odds: Min: %llu, Max: %llu, average: %f\n", *min, *max, 1.0*average[0]/sum_avg[0]);
	printf("Min: %f, Max: %f, Avg: %f\n", (1.0)*(*min)/CYCLES_PER_SECOND, (1.0)*(*max)/CYCLES_PER_SECOND, 1.0*(1.0*average[0]/sum_avg[0])/CYCLES_PER_SECOND);
	


	auto insert= std::chrono::high_resolution_clock::now();

  diff = insert-fill;

  std::cout << "Inserted in CQF in " << diff.count() << " seconds\n";

	//free materials;
	//TODO:
	free_buffers(qf, buffers, buffer_sizes, num_locks);

	cudaDeviceSynchronize();

	auto free= std::chrono::high_resolution_clock::now();

  diff = free-insert;

  std::cout << "Freed Buffers in " << diff.count() << " seconds\n";

  diff = free-bucketing_start;

  std::cout << "Total time is " << diff.count() << " seconds\n";

  cudaFree(min);
  cudaFree(max);
  cudaFree(average);
  cudaFree(sum_avg);


}

//given a fill ratio and a start, calculate num_locks and insert
//returns the end of the nvals
__host__ uint64_t bulk_insert_bucketing_smart(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t start, double fill_ratio, uint64_t nslots, uint64_t xnslots, uint8_t flags){


	//first one will be two passes
	//< 50% and <95%
	uint64_t num_slots = nslots;


	double fill_ratio_start = 1.0*start/num_slots;

	uint64_t end_val = fill_ratio*num_slots;

	if (end_val > nvals){
		end_val = nvals;
		fill_ratio = 1.0*nvals/num_slots;
	}

	//calculate lock size

	uint64_t num_items_to_insert = end_val - start;

	//std::log(10)
	uint64_t num_slots_per_lock = 1.0*EXP_BEFORE_FAILURE * std::log(nvals) / std::log(fill_ratio);

	uint64_t num_locks = (xnslots - 1)/num_slots_per_lock+1;

	//can't just ask for xnslots

	printf("Moving fill ratio from %f to %f\n", fill_ratio_start, fill_ratio);

	printf("Inserting %llu items starting from %llu\n", num_items_to_insert, start);

	printf("Lock size: %llu, # locks %llu\n", num_slots_per_lock, num_locks);

	printf("Total slots in existence %llu, slots covered %llu\n", xnslots, num_locks*num_slots_per_lock);


	bulk_insert_bucketing(qf, keys+start, value, count, num_items_to_insert, num_slots_per_lock, num_locks, flags);

	return end_val;

}


//need a precalculation for max possible #locks
__host__ uint64_t bulk_insert_bucketing_smart_buffer_provided(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t start, double fill_ratio, uint64_t nslots, uint64_t xnslots, uint8_t flags, uint64_t ** buffers, uint64_t * buffer_backing, volatile uint64_t * buffer_sizes){


	//first one will be two passes
	//< 50% and <95%
	uint64_t num_slots = nslots;


	double fill_ratio_start = 1.0*start/num_slots;

	uint64_t end_val = fill_ratio*num_slots;

	if (end_val > nvals){
		end_val = nvals;
		fill_ratio = 1.0*nvals/num_slots;
	}

	//calculate lock size

	uint64_t num_items_to_insert = end_val - start;

	//std::log(10)
	uint64_t num_slots_per_lock = 2.0*EXP_BEFORE_FAILURE * std::log(nvals) / std::log(fill_ratio);

	uint64_t num_locks = (xnslots - 1)/num_slots_per_lock+1;

	//can't just ask for xnslots

	printf("Moving fill ratio from %f to %f\n", fill_ratio_start, fill_ratio);

	printf("Inserting %llu items starting from %llu\n", num_items_to_insert, start);

	printf("Lock size: %llu, # locks %llu\n", num_slots_per_lock, num_locks);

	printf("Total slots in existence %llu, slots covered %llu\n", xnslots, num_locks*num_slots_per_lock);


	//bulk_insert_bucketing(qf, keys+start, value, count, num_items_to_insert, num_slots_per_lock, num_locks, flags);

	bulk_insert_bucketing_buffer_provided_timed(qf, keys+start, value, count, nvals, num_slots_per_lock, num_locks, flags, buffers, buffer_backing, buffer_sizes);


	return end_val;

}

__host__ void bulk_insert_bucketing_steps(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, double fill_steps, uint64_t nslots, uint64_t xnslots, uint8_t flags){

	uint64_t start = 0;

	double fill_ratio = 0.0;

	while (fill_ratio < 1.0){

		fill_ratio+=fill_steps;

		start = bulk_insert_bucketing_smart(qf, keys, value, count, nvals, start, fill_ratio, nslots, xnslots, flags);

	}

	return;



	}




__global__ void qf_insert_evenness(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, volatile uint32_t* locks, int evenness, uint8_t flags) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	int n_threads = blockDim.x * gridDim.x;
	//start and end points in the keys array
	int start = nvals * idx / n_threads;
	int end = nvals * (idx + 1) / n_threads;

  //need to add check for if idx >> nvals
  if (n_threads >= nvals){
    start = idx;
    end = idx+1;
    if (idx >= nvals) return;
  }

  //nslots or xnslots?
  //int num_locks = qf->metadata->xnslots/NUM_SLOTS_TO_LOCK + 10;

	//printf("Thread %d/%d: start %d end %d\n", idx, n_threads, start, end);
	int i = start;
	while (i < end) {

    // if (i % 100 ==0){
    //   printf("Still Alive %d/%llu\n", i,nvals);
    // }
		uint64_t key = keys[i];

    //adding back in hashing here - this is inefficient
    if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
  		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
  			key = MurmurHash64A(((void *)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
  		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
  			key = hash_64(key, BITMASK(qf->metadata->key_bits));
  	}

		uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));

    //printf("%d insert %d, key: %llu, hash: %llu \n", idx, i, key, hash);
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);
		uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		uint64_t lock_index = hash_bucket_index / NUM_SLOTS_TO_LOCK;
    //if this succeeds, implies we have an overwrite error with the lock
    //uint64_t lock_index = 0;

		//if (hash_bucket_index % 2 == evenness) {
    if (true){
      //printf("Even so inserting\n");
      //printf("Idx %d grabbing bucket %llu, lock should be %llu, is %llu\n", idx, hash_bucket_index, hash_bucket_index / NUM_SLOTS_TO_LOCK, lock_index);
			if (get_lock(locks, lock_index) == 0) {

        if (get_lock(locks, lock_index+1) ==0){

					int ret = qf_insert(qf, keys[i], 0, 1, QF_NO_LOCK);
					if (ret < 0) {
						printf("failed insertion for key: %d %llu", i, keys[i]);
						if (ret == QF_NO_SPACE)
							printf(" because CQF is full.\n");
						else if (ret == QF_COULDNT_LOCK)
							printf(" because TRY_ONCE_LOCK failed.\n");
						else
							printf(" because program does not recognise return value.\n");

					}
					i++;
					__threadfence();
	        unlock(locks, lock_index+1);
					

			}
				unlock(locks, lock_index);
			} 
		}
		else {
			i++;
		}

	}
	return;
}


__global__ void qf_insert_evenness_nolock(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t num_locks, volatile uint32_t* locks, int evenness, uint8_t flags) {
	
	int idx = 2*(threadIdx.x + blockDim.x * blockIdx.x)+evenness;

	if (idx >= num_locks) return;


	int n_threads = blockDim.x * gridDim.x;
	//start and end points in the keys array
	int start = 0;
	int end = nvals;

  //nslots or xnslots?
  //int num_locks = qf->metadata->xnslots/NUM_SLOTS_TO_LOCK + 10;

	//printf("Thread %d/%d: start %d end %d\n", idx, n_threads, start, end);
	int i = start;
	while (i < end) {

    // if (i % 100 ==0){
    //   printf("Still Alive %d/%llu\n", i,nvals);
    // }
		uint64_t key = keys[i];

    //adding back in hashing here - this is inefficient
    if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
  		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
  			key = MurmurHash64A(((void *)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
  		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
  			key = hash_64(key, BITMASK(qf->metadata->key_bits));
  	}

		uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));

    //printf("%d insert %d, key: %llu, hash: %llu \n", idx, i, key, hash);
		//uint64_t hash_remainder = hash & BITMASK(qf->metadata->bits_per_slot);
		uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
		uint64_t lock_index = hash_bucket_index / NUM_SLOTS_TO_LOCK;
    //if this succeeds, implies we have an overwrite error with the lock

    if (lock_index != idx){
    	i++;
    	continue;

    }
    

					int ret = qf_insert(qf, keys[i], 0, 1, QF_NO_LOCK);
					if (ret < 0) {
						printf("failed insertion for key: %d %llu", i, keys[i]);
						if (ret == QF_NO_SPACE)
							printf(" because CQF is full.\n");
						else if (ret == QF_COULDNT_LOCK)
							printf(" because TRY_ONCE_LOCK failed.\n");
						else
							printf(" because program does not recognise return value.\n");

					
	      
					}
					i++;

	}
	return;
}



__global__ void bufferSanityCheck(uint64_t**buffers, uint32_t* bufferLens){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  //should never happen, but just in case
  if (idx != 0) return;

  for (int i =0; i < NUM_BUFFERS; i++){

    printf("Buffer %d, length: %llu:\n", i, bufferLens[i]);

    for (int j =0; j < 10; j++){
      printf("%llu ", buffers[i][j]);
    }
    printf("\n\n");

  }
}

__global__ void initBuffer(uint64_t** buffers, uint64_t* temp_buffer, int i){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  //should never happen, but just in case
  if (idx != 0) return;

  buffers[i] = temp_buffer;


}

//inserts a buffer into the qf
//at the moment, this just prints
__device__ void process_buffer(QF* qf, uint64_t * buffer, uint64_t bufferLen, uint64_t bufferNum){

  //grab the buffer, print it, and then free the memory
  // uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  //
  // if (idx != 0) return;

  printf("Processing full buffer %llu of size %llu:\n", bufferNum, bufferLen);

  for (int i =0; i < bufferLen; i++){
    printf("%llu ", buffer[i]);
  }
  printf("\n\n");
  //fflush(stdout);

  //done with buffer, free
  cudaFree(buffer);

}


//attempt to insert a hash into a buffer, if this fails then try again
__device__ int insert_into_buffer(QF* qf, uint64_t hashVal, cudaStream_t cstream, uint64_t ** buffers, uint32_t * buffer_lens, uint32_t * locks, uint64_t num_buffer){

  //whatever the current len is, grab it and assert not larger
  //increment 1, return old address
  uint64_t* buffer = buffers[num_buffer];

  uint64_t nextFree = atomicAdd(buffer_lens + num_buffer, 1);

  if (nextFree < MAX_BUFFER_SIZE){

    //update lock
    atomicAdd(locks+num_buffer,1);

    //perform insert
    buffer[nextFree] = hashVal;

    atomicDec(locks+num_buffer,1);

    return 0;

  } else if (nextFree == MAX_BUFFER_SIZE){

    //malloc and memset
    uint64_t * new_buffer;
    //these aren't allowed in device code?
    //cudaMalloc((void**)&new_buffer, sizeof(uint64_t*) * MAX_BUFFER_SIZE);
    //cudaMemset(new_buffer, 0, sizeof(uint64_t) * MAX_BUFFER_SIZE);

    uint64_t * temp_buffer;

    temp_buffer = buffer;

    //wait
    uint32_t active_locks = locks[num_buffer];
    while (active_locks != 0){
      active_locks = locks[num_buffer];
    }

    //overwrite buffer
    buffers[num_buffer] = new_buffer;
    buffer_lens[num_buffer] = 0;

    //atm this will lock my warp up
    process_buffer(qf, temp_buffer, MAX_BUFFER_SIZE, num_buffer);

  }

  //otherwise I failed
  return 1;

}


//fill the buffers
//as they get filled, export to the correct stream for processing
__global__ void fill_buffers(QF* qf, cudaStream_t * streams, uint64_t * hashes, uint64_t nvals, uint64_t max_hash, uint64_t **buffers, uint32_t*buffer_lens, uint32_t* locks){


  uint64_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  int n_threads = blockDim.x * gridDim.x;
	//start and end points in the keys array
	int start = nvals * idx / n_threads;
	int end = nvals * (idx + 1) / n_threads;

  //need to add check for if idx >> nvals
  if (n_threads >= nvals){
    start = idx;
    end = idx+1;
    if (idx >= nvals) return;
  }

  //for each hash, find associated buffer and fill
  for (int i = start; i < end; i++){
    uint64_t bufferid = hashes[start] * NUM_BUFFERS/max_hash;

    while (true){
      //1/bufferSize inserts should fail this
      int result = insert_into_buffer(qf, hashes[start], streams[bufferid], buffers, buffer_lens, locks, bufferid);
      if (result == 0) break;
      printf("Failed to insert into buffer\n");
    }

  }


}





__host__ void qf_bulk_hash_insert(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint32_t* QFlocks, uint8_t flags) {

  //first alloc hash space

  printf("Starting bulk insert\n");
  uint64_t* hashed;
	CUDA_CHECK(cudaMalloc((void**)&hashed, sizeof(uint64_t) * nvals));

	//hash items
	int block_size = 1024;
	int num_blocks = (nvals + block_size - 1) / block_size;

  printf("Before hash all\n");
  fflush(stdout);
	hash_all <<< num_blocks, block_size >>> (qf, keys, hashed, nvals, value, flags);

  cudaDeviceSynchronize();
  //now allocate buffers
  //atm these are arbitrary

  uint64_t** buffers;
  uint32_t* buffer_lens;
  uint32_t * locks;

  printf("mallocing buffers\n");
  fflush(stdout);
  //allocate
  CUDA_CHECK(cudaMalloc((void**)&buffers, sizeof(uint64_t*) * NUM_BUFFERS));

  printf("buffers allocated?\n");
  fflush(stdout);

  for (int i=0; i < NUM_BUFFERS; i++){

    printf("Working on %d\n", i);
    fflush(stdout);
    uint64_t *temp_buffer;
    CUDA_CHECK(cudaMalloc((void**)&temp_buffer, sizeof(uint64_t*) * MAX_BUFFER_SIZE));
    CUDA_CHECK(cudaMemset(temp_buffer, 0, sizeof(uint64_t) * MAX_BUFFER_SIZE));
    cudaDeviceSynchronize();

    //update buffer pointers
    //crap this needs to be set - the buffer[i] is device code :(
    initBuffer<<<1,1>>>(buffers, temp_buffer, i);
    cudaDeviceSynchronize();

  }

  //ok devices are buffered, lets see those insides




  CUDA_CHECK(cudaMalloc((void**)&buffer_lens, sizeof(uint32_t) * NUM_BUFFERS));
  CUDA_CHECK(cudaMemset(buffer_lens, 0, sizeof(uint32_t) * NUM_BUFFERS));

  CUDA_CHECK(cudaMalloc((void**)&locks, sizeof(uint32_t) * NUM_BUFFERS));
  CUDA_CHECK(cudaMemset(locks, 0, sizeof(uint32_t) * NUM_BUFFERS));

  bufferSanityCheck<<<1,1>>>(buffers, buffer_lens);

  //buffers should be good after this, lets start filling!

  //lets init streams
  cudaStream_t streams[NUM_BUFFERS];
  for (int i=0; i < NUM_BUFFERS; i++){
    cudaStreamCreate(&streams[i]);
  }

  fill_buffers<<<1,1>>>(qf, streams, hashed, nvals, qf->metadata->xnslots, buffers, buffer_lens, locks);

  cudaDeviceSynchronize();
  for (int i=0; i < NUM_BUFFERS; i++){
    cudaStreamDestroy(streams[i]);
  }


}

__host__ void qf_bulk_insert(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, volatile uint32_t* locks, uint8_t flags) {
	//todo: number of threads
	uint64_t evenness = 1;
	//num_blocks = 100;
	//block_size = 320;
  uint64_t block_size = 256;
	uint64_t num_blocks = (nvals - 1) / block_size +1;

	printf("%llu blocks of size %llu\n", num_blocks, block_size);
	qf_insert_evenness <<< num_blocks, block_size >>> (qf, keys, value, count, nvals, locks, evenness, flags);
//	printf("sizeofqf is %lu\n", qf->metadata->xnslots);
	//evenness = 0;
	//qf_insert_evenness <<< num_blocks, block_size >>> (qf, keys, value, count, nvals, locks, evenness, flags);

	//qf_dump_kernel<<<1,1>>>(qf);


}

//this func will take in a set of keys and consume them, freeing memory when done
// with locking / threadfence, we can launch as many of these as possible
__host__ void qf_bulk_insert_streaming(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, volatile uint32_t* locks, uint8_t flags) {

	cudaStream_t temp_stream;
	cudaStreamCreate(& temp_stream);

	uint64_t evenness = 1;
	uint64_t block_size = 32;
	uint64_t num_blocks = (nvals/400 - 1) / block_size +1;


	qf_insert_evenness <<< num_blocks, block_size, 0, temp_stream>>> (qf, keys, value, count, nvals, locks, evenness, flags);

	cudaFree(keys);

	cudaStreamDestroy(temp_stream);

}


__host__ void qf_bulk_insert_nolock(QF* qf, uint64_t* keys, uint64_t value, uint64_t count, uint64_t nvals, uint64_t num_locks, volatile uint32_t* locks, uint8_t flags) {
	//todo: number of threads
	uint64_t evenness = 1;
	int block_size = 32;
	int num_blocks = (num_locks-1)/block_size + 1;
	
  //int block_size = 1024;
	//int num_blocks = (nvals + block_size - 1) / block_size;
	qf_insert_evenness_nolock <<< num_blocks, block_size >>> (qf, keys, value, count, nvals, num_locks, locks, evenness, flags);
//	printf("sizeofqf is %lu\n", qf->metadata->xnslots);
	evenness = 0;
	qf_insert_evenness_nolock <<< num_blocks, block_size >>> (qf, keys, value, count, nvals, num_locks, locks, evenness, flags);

}


__host__ void copy_to_host(QF* host, QF* device) {
	qfruntime runtime;
	qfmetadata metadata;
	//qfblock* blocks = malloc(qf_get_total_size_in_bytes(device));

	//copy back to host
	//may need to resize host qf before copying back when we start to support resizing.
	CUDA_CHECK(cudaMemcpy(host->runtimedata, device->runtimedata, sizeof(qfruntime), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host->metadata, device->metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(host->blocks, device->blocks, qf_get_total_size_in_bytes(device), cudaMemcpyDeviceToHost));
}


__global__ void bulk_get(QF * qf, uint64_t * vals,  uint64_t nvals, uint64_t key_count, uint64_t * counter, uint8_t flags){

	int idx = threadIdx.x + blockDim.x * blockIdx.x;

  //should never happen, but just in case
  if (idx >= nvals) return;

		uint64_t count = qf_count_key_value(qf, vals[idx], 0, 0);

		if (count < key_count) {

			atomicAdd((long long unsigned int *)counter, (long long unsigned int) 1);

		}
}

__host__ uint64_t bulk_get_wrapper(QF * qf, uint64_t * vals, uint64_t nvals){

	uint64_t * misses;
	//this is fine, should never be triggered
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  cudaMemset(misses, 0, sizeof(uint64_t));

  bulk_get<<<(nvals-1)/512+1, 512>>>(qf, vals, nvals, 1, misses, QF_NO_LOCK);

  cudaDeviceSynchronize();
  uint64_t toReturn = *misses;

  cudaFree(misses);
  return toReturn;

}


__host__ void  qf_gpu_launch(QF* qf, uint64_t* vals, uint64_t nvals, uint64_t key_count, uint64_t nhashbits, uint64_t nslots) {

	QF* _qf;
	QF temp_qf;

	qfruntime* _runtime;
	qfmetadata* _metadata;
	qfblock* _blocks;

	//init cyle counter
	// CUDA_CHECK(cudaMallocManaged((void **)& memmove_cycle_counter, sizeof(uint64_t)));
	// CUDA_CHECK(cudaMallocManaged((void **)& total_cycle_counter, sizeof(uint64_t)));

	// //clear these for final anlysis
	// //memmove_cycle_counter[0] = 0;
	// //total_cycle_counter[0] = 0;
	// reset_global_counters<<<1,1>>>();
	// cudaDeviceSynchronize();

	auto start = std::chrono::high_resolution_clock::now();

  printf("Inside launch\n");
	CUDA_CHECK(cudaMalloc((void**)&_runtime, sizeof(qfruntime)));
	CUDA_CHECK(cudaMalloc((void**)&_metadata, sizeof(qfmetadata)));
	CUDA_CHECK(cudaMalloc((void**)&_blocks, qf_get_total_size_in_bytes(qf)));

	CUDA_CHECK(cudaMemcpy(_runtime, qf->runtimedata, sizeof(qfruntime), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(_metadata, qf->metadata, sizeof(qfmetadata), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(_blocks, qf->blocks, qf_get_total_size_in_bytes(qf), cudaMemcpyHostToDevice));
	temp_qf.runtimedata = _runtime;
	temp_qf.metadata = _metadata;
	temp_qf.blocks = _blocks;

	CUDA_CHECK(cudaMalloc((void**)&_qf, sizeof(QF)));
	CUDA_CHECK(cudaMemcpy((void**)_qf, &temp_qf, sizeof(QF), cudaMemcpyHostToDevice));

	//etodo: locks
  printf("Qf setup done\n");

	uint64_t* _vals;
	CUDA_CHECK(cudaMalloc(&_vals, sizeof(uint64_t) * nvals));

	CUDA_CHECK(cudaMemcpy(_vals, vals, sizeof(uint64_t) * nvals, cudaMemcpyHostToDevice));
	// uint64_t* _hashed;
	// CUDA_CHECK(cudaMalloc(&_hashed, sizeof(uint64_t) * nvals));
  //
	// //hash items
	// int block_size = 1024;
	// int num_blocks = (nvals + block_size - 1) / block_size;
	// hash_all <<< num_blocks, block_size >>> (_vals, _hashed, nvals, nhashbits);

	unsigned short int * _locks;

	//TODO: pass this down into bulk insert - bad practice to have code recalculate values
	int num_locks = qf->metadata->xnslots/NUM_SLOTS_TO_LOCK + 10;//todo: figure out nslots and why is 0

	uint64_t xnslots = qf->metadata->xnslots;
	uint64_t numslots = qf->metadata->nslots;

  cudaMalloc((void**)&_locks, sizeof(unsigned short int)*num_locks);
  cudaDeviceSynchronize();
  printf("Num locks %d\n", num_locks);
  fflush( stdout );
	CUDA_CHECK(cudaMemset( (uint32_t *) _locks, 0, sizeof(unsigned short int) * num_locks));
  printf("Locks set!\n");
	cudaDeviceSynchronize();


	//modified version
	//here we need to establish num locks
	//this only matters for bulk_insert_smart_buf_provided

	//5 % fill ratio to start - this will be different for # npoints in bm_gpu_only
	// uint64_t max_num_slots_per_lock = 1.0*EXP_BEFORE_FAILURE * std::log(nvals) / std::log(.05);

	// uint64_t max_num_locks = (xnslots - 1)/max_num_slots_per_lock+1;


	// volatile uint64_t * buffer_sizes;
	// CUDA_CHECK(cudaMalloc((void **) & buffer_sizes, max_num_locks*sizeof(uint64_t)));
	// CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0, max_num_locks*sizeof(uint64_t)));
	// uint64_t ** buffers;
	// CUDA_CHECK(cudaMalloc((void **)&buffers, max_num_locks*sizeof(uint64_t*)));
	// uint64_t * buffer_backing;
	// cudaMalloc((void **)& buffer_backing, nvals*sizeof(uint64_t));

	//here we designate the maximum number of threads that can participate

	// 100,00
	//const uint64_t MAX_THREADS = nvals/32000;


	//printf("MAX THEADS: %llu\n", MAX_THREADS);
	//regular version
	volatile uint64_t * buffer_sizes;
	CUDA_CHECK(cudaMalloc((void **) & buffer_sizes, num_locks*sizeof(uint64_t)));
	CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0, num_locks*sizeof(uint64_t)));
	uint64_t ** buffers;
	CUDA_CHECK(cudaMalloc((void **)&buffers, num_locks*sizeof(uint64_t*)));
	//uint64_t * buffer_backing;
	//cudaMalloc((void **)& buffer_backing, nvals*sizeof(uint64_t));




	// volatile uint64_t * buffer_sizes;
	// CUDA_CHECK(cudaMalloc((void **) & buffer_sizes, MAX_THREADS*sizeof(uint64_t)));
	// CUDA_CHECK(cudaMemset((uint64_t *) buffer_sizes, 0, MAX_THREADS*sizeof(uint64_t)));
	// uint64_t ** buffers;
	// CUDA_CHECK(cudaMalloc((void **)&buffers, MAX_THREADS*sizeof(uint64_t*)));
	//uint64_t * buffer_backing;
	//cudaMalloc((void **)& buffer_backing, nvals*sizeof(uint64_t));




	printf("Buffers allocated\n");
	cudaDeviceSynchronize();

	cudaProfilerStart();


	auto s_insert = std::chrono::high_resolution_clock::now();

  printf("Starting Bulk insert\n");
  fflush(stdout);
  //qf_bulk_insert(_qf, _vals, 0, 1, nvals, _lock, QF_NO_LOCK);
	//qf_bulk_insert_nolock(_qf, _vals, 0, 1, nvals, num_locks, _lock, QF_NO_LOCK);
	//bulk_insert_bucketing_timed(_qf, _vals, 0, 1, nvals, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK);

  //bulk smart insert
  //bulk_insert_bucketing_steps(_qf, _vals, 0, 1, nvals, .6, numslots, xnslots, QF_NO_LOCK);
  //printf("\n\nmidpoint: %llu \n\n", midpoint);

  //smarter inserts?
  //uint64_t midpoint =  bulk_insert_bucketing_smart_buffer_provided(_qf, _vals, 0, 1, nvals, 0, .5, numslots, xnslots, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  //bulk_insert_bucketing_smart_buffer_provided(_qf, _vals, 0, 1, nvals, midpoint, 1.0, numslots, xnslots, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);

  //bulk_insert_one_hash_timed(_qf, _vals, 0, 1, nvals, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);

  bulk_insert_no_atomics_timed(_qf, _vals, 0, 1, nvals, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_sizes);

  //bulk_insert_locking(_qf, _vals, 0, 1, nvals, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_sizes, _locks);

  //bulk_insert_per_thread_timed(_qf, _vals, 0, 1, nvals, MAX_THREADS, QF_NO_LOCK, buffers, buffer_sizes);

  //bulk_insert_bucketing_buffer_provided_timed(_qf, _vals, 0, 1, nvals, NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  

  //uint64_t end_slot = bulk_insert_bucketing_smart(_qf, _vals, 0, 1, nvals, midpoint, 1.0, numslots, xnslots, QF_NO_LOCK);
  //two step!
  //bulk_insert_bucketing_buffer_provided_timed(_qf, _vals, 0, 1, nvals/2, NUM_SLOTS_TO_LOCK/2, 2*num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);
  //bulk_insert_bucketing_buffer_provided_timed(_qf, _vals+nvals/2, 0, 1, nvals/2 + (nvals % 2), NUM_SLOTS_TO_LOCK, num_locks, QF_NO_LOCK, buffers, buffer_backing, buffer_sizes);


	cudaDeviceSynchronize();
  printf("Bulk Insert completed\n");

  cudaProfilerStop();
  

  free_buffers_nobacking(qf, buffers,  buffer_sizes, num_locks);

   auto s_end = std::chrono::high_resolution_clock::now();

  cudaDeviceSynchronize();


  auto end = std::chrono::high_resolution_clock::now();


  std::chrono::duration<double> diff2 = s_end-s_insert;


  std::cout << "Sans buffers, Inserted " << nvals << " in " << diff2.count() << " seconds\n";

  printf("Inserts per second: %f\n", nvals/diff2.count());


  std::chrono::duration<double> diff = end-start;

  std::cout << "Inserted " << nvals << " in " << diff.count() << " seconds\n";

  printf("Inserts per second: %f\n", nvals/diff.count());


  CUDA_CHECK(cudaMemcpy(_vals, vals, sizeof(uint64_t) * nvals, cudaMemcpyHostToDevice));


  uint64_t * misses;
  cudaMalloc((void **)&misses, sizeof(uint64_t));
  cudaMemset(misses, 0, sizeof(uint64_t));

  cudaDeviceSynchronize();

  read_global_counters<<<1,1>>>();

  cudaDeviceSynchronize();

  auto get_start = std::chrono::high_resolution_clock::now();

  bulk_get<<<(nvals-1)/1024+1, 1024>>>(_qf, _vals, nvals, key_count, misses, QF_NO_LOCK);

  cudaDeviceSynchronize();

  auto get_timer = std::chrono::high_resolution_clock::now();

  diff = get_timer-get_start;

  std::cout << "Searched for " << nvals << " in " << diff.count() << " seconds\n";

  printf("Gets per second: %f\n", nvals/diff.count());




  uint64_t * host_misses;
  cudaMallocHost((void **)&host_misses, sizeof(uint64_t));
  cudaMemcpy(host_misses, misses, sizeof(uint64_t), cudaMemcpyDeviceToHost);

  printf("Number of misses: %llu\n", *host_misses);

	CUDA_CHECK(cudaMemcpy((void**)&temp_qf, _qf, sizeof(QF), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

	CUDA_CHECK(cudaMemcpy((void*)qf->metadata, temp_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  //Copy over runtime as well?
  CUDA_CHECK(cudaMemcpy((void*)qf->runtimedata, temp_qf.runtimedata, sizeof(qfruntime), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();

  //metadata is copied so this is safe
  //erroring here because you can't check the metadata from a decice qf
  uint64_t total_size = qf_get_total_size_in_bytes(qf);

  CUDA_CHECK(cudaMemcpy((void*)qf->blocks, temp_qf.blocks, total_size, cudaMemcpyDeviceToHost));

	//copy arrays back to host

  cudaDeviceSynchronize();

  //done
  

	//copy_to_host(qf, temp_qf);



}

__host__ __device__ int qf_set_count(QF *qf, uint64_t key, uint64_t value, uint64_t count, uint8_t
								 flags)
{
	if (count == 0)
		return 0;

	uint64_t cur_count = qf_count_key_value(qf, key, value, flags);
	int64_t delta = count - cur_count;

	int ret;
	if (delta == 0)
		ret = 0;
	else if (delta > 0)
		ret = qf_insert(qf, key, value, delta, flags);
	else
		ret = qf_remove(qf, key, value, labs(delta), flags);

	return ret;
}

__host__ __device__ int qf_remove(QF *qf, uint64_t key, uint64_t value, uint64_t count, uint8_t
							flags)
{
	if (count == 0)
		return true;

	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}
	uint64_t hash = (key << qf->metadata->value_bits) | (value &
																											 BITMASK(qf->metadata->value_bits));
	return _remove(qf, hash, count, flags);
}

__host__ __device__ int qf_delete_key_value(QF *qf, uint64_t key, uint64_t value, uint8_t flags)
{
	uint64_t count = qf_count_key_value(qf, key, value, flags);
	if (count == 0)
		return true;

	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}
	uint64_t hash = (key << qf->metadata->value_bits) | (value &
																											 BITMASK(qf->metadata->value_bits));
	return _remove(qf, hash, count, flags);
}

__host__ __device__ uint64_t qf_count_key_value(const QF *qf, uint64_t key, uint64_t value,
														uint8_t flags)
{


	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}

	uint64_t hash = (key << qf->metadata->value_bits) | (value &
																											 BITMASK(qf->metadata->value_bits));
	uint64_t hash_remainder   = hash & BITMASK(qf->metadata->bits_per_slot);
	int64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;

	if (!is_occupied(qf, hash_bucket_index))
		return 0;

	int64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf,
																																hash_bucket_index-1)
		+ 1;
	if (runstart_index < hash_bucket_index)
		runstart_index = hash_bucket_index;

	/* printf("MC RUNSTART: %02lx RUNEND: %02lx\n", runstart_index, runend_index); */

	uint64_t current_remainder, current_count, current_end;
	do {
		current_end = decode_counter(qf, runstart_index, &current_remainder,
																 &current_count);
		if (current_remainder == hash_remainder)
			return current_count;
		runstart_index = current_end + 1;
	} while (!is_runend(qf, current_end));

	return 0;
}

__host__ __device__ uint64_t qf_query(const QF *qf, uint64_t key, uint64_t *value, uint8_t flags)
{
	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}
	uint64_t hash = key;
	uint64_t hash_remainder   = hash & BITMASK(qf->metadata->key_remainder_bits);
	int64_t hash_bucket_index = hash >> qf->metadata->key_remainder_bits;

	if (!is_occupied(qf, hash_bucket_index))
		return 0;

	int64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf,
																																hash_bucket_index-1)
		+ 1;
	if (runstart_index < hash_bucket_index)
		runstart_index = hash_bucket_index;

	/* printf("MC RUNSTART: %02lx RUNEND: %02lx\n", runstart_index, runend_index); */

	uint64_t current_remainder, current_count, current_end;
	do {
		current_end = decode_counter(qf, runstart_index, &current_remainder,
																 &current_count);
		*value = current_remainder & BITMASK(qf->metadata->value_bits);
		current_remainder = current_remainder >> qf->metadata->value_bits;
		if (current_remainder == hash_remainder) {
			return current_count;
		}
		runstart_index = current_end + 1;
	} while (!is_runend(qf, current_end));

	return 0;
}

__host__ __device__ int64_t qf_get_unique_index(const QF *qf, uint64_t key, uint64_t value,
														uint8_t flags)
{
	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}
	uint64_t hash = (key << qf->metadata->value_bits) | (value &
																											 BITMASK(qf->metadata->value_bits));
	uint64_t hash_remainder   = hash & BITMASK(qf->metadata->bits_per_slot);
	int64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;

	if (!is_occupied(qf, hash_bucket_index))
		return QF_DOESNT_EXIST;

	int64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf,
																																hash_bucket_index-1)
		+ 1;
	if (runstart_index < hash_bucket_index)
		runstart_index = hash_bucket_index;

	/* printf("MC RUNSTART: %02lx RUNEND: %02lx\n", runstart_index, runend_index); */

	uint64_t current_remainder, current_count, current_end;
	do {
		current_end = decode_counter(qf, runstart_index, &current_remainder,
																 &current_count);
		if (current_remainder == hash_remainder)
			return runstart_index;

		runstart_index = current_end + 1;
	} while (!is_runend(qf, current_end));

	return QF_DOESNT_EXIST;
}

enum qf_hashmode qf_get_hashmode(const QF *qf) {
	return qf->metadata->hash_mode;
}
uint64_t qf_get_hash_seed(const QF *qf) {
	return qf->metadata->seed;
}
__uint64_t qf_get_hash_range(const QF *qf) {
	return qf->metadata->range;
}

bool qf_is_auto_resize_enabled(const QF *qf) {
	if (qf->runtimedata->auto_resize == 1)
		return true;
	return false;
}
uint64_t qf_get_total_size_in_bytes(const QF *qf) {
	return qf->metadata->total_size_in_bytes;
}
uint64_t qf_get_nslots(const QF *qf) {
	return qf->metadata->nslots;
}
uint64_t qf_get_num_occupied_slots(const QF *qf) {
	pc_sync(&qf->runtimedata->pc_noccupied_slots);
	return qf->metadata->noccupied_slots;
}

uint64_t qf_get_num_key_bits(const QF *qf) {
	return qf->metadata->key_bits;
}
uint64_t qf_get_num_value_bits(const QF *qf) {
	return qf->metadata->value_bits;
}
uint64_t qf_get_num_key_remainder_bits(const QF *qf) {
	return qf->metadata->key_remainder_bits;
}
uint64_t qf_get_bits_per_slot(const QF *qf) {
	return qf->metadata->bits_per_slot;
}

uint64_t qf_get_sum_of_counts(const QF *qf) {
	pc_sync(&qf->runtimedata->pc_nelts);
	return qf->metadata->nelts;
}
uint64_t qf_get_num_distinct_key_value_pairs(const QF *qf) {
	pc_sync(&qf->runtimedata->pc_ndistinct_elts);
	return qf->metadata->ndistinct_elts;
}

void qf_sync_counters(const QF *qf) {
	pc_sync(&qf->runtimedata->pc_ndistinct_elts);
	pc_sync(&qf->runtimedata->pc_nelts);
	pc_sync(&qf->runtimedata->pc_noccupied_slots);
}

/* initialize the iterator at the run corresponding
 * to the position index
 */
int64_t qf_iterator_from_position(const QF *qf, QFi *qfi, uint64_t position)
{
	if (position == 0xffffffffffffffff) {
		qfi->current = 0xffffffffffffffff;
		qfi->qf = qf;
		return QFI_INVALID;
	}
	assert(position < qf->metadata->nslots);
	if (!is_occupied(qf, position)) {
		uint64_t block_index = position;
		uint64_t idx = bitselect(get_block(qf, block_index)->occupieds[0], 0);
		if (idx == 64) {
			while(idx == 64 && block_index < qf->metadata->nblocks) {
				block_index++;
				idx = bitselect(get_block(qf, block_index)->occupieds[0], 0);
			}
		}
		position = block_index * QF_SLOTS_PER_BLOCK + idx;
	}

	qfi->qf = qf;
	qfi->num_clusters = 0;
	qfi->run = position;
	qfi->current = position == 0 ? 0 : run_end(qfi->qf, position-1) + 1;
	if (qfi->current < position)
		qfi->current = position;

#ifdef LOG_CLUSTER_LENGTH
	qfi->c_info = (cluster_data* )calloc(qf->metadata->nslots/32,
																			 sizeof(cluster_data));
	if (qfi->c_info == NULL) {
		perror("Couldn't allocate memory for c_info.");
		exit(EXIT_FAILURE);
	}
	qfi->cur_start_index = position;
	qfi->cur_length = 1;
#endif

	if (qfi->current >= qf->metadata->nslots)
		return QFI_INVALID;
	return qfi->current;
}

int64_t qf_iterator_from_key_value(const QF *qf, QFi *qfi, uint64_t key,
																	 uint64_t value, uint8_t flags)
{
	if (key >= qf->metadata->range) {
		qfi->current = 0xffffffffffffffff;
		qfi->qf = qf;
		return QFI_INVALID;
	}

	qfi->qf = qf;
	qfi->num_clusters = 0;

	if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
		if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
			key = MurmurHash64A(((void *)&key), sizeof(key),
													qf->metadata->seed) % qf->metadata->range;
		else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			key = hash_64(key, BITMASK(qf->metadata->key_bits));
	}
	uint64_t hash = (key << qf->metadata->value_bits) | (value &
																											 BITMASK(qf->metadata->value_bits));

	uint64_t hash_remainder   = hash & BITMASK(qf->metadata->bits_per_slot);
	uint64_t hash_bucket_index = hash >> qf->metadata->bits_per_slot;
	bool flag = false;

	// If a run starts at "position" move the iterator to point it to the
	// smallest key greater than or equal to "hash".
	if (is_occupied(qf, hash_bucket_index)) {
		uint64_t runstart_index = hash_bucket_index == 0 ? 0 : run_end(qf,
																																	 hash_bucket_index-1)
			+ 1;
		if (runstart_index < hash_bucket_index)
			runstart_index = hash_bucket_index;
		uint64_t current_remainder, current_count, current_end;
		do {
			current_end = decode_counter(qf, runstart_index, &current_remainder,
																	 &current_count);
			if (current_remainder >= hash_remainder) {
				flag = true;
				break;
			}
			runstart_index = current_end + 1;
		} while (!is_runend(qf, current_end));
		// found "hash" or smallest key greater than "hash" in this run.
		if (flag) {
			qfi->run = hash_bucket_index;
			qfi->current = runstart_index;
		}
	}
	// If a run doesn't start at "position" or the largest key in the run
	// starting at "position" is smaller than "hash" then find the start of the
	// next run.
	if (!is_occupied(qf, hash_bucket_index) || !flag) {
		uint64_t position = hash_bucket_index;
		assert(position < qf->metadata->nslots);
		uint64_t block_index = position / QF_SLOTS_PER_BLOCK;
		uint64_t idx = bitselect(get_block(qf, block_index)->occupieds[0], 0);
		if (idx == 64) {
			while(idx == 64 && block_index < qf->metadata->nblocks) {
				block_index++;
				idx = bitselect(get_block(qf, block_index)->occupieds[0], 0);
			}
		}
		position = block_index * QF_SLOTS_PER_BLOCK + idx;
		qfi->run = position;
		qfi->current = position == 0 ? 0 : run_end(qfi->qf, position-1) + 1;
		if (qfi->current < position)
			qfi->current = position;
	}

	if (qfi->current >= qf->metadata->nslots)
		return QFI_INVALID;
	return qfi->current;
}

static int qfi_get(const QFi *qfi, uint64_t *key, uint64_t *value, uint64_t
									 *count)
{
	if (qfi_end(qfi))
		return QFI_INVALID;

	uint64_t current_remainder, current_count;
	decode_counter(qfi->qf, qfi->current, &current_remainder, &current_count);

	*value = current_remainder & BITMASK(qfi->qf->metadata->value_bits);
	current_remainder = current_remainder >> qfi->qf->metadata->value_bits;
	*key = (qfi->run << qfi->qf->metadata->key_remainder_bits) | current_remainder;
	*count = current_count;

	return 0;
}

int qfi_get_key(const QFi *qfi, uint64_t *key, uint64_t *value, uint64_t
								*count)
{
	*key = *value = *count = 0;
	int ret = qfi_get(qfi, key, value, count);
	if (ret == 0) {
		if (qfi->qf->metadata->hash_mode == QF_HASH_DEFAULT) {
			*key = 0; *value = 0; *count = 0;
			return QF_INVALID;
		} else if (qfi->qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
			*key = hash_64i(*key, BITMASK(qfi->qf->metadata->key_bits));
	}

	return ret;
}

int qfi_get_hash(const QFi *qfi, uint64_t *key, uint64_t *value, uint64_t
								 *count)
{
	*key = *value = *count = 0;
	return qfi_get(qfi, key, value, count);
}

int qfi_next(QFi *qfi)
{
	if (qfi_end(qfi))
		return QFI_INVALID;
	else {
		/* move to the end of the current counter*/
		uint64_t current_remainder, current_count;
		qfi->current = decode_counter(qfi->qf, qfi->current, &current_remainder,
																	&current_count);

		if (!is_runend(qfi->qf, qfi->current)) {
			qfi->current++;
#ifdef LOG_CLUSTER_LENGTH
			qfi->cur_length++;
#endif
			if (qfi_end(qfi))
				return QFI_INVALID;
			return 0;
		} else {
#ifdef LOG_CLUSTER_LENGTH
			/* save to check if the new current is the new cluster. */
			uint64_t old_current = qfi->current;
#endif
			uint64_t block_index = qfi->run / QF_SLOTS_PER_BLOCK;
			uint64_t rank = bitrank(get_block(qfi->qf, block_index)->occupieds[0],
															qfi->run % QF_SLOTS_PER_BLOCK);
			uint64_t next_run = bitselect(get_block(qfi->qf,
																							block_index)->occupieds[0],
																		rank);
			if (next_run == 64) {
				rank = 0;
				while (next_run == 64 && block_index < qfi->qf->metadata->nblocks) {
					block_index++;
					next_run = bitselect(get_block(qfi->qf, block_index)->occupieds[0],
															 rank);
				}
			}
			if (block_index == qfi->qf->metadata->nblocks) {
				/* set the index values to max. */
				qfi->run = qfi->current = qfi->qf->metadata->xnslots;
				return QFI_INVALID;
			}
			qfi->run = block_index * QF_SLOTS_PER_BLOCK + next_run;
			qfi->current++;
			if (qfi->current < qfi->run)
				qfi->current = qfi->run;
#ifdef LOG_CLUSTER_LENGTH
			if (qfi->current > old_current + 1) { /* new cluster. */
				if (qfi->cur_length > 10) {
					qfi->c_info[qfi->num_clusters].start_index = qfi->cur_start_index;
					qfi->c_info[qfi->num_clusters].length = qfi->cur_length;
					qfi->num_clusters++;
				}
				qfi->cur_start_index = qfi->run;
				qfi->cur_length = 1;
			} else {
				qfi->cur_length++;
			}
#endif
			return 0;
		}
	}
}

bool qfi_end(const QFi *qfi)
{
	if (qfi->current >= qfi->qf->metadata->xnslots /*&& is_runend(qfi->qf, qfi->current)*/)
		return true;
	return false;
}

/*
 * Merge qfa and qfb into qfc
 */
/*
 * iterate over both qf (qfa and qfb)
 * simultaneously
 * for each index i
 * min(get_value(qfa, ia) < get_value(qfb, ib))
 * insert(min, ic)
 * increment either ia or ib, whichever is minimum.
 */
void qf_merge(const QF *qfa, const QF *qfb, QF *qfc)
{
	QFi qfia, qfib;
	qf_iterator_from_position(qfa, &qfia, 0);
	qf_iterator_from_position(qfb, &qfib, 0);

	if (qfa->metadata->hash_mode != qfc->metadata->hash_mode &&
			qfa->metadata->seed != qfc->metadata->seed &&
			qfb->metadata->hash_mode  != qfc->metadata->hash_mode &&
			qfb->metadata->seed  != qfc->metadata->seed) {
		fprintf(stderr, "Output QF and input QFs do not have the same hash mode or seed.\n");
		exit(1);
	}

	uint64_t keya, valuea, counta, keyb, valueb, countb;
	qfi_get_hash(&qfia, &keya, &valuea, &counta);
	qfi_get_hash(&qfib, &keyb, &valueb, &countb);
	do {
		if (keya < keyb) {
			qf_insert(qfc, keya, valuea, counta, QF_NO_LOCK | QF_KEY_IS_HASH);
			qfi_next(&qfia);
			qfi_get_hash(&qfia, &keya, &valuea, &counta);
		}
		else {
			qf_insert(qfc, keyb, valueb, countb, QF_NO_LOCK | QF_KEY_IS_HASH);
			qfi_next(&qfib);
			qfi_get_hash(&qfib, &keyb, &valueb, &countb);
		}
	} while(!qfi_end(&qfia) && !qfi_end(&qfib));

	if (!qfi_end(&qfia)) {
		do {
			qfi_get_hash(&qfia, &keya, &valuea, &counta);
			qf_insert(qfc, keya, valuea, counta, QF_NO_LOCK | QF_KEY_IS_HASH);
		} while(!qfi_next(&qfia));
	}
	if (!qfi_end(&qfib)) {
		do {
			qfi_get_hash(&qfib, &keyb, &valueb, &countb);
			qf_insert(qfc, keyb, valueb, countb, QF_NO_LOCK | QF_KEY_IS_HASH);
		} while(!qfi_next(&qfib));
	}
}

/*
 * Merge an array of qfs into the resultant QF
 */
void qf_multi_merge(const QF *qf_arr[], int nqf, QF *qfr)
{
	int i;
	QFi qfi_arr[nqf];
	int smallest_idx = 0;
	uint64_t smallest_key = UINT64_MAX;
	for (i=0; i<nqf; i++) {
		if (qf_arr[i]->metadata->hash_mode != qfr->metadata->hash_mode &&
				qf_arr[i]->metadata->seed != qfr->metadata->seed) {
			fprintf(stderr, "Output QF and input QFs do not have the same hash mode or seed.\n");
			exit(1);
		}
		qf_iterator_from_position(qf_arr[i], &qfi_arr[i], 0);
	}

	DEBUG_CQF("Merging %d CQFs\n", nqf);
	for (i=0; i<nqf; i++) {
		DEBUG_CQF("CQF %d\n", i);
		DEBUG_DUMP(qf_arr[i]);
	}

	while (nqf > 1) {
		uint64_t keys[nqf];
		uint64_t values[nqf];
		uint64_t counts[nqf];
		for (i=0; i<nqf; i++)
			qfi_get_hash(&qfi_arr[i], &keys[i], &values[i], &counts[i]);

		do {
			smallest_key = UINT64_MAX;
			for (i=0; i<nqf; i++) {
				if (keys[i] < smallest_key) {
					smallest_key = keys[i]; smallest_idx = i;
				}
			}
			qf_insert(qfr, keys[smallest_idx], values[smallest_idx],
								counts[smallest_idx], QF_NO_LOCK | QF_KEY_IS_HASH);
			qfi_next(&qfi_arr[smallest_idx]);
			qfi_get_hash(&qfi_arr[smallest_idx], &keys[smallest_idx],
									 &values[smallest_idx],
							&counts[smallest_idx]);
		} while(!qfi_end(&qfi_arr[smallest_idx]));

		/* remove the qf that is exhausted from the array */
		if (smallest_idx < nqf-1)
			memmove(&qfi_arr[smallest_idx], &qfi_arr[smallest_idx+1],
							(nqf-smallest_idx-1)*sizeof(qfi_arr[0]));
		nqf--;
	}
	if (!qfi_end(&qfi_arr[0])) {
		uint64_t iters = 0;
		do {
			uint64_t key, value, count;
			qfi_get_hash(&qfi_arr[0], &key, &value, &count);
			qf_insert(qfr, key, value, count, QF_NO_LOCK | QF_KEY_IS_HASH);
			qfi_next(&qfi_arr[0]);
			iters++;
		} while(!qfi_end(&qfi_arr[0]));
		DEBUG_CQF("Num of iterations: %lu\n", iters);
	}

	DEBUG_CQF("%s", "Final CQF after merging.\n");
	DEBUG_DUMP(qfr);

	return;
}

/* find cosine similarity between two QFs. */
uint64_t qf_inner_product(const QF *qfa, const QF *qfb)
{
	uint64_t acc = 0;
	QFi qfi;
	const QF *qf_mem, *qf_disk;

	if (qfa->metadata->hash_mode != qfb->metadata->hash_mode &&
			qfa->metadata->seed != qfb->metadata->seed) {
		fprintf(stderr, "Input QFs do not have the same hash mode or seed.\n");
		exit(1);
	}

	// create the iterator on the larger QF.
	if (qfa->metadata->total_size_in_bytes > qfb->metadata->total_size_in_bytes)
	{
		qf_mem = qfb;
		qf_disk = qfa;
	} else {
		qf_mem = qfa;
		qf_disk = qfb;
	}

	qf_iterator_from_position(qf_disk, &qfi, 0);
	do {
		uint64_t key = 0, value = 0, count = 0;
		uint64_t count_mem;
		qfi_get_hash(&qfi, &key, &value, &count);
		if ((count_mem = qf_count_key_value(qf_mem, key, 0, QF_KEY_IS_HASH)) > 0) {
			acc += count*count_mem;
		}
	} while (!qfi_next(&qfi));

	return acc;
}

/* find cosine similarity between two QFs. */
void qf_intersect(const QF *qfa, const QF *qfb, QF *qfr)
{
	QFi qfi;
	const QF *qf_mem, *qf_disk;

	if (qfa->metadata->hash_mode != qfr->metadata->hash_mode &&
			qfa->metadata->seed != qfr->metadata->seed &&
			qfb->metadata->hash_mode  != qfr->metadata->hash_mode &&
			qfb->metadata->seed  != qfr->metadata->seed) {
		fprintf(stderr, "Output QF and input QFs do not have the same hash mode or seed.\n");
		exit(1);
	}

	// create the iterator on the larger QF.
	if (qfa->metadata->total_size_in_bytes > qfb->metadata->total_size_in_bytes)
	{
		qf_mem = qfb;
		qf_disk = qfa;
	} else {
		qf_mem = qfa;
		qf_disk = qfb;
	}

	qf_iterator_from_position(qf_disk, &qfi, 0);
	do {
		uint64_t key = 0, value = 0, count = 0;
		qfi_get_hash(&qfi, &key, &value, &count);
		if (qf_count_key_value(qf_mem, key, 0, QF_KEY_IS_HASH) > 0)
			qf_insert(qfr, key, value, count, QF_NO_LOCK | QF_KEY_IS_HASH);
	} while (!qfi_next(&qfi));
}
