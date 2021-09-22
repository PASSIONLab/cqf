#ifndef APPROX_FILTER
#define APPROX_FILTER

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "gqf.cuh"
#include "kmer.hpp"

#ifndef NUM_SLOTS_TO_LOCK
#define NUM_SLOTS_TO_LOCK (1ULL<<13)
#endif

template <typename T>
class ApproxFilter{



public:

	//Construct the filter on the local GPU, aiming for ~95% fill ratio based on the expected # of inputs.
	__host__ ApproxFilter(uint64_t approx_n_items);

	//Insert an item into the filter
	//If a copy of the item already exists, extract the other key
	// returns A,C,T,G
	//If this is the first instance of the item, return the null char
	//otherwise return some char associated with yes
	__device__ char insert_and_find(uint64_t pre_hashed, char ext);


	//return the count of the item (with or without this specific extension?)
	//free the memory associated with the object
	__host__ ~ApproxFilter();


private:

	QF* qf;
	uint16_t * locks;

}

#endif



ApproxFilter::ApproxFilter(uint64_t approx_n_items){




}