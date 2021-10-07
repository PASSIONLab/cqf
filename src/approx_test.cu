/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>
 *
 * ============================================================================
 */

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <time.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <openssl/rand.h>
#include <chrono>
#include<iostream>

#include "include/gqf_int.cuh"
#include "include/gqf_file.cuh"
#include "hashutil.cuh"
#include "include/gqf.cuh"
//#include "src/gqf.cu"

#define CYCLES_PER_SECOND 1601000000

#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "Please specify the first and second indices to test CQF.\n");
		exit(1);

	}
	QF* qf;

	auto setup_start =  std::chrono::high_resolution_clock::now();

	printf("Start of everything.\n");
	uint64_t qbits = atoi(argv[1]);
	uint64_t rbits = 8;
	uint64_t vbits = 8;
	uint64_t nhashbits = qbits + rbits;
	uint64_t nslots = (1ULL << qbits);
	// //this can be changed to change the % it fills up
	// uint64_t nvals = 95 * nslots / 100;
	// //uint64_t nvals =  nslots/2;
	// //uint64_t nvals = 4;
	// //uint64_t nvals = 1;
	// uint64_t key_count = 1;
	// uint64_t* vals;

	qf_malloc_device(&qf, qbits);


	uint64_t nvals = .5 * (1ULL << qbits);

	uint64_t * vals;


	// /* Initialise the CQF */
	// if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_INVERTIBLE, false, 0)) {
	// 	fprintf(stderr, "Can't allocate CQF.\n");
	// 	abort();
	// }



	// /*
	// if (!qf_initfile(&qf, nslots, nhashbits, 0, QF_HASH_NONE, 0,
	// 								 "/tmp/mycqf.file")) {
	// 	fprintf(stderr, "Can't allocate CQF.\n");
	// 	abort();
	// }
	// */
	// qf_set_auto_resize(&qf, false);
	// /* Generate random values */
	vals = (uint64_t*)malloc(nvals * sizeof(uint64_t));
	RAND_bytes((unsigned char*)vals, sizeof(*vals) * nvals);
	// //uint64_t* _vals;
	for (uint64_t i = 0; i < nvals; i++) {
		//nslots is the range - why are these different?
		vals[i] = (1 * vals[i]) % (1ULL << qbits);
	 	vals[i] = hash_64(vals[i], BITMASK(nhashbits));
	}


	uint8_t * first;
	uint8_t * second;

	first = (uint8_t * ) malloc(nvals*sizeof(uint8_t));
	second = (uint8_t *) malloc(nvals*sizeof(uint8_t));

	RAND_bytes((unsigned char*)first, sizeof(*first) * nvals);
	RAND_bytes((unsigned char*)second, sizeof(*second) * nvals);

	for (uint64_t i=0; i < nvals; i++){


		//0-4
		first[i] = first[i] % 5;

		//5-9
		second[i] = second[i] % 5 + 5;

	}

	//copy over

	uint64_t * dev_hashes;
	uint8_t * dev_firsts;
	uint8_t * dev_seconds;

	cudaMalloc((void ** )&dev_hashes, nvals*sizeof(uint64_t));
	cudaMalloc((void ** )&dev_firsts, nvals*sizeof(uint8_t));
	cudaMalloc((void ** )&dev_seconds, nvals*sizeof(uint8_t));


	cudaMemcpy(dev_hashes, vals, nvals*sizeof(uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_firsts, first, nvals*sizeof(uint8_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_seconds, second, nvals*sizeof(uint8_t), cudaMemcpyHostToDevice);



	uint64_t * counter1;
	uint64_t * counter2;

	cudaMallocManaged((void **)&counter1, sizeof(uint64_t));
	cudaMallocManaged((void **)&counter2, sizeof(uint64_t));

	counter1[0] = 0;
	counter2[0] = 0;


	uint64_t * max;
	uint64_t * min;
	uint64_t * total;


	cudaMallocManaged((void **)&max, sizeof(uint64_t));
	cudaMallocManaged((void **)&min, sizeof(uint64_t));
	cudaMallocManaged((void **)&total, sizeof(uint64_t));

	max[0] = 0;
	min[0] = 0;
	total[0] = 0;;
	// // vals = (uint64_t *) malloc(nvals * sizeof(uint64_t));
	// // for (uint64_t i =0l; i< nvals; i++){
	// // 	vals[i] = i;
	// // }

	// srand(0);
	// /* Insert keys in the CQF */
	// printf("starting kernel\n");
	// qf_gpu_launch(&qf, vals, nvals, key_count, nhashbits, nslots);
	// cudaDeviceSynchronize();

	// printf("GPU launch succeeded\n");
	// fflush(stdout);


	//remove slots per lock

	cudaDeviceSynchronize();

	auto setup_end =  std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> setup_diff = setup_end-setup_start;

	std::cout << "Setup done in " << setup_diff.count() << " seconds\n";


	cudaDeviceSynchronize();

	auto start =  std::chrono::high_resolution_clock::now();



	insert_multi_kmer_kernel<<<(nvals-1)/32 +1, 32>>>(qf, dev_hashes, dev_firsts, dev_seconds, nvals, counter1, max, min, total);
	insert_multi_kmer_kernel<<<(nvals-1)/32 +1, 32>>>(qf, dev_hashes, dev_firsts, dev_seconds, nvals, counter2, max, min, total);
    
	
	cudaDeviceSynchronize();

	auto end = std::chrono::high_resolution_clock::now();


  	std::chrono::duration<double> diff = end-start;


  	std::cout << "Sans buffers, Inserted " << nvals << " in " << diff.count() << " seconds\n";

 	printf("Inserts per second: %f\n", nvals/diff.count());

 	printf("Inserts per find: %f\n", 2*nvals/diff.count());

 	printf("Positive rate for first round: %llu/%llu: %f\n", counter1[0], nvals, 1.0*counter1[0]/nvals);
 	printf("Positive rate for second round: %llu/%llu: %f\n", counter2[0], nvals, 1.0*counter2[0]/nvals);


 	uint64_t found_nslots = host_qf_get_nslots(qf);
	uint64_t occupied = host_qf_get_num_occupied_slots(qf);

 	printf("Fill ratio: %f %llu %llu\n", 1.0*occupied/found_nslots, occupied, found_nslots);

	printf("Min time: %f %llu/%llu\n", 1.0*min[0]/CYCLES_PER_SECOND, min[0], CYCLES_PER_SECOND);
	
	printf("Max time: %f %llu/%llu\n", 1.0*max[0]/CYCLES_PER_SECOND, max[0], CYCLES_PER_SECOND);

	printf("Average time: %f %llu/%llu\n", 1.0*total[0]/(nvals*CYCLES_PER_SECOND), total[0], nvals*CYCLES_PER_SECOND);


 	cudaFree(counter1);
 	cudaFree(counter2);

 	cudaFree(max);
 	cudaFree(min);
 	cudaFree(total);

 	cudaFree(dev_hashes);
 	cudaFree(dev_firsts);
 	cudaFree(dev_seconds);

 	qf_destroy_device(qf);


	return 0;

}
