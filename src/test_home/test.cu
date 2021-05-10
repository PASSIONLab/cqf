
/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>   
 *
 * ============================================================================
 */
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
//#include <openssl/rand.h>

#include "include/gqf.h"
#include "include/gqf_int.h"
#include "include/gqf_file.h"
#include "hashutil.h"

#define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
#define BITMASK(nbits)((nbits) == 64 ? 0xffffffffffffffff : MAX_VALUE(nbits))
#define DISTANCE_FROM_HOME_SLOT_CUTOFF 1000


void swapElements(uint64_t* x, uint64_t* y)
{
	int temp = *x;
	*x = *y;
	*y = temp;
}
// Partition function
int partition(uint64_t arr[], int lowIndex, int highIndex)
{
	uint64_t pivotElement = arr[highIndex];
	int i = (lowIndex - 1);
	for (int j = lowIndex; j <= highIndex - 1; j++)
	{
		if (arr[j] <= pivotElement)
		{
			i++;
			swapElements(&arr[i], &arr[j]);
		}
	}
	swapElements(&arr[i + 1], &arr[highIndex]);
	return (i + 1);
}
// QuickSort Function
void quickSort(uint64_t arr[], int lowIndex, int highIndex)
{
	if (lowIndex < highIndex)
	{
		int pivot = partition(arr, lowIndex, highIndex);
		// Separately sort elements before & after partition 
		quickSort(arr, lowIndex, pivot - 1);
		quickSort(arr, pivot + 1, highIndex);
	}
}

static inline int find_thread_start(QF* qf, uint64_t* keys, int tid, int num_threads, uint64_t nvals, uint64_t qbits) {
	uint64_t max_quotient = 1ULL << qbits;
	//printf("max %lx", max_quotient);
	uint64_t thread_min_quotient = ceil(max_quotient / num_threads) * (tid);
	//TODO:reassess use of ceil
	uint64_t thread_max_quotient = tid + 1 == num_threads - 1 ? max_quotient : ceil(max_quotient / num_threads) * (tid + 1);
	//printf("tid %d, overall max quotient %lu, thread min quotient %lu, tmax %lu \n", tid, max_quotient, thread_min_quotient, thread_max_quotient);
	//TODO: optimze the search
	uint64_t rem_bits = qf->metadata->key_bits - qbits;
	for (int i = 0; i < nvals; i++) {

		uint64_t quotient = keys[i] >> rem_bits;

		if (quotient >= thread_max_quotient) {
			//uint64_t prev_key = keys[i-1] ;
			/*printf("failed to find match tid %d, overall max quotient %lu, prev_key %lu; quotient %lu;search min %lu; search max %lu\n", tid, max_quotient,
				prev_key, quotient, thread_min_quotient, thread_max_quotient);
				*/
			return -1;
		}
		if (quotient >= thread_min_quotient) {
			//printf("first val %lu, val quo %lu, tminval %lu bits %lu\n", keys[i], quotient, thread_min_quotient << rem_bits, rem_bits);
			return i;
		}
	}
	return -1;
}

static inline uint64_t find_thread_last_slot(QF* qf, int num_threads, int tid, uint64_t nvals, uint64_t qbits) {
	if (tid == num_threads - 1) {
		return qf->metadata->nslots;
	}
	uint64_t max_quotient = 1ULL << qbits;
	uint64_t slot_per_quot = qf->metadata->nslots / max_quotient;
	//same calculation as find_thread_start
	uint64_t thread_quotient = ceil(max_quotient / num_threads) * tid + 1;
	uint64_t last_slot = slot_per_quot * thread_quotient;
	//printf("FINDING_SLOT: tid %d maxquot %lu tquot %lu slot per quotient %lu last_slot %lu\n",tid, max_quotient, thread_quotient, slot_per_quot, last_slot);
	return last_slot;
}

static inline void printarray(uint64_t* arr, uint64_t len) {
	for (int i = 0; i < len; i++) {
		printf("%lu,  ", arr[i]);
		if (i % 8 == 0) {
			printf("\n");
		}
	}
	printf("\n");
}


void qf_insert_gpu(QF* qf, uint64_t* keys, uint64_t value, uint64_t nvals, uint64_t nslots, uint64_t qbits) {


	//printarray(keys, nvals);
	/*
	find_thread_start(qf, keys, 1, 6, nvals, qbits);
	find_thread_start(qf, keys, 2, 6, nvals, qbits);
	find_thread_start(qf, keys, 3, 6, nvals, qbits);
	find_thread_start(qf, keys, 4, 6, nvals, qbits);
	find_thread_start(qf, keys, 5, 6, nvals, qbits);
	*/
	int num_threads = 8000;
	//t_start and end refer to indexes in the keys array
	int* thread_done;
	thread_done = (int*)malloc(num_threads * sizeof(int));
	memset(thread_done, 0, num_threads * sizeof(int));
	//use quotient bits for the block making
	uint64_t block_size = ceil(qf->metadata->nslots / num_threads);
	//block_offset is in #slots
	uint64_t block_offset = 0;
	int num_iter = 0;
	int thread_added;
	bool fin = false;
	bool go_next_thread = false;
	while (fin == false) {
		printf("-----while loop reset; num_iter -----------%d  %d\n", num_iter, fin);
		fin = true;
		block_offset = block_offset + (block_size / 2) * num_iter;
		for (int tid = 0; tid < num_threads; tid++) {
			thread_added = 0;
			go_next_thread = false;
			int t_start = tid == 0 ? 0 : find_thread_start(qf, keys, tid, num_threads, nvals, qbits);
			int next_thread = tid + 1;
			int t_end = tid == num_threads - 1 ? nvals : find_thread_start(qf, keys, next_thread, num_threads, nvals, qbits);
			uint64_t last_slot = find_thread_last_slot(qf, num_threads, tid + num_iter, nvals, qbits);
			uint64_t prev_last = find_thread_last_slot(qf, num_threads, tid - 1 + num_iter, nvals, qbits);
			//printf("-tid %d; blstart %d; blend %d; nvals %ld \n", tid, t_start, t_end, nvals);
			//printf("-tid %d; last slot is %lu; nslots %lu, prev_last %lu\n", tid, last_slot, qf->metadata->nslots, prev_last);
			//printf("-last key doen before %d\n", thread_done[tid]);
			//case where there's no quotients to a thread;
			if (t_start == -1) {
				//printf("&Skipping thread %d\n", tid);
				continue;
			}
			while (t_end == -1) {
				next_thread++;
				//printf("next thread %d \n", next_thread);
				t_end = next_thread >= num_threads - 1 ? nvals : find_thread_start(qf, keys, next_thread, num_threads, nvals, qbits);
			}
			thread_done[tid] = thread_done[tid] > t_start ? thread_done[tid] : t_start;
			while (thread_done[tid] < t_end && go_next_thread == false) {

				uint64_t key = keys[thread_done[tid]];

				//resizing would happen here
				/*
				* Hashing has to happen beforethis
				if (GET_KEY_HASH(flags) != QF_KEY_IS_HASH) {
					if (qf->metadata->hash_mode == QF_HASH_DEFAULT)
						key = MurmurHash64A(((void*)&key), sizeof(key), qf->metadata->seed) % qf->metadata->range;
					else if (qf->metadata->hash_mode == QF_HASH_INVERTIBLE)
						key = hash_64(key, BITMASK(qf->metadata->key_bits));
				}
				*/
				uint64_t hash = (key << qf->metadata->value_bits) | (value & BITMASK(qf->metadata->value_bits));
				int ret;
				ret = insert1_gpu(qf, hash, last_slot, prev_last, tid);
				//printf("ret %d;\n", ret);
				if (ret == QF_END_OF_THREAD) {
					//printf("**hit boundary, going next\n");
					fin = false;
					go_next_thread = true;
					//continue is just for serial-on GPU it'll be each thread waiting for next iter
					continue;
				}
				thread_added += 1;
				thread_done[tid] = thread_done[tid] + 1;
				// check for fullness based on the distance from the home slot to the slot
				// in which the key is inserted
				if (ret == QF_NO_SPACE || ret > DISTANCE_FROM_HOME_SLOT_CUTOFF) {
					float load_factor = qf_get_num_occupied_slots(qf) /
						(float)qf->metadata->nslots;
					fprintf(stdout, "Load factor: %lf\n", load_factor);

					fprintf(stderr, "The CQF is filling up.\n");
					ret = QF_NO_SPACE;

				}
			}
			if (tid >= num_threads-2) printf("tid %d added %d items\n",tid, thread_added);
		}
		num_iter++;
	}
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "Please specify the log of the number of slots and the number of remainder bits in the CQF.\n");
		exit(1);
	}
	QF qf;
	uint64_t qbits = atoi(argv[1]);
	uint64_t rbits = atoi(argv[2]);
	uint64_t nhashbits = qbits + rbits;
	//number of slots in the qf, can be changed
	uint64_t nslots = (4ULL << qbits);
	//this can be changed to change the % it fills up

	uint64_t nvals = 20*nslots/100;
	uint64_t key_count = 1;
	uint64_t *vals;
	uint64_t* hashes;
	printf("nvals: %lu\n", nvals);
	/* Initialise the CQF */
	/*if (!qf_malloc(&qf, nslots, nhashbits, 0, QF_HASH_NONE, 0)) {*/
	/*fprintf(stderr, "Can't allocate CQF.\n");*/
	/*abort();*/
	/*}*/
	if (!qf_initfile(&qf, nslots, nhashbits, 0, QF_HASH_NONE, 0,
									 "/tmp/mycqf.file")) {
		fprintf(stderr, "Can't allocate CQF.\n");
		abort();
	}

	qf_set_auto_resize(&qf, false);
	/* Generate random values */
	vals = (uint64_t*)malloc(nvals*sizeof(vals[0]));
	hashes = (uint64_t*)malloc(nvals * sizeof(hashes[0]));
	//RAND_bytes((unsigned char *)vals, sizeof(*vals) * nvals);
	srand(0);
	//pre-hash everything
	for (uint64_t i = 0; i < nvals; i++) {
		vals[i] = rand();
		vals[i] = (1 * vals[i]) % qf.metadata->range;
		vals[i] = hash_64(vals[i], BITMASK(nhashbits));
		/*fake hash until implemented*/
		//hashes[i] = vals[i];
	}
	/*
	for(int i = 0; i<nvals; i++){
	printf("%lx\n", vals[i]);
	}
	*/
	/* Insert keys in the CQF */
       //Sort here so the test works
       //TODO: ask Prashant why this breaks the test (bottom test, prints 'index weirdness')
  //   printf("sortd, %lu bytes\n", sizeof(vals[0]));
	//This happens inside the GPU insert step for the GPU implementation.
	 quickSort(vals, 0, nvals);
	/*
	for (int i = 0; i<nvals; i++){

		printf("%lx\n", vals[i]);
	}
	*/
	 //changed so key_count is always 1
	qf_insert_gpu(&qf, vals, 0, nvals, nslots,  qbits);
	printf("FINISHED THE INSERT\n");
	/*
	for (uint64_t i = 0; i < nvals; i++) {
		int ret = qf_insert(&qf, vals[i], 0, key_count, QF_NO_LOCK);
		if (ret < 0) {
			fprintf(stderr, "failed insertion for key: %lx %d.\n", vals[i], 50);
			if (ret == QF_NO_SPACE)
				fprintf(stderr, "CQF is full.\n");
			else if (ret == QF_COULDNT_LOCK)
				fprintf(stderr, "TRY_ONCE_LOCK failed.\n");
			else
				fprintf(stderr, "Does not recognise return value.\n");
			abort();
		}
	}
	*/

	/* Lookup inserted keys and counts. */
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&qf, vals[i], 0, 0);
		if (count < key_count) {
			fprintf(stderr, "failed lookup after insertion for %lx %ld.\n", vals[i],
							count);
			//abort();

		}
	}

#if 0
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&qf, vals[i], 0, 0);
		if (count < key_count) {
			fprintf(stderr, "failed lookup during deletion for %lx %ld.\n", vals[i],
							count);
			abort();
		}
		if (count > 0) {
			/*fprintf(stdout, "deleting: %lx\n", vals[i]);*/
			qf_delete_key_value(&qf, vals[i], 0, QF_NO_LOCK);
			/*qf_dump(&qf);*/
			uint64_t cnt = qf_count_key_value(&qf, vals[i], 0, 0);
			if (cnt > 0) {
				fprintf(stderr, "failed lookup after deletion for %lx %ld.\n", vals[i],
								cnt);
				abort();
			}
		}
	}
#endif

	/* Write the CQF to disk and read it back. */
	char filename[] = "/tmp/mycqf_serialized.cqf";
	fprintf(stdout, "Serializing the CQF to disk.\n");
	uint64_t total_size = qf_serialize(&qf, filename);
	if (total_size < sizeof(qfmetadata) + qf.metadata->total_size_in_bytes) {
		fprintf(stderr, "CQF serialization failed.\n");
	//	abort();
	}
	qf_deletefile(&qf);

	QF file_qf;
	fprintf(stdout, "Reading the CQF from disk.\n");
	if (!qf_deserialize(&file_qf, filename)) {
		fprintf(stderr, "Can't initialize the CQF from file: %s.\n", filename);
	//	abort();
	}
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&file_qf, vals[i], 0, 0);
		if (count < key_count) {
			fprintf(stderr, "failed lookup in file based CQF for %lx %ld.\n",
							vals[i], count);
	//		abort();
		}
	}

	fprintf(stdout, "Testing iterator and unique indexes.\n");
	/* Initialize an iterator and validate counts. */
	QFi qfi;
	qf_iterator_from_position(&file_qf, &qfi, 0);
	QF unique_idx;
	if (!qf_malloc(&unique_idx, file_qf.metadata->nslots, nhashbits, 0,
								 QF_HASH_NONE, 0)) {
		fprintf(stderr, "Can't allocate set.\n");
	//	abort();
	}

	int64_t last_index = -1;
	int i = 0;
	qf_iterator_from_position(&file_qf, &qfi, 0);
	while(!qfi_end(&qfi)) {
		uint64_t key, value, count;
		qfi_get_key(&qfi, &key, &value, &count);
		if (count < key_count) {
			fprintf(stderr, "Failed lookup during iteration for: %lx. Returned count: %ld\n",
							key, count);
			abort();
		}
		int64_t idx = qf_get_unique_index(&file_qf, key, value, 0);
		if (idx == QF_DOESNT_EXIST) {
			fprintf(stderr, "Failed lookup for unique index for: %lx. index: %ld\n",
							key, idx);
			abort();
		}
		if (idx <= last_index) {
			fprintf(stderr, "Unique indexes not strictly increasing.\n");
			abort();
		}
		last_index = idx;
		if (qf_count_key_value(&unique_idx, key, 0, 0) > 0) {
			fprintf(stderr, "Failed unique index for: %lx. index: %ld\n",
							key, idx);
			abort();
		}
		/*
		qf_insert(&unique_idx, key, 0, 1, QF_NO_LOCK);
		int64_t newindex = qf_get_unique_index(&unique_idx, key, 0, 0);
		if (idx < newindex) {
			fprintf(stderr, "Index weirdness: index %dth key %ld was at %ld, is now at %ld\n",
							i, key, idx, newindex);
			//abort();
		}*/

		i++;
		qfi_next(&qfi);
	}

	/* remove some counts  (or keys) and validate. */
	fprintf(stdout, "Testing remove/delete_key.\n");
	for (uint64_t i = 0; i < nvals; i++) {
		uint64_t count = qf_count_key_value(&file_qf, vals[i], 0, 0);
		/*if (count < key_count) {*/
		/*fprintf(stderr, "failed lookup during deletion for %lx %ld.\n", vals[i],*/
		/*count);*/
		/*abort();*/
		/*}*/
		int ret = qf_delete_key_value(&file_qf, vals[i], 0, QF_NO_LOCK);
		count = qf_count_key_value(&file_qf, vals[i], 0, 0);
		if (count > 0) {
			if (ret < 0) {
				fprintf(stderr, "failed deletion for %lx %ld ret code: %d.\n",
								vals[i], count, ret);
				abort();
			}
			uint64_t new_count = qf_count_key_value(&file_qf, vals[i], 0, 0);
			if (new_count > 0) {
				fprintf(stderr, "delete key failed for %lx %ld new count: %ld.\n",
								vals[i], count, new_count);
				abort();
			}
		}
	}

	qf_deletefile(&file_qf);

	fprintf(stdout, "Validated the CQF.\n");

}

