from argparse import ArgumentParser
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Final
)
from itertools import product
import functools
from time import time
import signal

import numpy as np
from torch import Tensor


class BatchGenerator:
    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int,
                 nb_workers: int = 1,
                 data_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None,
                 labels_preprocessing_fn: Optional[Callable[[Path], np.ndarray]] = None,
                 cpu_pipeline: Optional[Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = None,
                 gpu_pipeline: Optional[Callable[[np.ndarray, np.ndarray], tuple[Tensor, Tensor]]] = None,
                 shuffle: bool = False, verbose: bool = False):
        """
        Args:
            data: Numpy array with the data. It can be be only path to the datapoints to load (or other forms of data)
                  if the loading function is given as data_preprocessing_fn.
            labels: Numpy array with the labels, as for data it can be not yet fully ready labels.
            nb_workers: Number of workers to use for multiprocessing (>=1).
            data_preprocessing_fn: If not None, data is expected to be paths that will be passed through this function.
            labels_preprocessing_fn: If not None, labels is should be paths that will be passed through this function.
            cpu_pipeline: Function that takes in data and labels and do some data augmentation on them (on cpu, numpy)
            gpu_pipeline: Function that takes in data and labels and do some data augmentation on them.
                                       Should start by transforming numpy arrays into torch Tensors.
        shuffle: If True, then dataset is shuffled for each epoch
            verbose: If true then the BatchGenerator will print debug information
        """
        self.verbose: Final[bool] = verbose
        # Handles ctrl+c to have a clean exit.
        self.init_signal_handling(KeyboardInterrupt, signal.SIGINT, self.signal_handler)

        self.data: Final[np.ndarray] = data
        self.labels: Final[np.ndarray] = labels

        self.batch_size: Final[int] = batch_size
        self.shuffle: Final[bool] = shuffle
        self.nb_workers: Final[int] = nb_workers
        self.data_preprocessing_fn = data_preprocessing_fn
        self.labels_preprocessing_fn = labels_preprocessing_fn
        self.cpu_pipeline = cpu_pipeline
        self.gpu_pipeline = gpu_pipeline

        # TODOLIST
        # TODO: Add possibility to save dataset as hdf5
        # TODO: Add possibility to drop last batch
        # TODO: Have the prefetch in a worker

        self.nb_datapoints: Final[int] = len(self.data)

        index_list: np.ndarray = np.arange(self.nb_datapoints)
        if self.shuffle:
            np.random.shuffle(index_list)

        # Prepare a batch of data to know its size and shape
        data_batch: np.ndarray = np.asarray([data_preprocessing_fn(entry) if data_preprocessing_fn else entry
                                             for entry in data[:batch_size]])
        labels_batch: np.ndarray = np.asarray([labels_preprocessing_fn(entry) if labels_preprocessing_fn else entry
                                               for entry in labels[:batch_size]])
        if self.cpu_pipeline:
            data_batch, labels_batch = self.cpu_pipeline(data_batch, labels_batch)
        if self.gpu_pipeline:
            gpu_data_batch, gpu_labels_batch = self.gpu_pipeline(data_batch, labels_batch)

        # The shapes are not used in the BatchGenerator, but they can be accessed by other functions
        self.data_shape = gpu_data_batch.shape[1:] if self.gpu_pipeline else data_batch.shape[1:]
        self.label_shape = gpu_labels_batch.shape[1:] if self.gpu_pipeline else labels_batch.shape[1:]

        self.step_per_epoch = (self.nb_datapoints + (batch_size-1)) // self.batch_size
        self.last_batch_size = self.nb_datapoints % self.batch_size
        if self.last_batch_size == 0:
            self.last_batch_size = self.batch_size
        self.current_batch_size = self.batch_size if self.step_per_epoch > 1 else self.last_batch_size

        self.epoch = 0
        self.global_step = 0
        self.step = 0

        # Create shared memories for indices, data and labels.
        self.memories_released = mp.Event()   # TODO: change that to a boolean
        # For data and labels, 2 memories / caches are required for prefetch to work.
        # (One for the main process to read from, one for the workers to write in)
        self._current_cache = 0
        # Indices
        self._cache_memory_indices = shared_memory.SharedMemory(create=True, size=index_list.nbytes)
        self._cache_indices = np.ndarray(self.nb_datapoints, dtype=int, buffer=self._cache_memory_indices.buf)
        self._cache_indices[:] = index_list
        # Data
        self._cache_memory_data = [
            shared_memory.SharedMemory(create=True, size=data_batch.nbytes),
            shared_memory.SharedMemory(create=True, size=data_batch.nbytes)]
        self._cache_data = [
            np.ndarray(data_batch.shape, dtype=data_batch.dtype, buffer=self._cache_memory_data[0].buf),
            np.ndarray(data_batch.shape, dtype=data_batch.dtype, buffer=self._cache_memory_data[1].buf)]
        # Labels
        self._cache_memory_labels = [
            shared_memory.SharedMemory(create=True, size=labels_batch.nbytes),
            shared_memory.SharedMemory(create=True, size=labels_batch.nbytes)]
        self._cache_labels = [
            np.ndarray(labels_batch.shape, dtype=labels_batch.dtype, buffer=self._cache_memory_labels[0].buf),
            np.ndarray(labels_batch.shape, dtype=labels_batch.dtype, buffer=self._cache_memory_labels[1].buf)]

        # Create workers
        self._process_id = "NA"
        self._init_workers()
        self._prefetch_batch()
        self._process_id = "main"

    def _init_workers(self):
        """Create workers and pipes / events used to communicate with them"""
        self.stop_event = mp.Event()
        self.worker_pipes = [mp.Pipe() for _ in range(self.nb_workers)]
        self.worker_processes = []
        for worker_index in range(self.nb_workers):
            self.worker_processes.append(mp.Process(target=self._worker_fn, args=(worker_index,)))
            self.worker_processes[-1].start()

    def _worker_fn(self, worker_index: int):
        """ Function executed by workers, loads and process a mini-batch of data and puts it in the shared memory"""
        self._process_id = f"worker_{worker_index}"
        pipe = self.worker_pipes[worker_index][1]

        while not self.stop_event.is_set():
            try:
                # Check if there is a message to be received. (prevents process from getting stuck)
                if pipe.poll(0.05):
                    current_cache, cache_start_index, indices_start_index, nb_elts = pipe.recv()
                    # If the worker is in excess, then it has nothing to do (small last batch for exemple)
                    if nb_elts == 0:
                        pipe.send(True)
                        continue
                else:
                    continue
                if self.verbose > 2:
                    print(f"Worker {worker_index}, Starting to prepare mini-batch of {nb_elts} elements")

                indices_to_process = self._cache_indices[indices_start_index:indices_start_index+nb_elts]

                # Get the data (and process it)
                if self.data_preprocessing_fn:
                    processed_data = self.data_preprocessing_fn(self.data[indices_to_process])
                else:
                    processed_data = self.data[indices_to_process]
                if self.verbose > 2:
                    print(f"Worker {worker_index}, data processed successfully")

                # Do the same for labels
                if self.labels_preprocessing_fn:
                    processed_labels = self.labels_preprocessing_fn(self.labels[indices_to_process])
                else:
                    processed_labels = self.labels[indices_to_process]
                if self.verbose > 2:
                    print(f"Worker {worker_index}, labels processed successfully")

                # Do data augmentation if required
                if self.cpu_pipeline:
                    processed_data, processed_labels = self.cpu_pipeline(processed_data, processed_labels)
                    if self.verbose > 2:
                        print(f"Worker {worker_index}, data augmentation done successfully")

                # Put the mini-batch into the shared memory
                self._cache_data[current_cache][cache_start_index:cache_start_index+nb_elts] = processed_data
                self._cache_labels[current_cache][cache_start_index:cache_start_index+nb_elts] = processed_labels
                if self.verbose > 2:
                    print(f"Worker {worker_index}, data and labels put to cache successfully")

                # Send signal to the main process to say that everything is ready
                pipe.send(True)
            except (KeyboardInterrupt, ValueError):
                break

    def _prefetch_batch(self):
        """ Start sending intructions to workers to load the next batch while the previous one is being used """
        # Prefetch step is one step ahead of the actual one
        if self.step < self.step_per_epoch:
            step = (self.step + 1)
        else:
            step = 1
            # Here is the true beginning of the new epoch as far as data preparation is concerned, hence the shuffle
            if self.shuffle:
                np.random.shuffle(self._cache_indices)

        # Prepare arguments for workers and send them
        prefetch_batch_size = self.batch_size if step != self.step_per_epoch else self.last_batch_size
        prefetch_cache = 1 - self._current_cache
        nb_workers = min(self.nb_workers, prefetch_batch_size)  # Do not use all the workers if the batch size is small
        nb_elts_per_worker = prefetch_batch_size // nb_workers  # Minimum number of samples processed by a given worker
        remaining_elts = prefetch_batch_size % nb_workers  # The first remaining_elts workers will process 1 more sample
        for worker_idx in range(self.nb_workers):
            if worker_idx < nb_workers:
                cache_start_index = worker_idx * nb_elts_per_worker + min(worker_idx, remaining_elts)
                indices_start_index = (step-1) * self.batch_size + cache_start_index
                nb_elts = nb_elts_per_worker+1 if worker_idx < remaining_elts else nb_elts_per_worker
                self.worker_pipes[worker_idx][0].send((prefetch_cache, cache_start_index, indices_start_index, nb_elts))
            else:
                # Send empty instructions to excess workers
                self.worker_pipes[worker_idx][0].send((0, 0, 0, 0))

    def next_batch(self):
        """
        Returns a batch of data, goes to the next epoch when the previous one is finished.
        Does not raise a StopIteration, looping using this function means the loop will never stop.
        """
        # Check if the current epoch is finished. If it is then start a new one.
        if self.step >= self.step_per_epoch:
            self._next_epoch()

        self.global_step += 1
        self.step += 1   # Steps start at 1

        # Wait for every worker to have finished preparing its mini-batch
        for pipe, _ in self.worker_pipes:
            pipe.recv()

        self.current_batch_size = self.batch_size if self.step != self.step_per_epoch else self.last_batch_size
        self._current_cache = (self._current_cache+1) % 2
        data_batch = self._cache_data[self._current_cache][:self.current_batch_size]
        labels_batch = self._cache_labels[self._current_cache][:self.current_batch_size]

        # Start prefetching the next batch
        self._prefetch_batch()

        # Transform data to tensor on gpu and do some data augmentation if needed
        if self.gpu_pipeline:
            data_batch, labels_batch = self.gpu_pipeline(data_batch, labels_batch)

        return data_batch, labels_batch

    def reset_epoch(self):
        """ Go back to the first step of the current epoch. (data will be shuffled if shuffle is set to True)"""
        self.step = self.step_per_epoch - 1  # Go to the last step of the epoch
        self.next_batch()  # Take the last batch and ignore it  (to have the prefetch function called)
        self.global_step -= 1  # Do not count the extra step done in nest_batch() in the global counter
        self._next_epoch()
        self.epoch -= 1  # Since the call to _next_epoch increments the counter, substract 1

    def init_signal_handling(self, exception_class: type, signal_num: int, handler: Callable):
        handler = functools.partial(handler, exception_class)
        signal.signal(signal_num, handler)
        signal.siginterrupt(signal_num, False)

    def signal_handler(self, exception_class: type, signal_num: int, current_stack_frame):
        self.release()
        if self.stop_event.is_set():
            raise exception_class()

    def _next_epoch(self):
        """Prepares variables for the next epoch"""
        self.epoch += 1
        self.step = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.step < self.step_per_epoch:
            return self.next_batch()
        else:
            self._next_epoch()
            raise StopIteration

    def __del__(self):
        self.release()

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.release()

    def __enter__(self):
        return self

    def __len__(self):
        return self.nb_datapoints

    def release(self):
        """Terminates all workers and releases all the shared ressources"""
        # Closes acces to the shared memories
        for shared_mem in self._cache_memory_data + self._cache_memory_labels + [self._cache_memory_indices]:
            shared_mem.close()

        if self._process_id == "main":
            self.stop_event.set()   # Sends signal to stop to all the workers

            # Terminates all the workers
            end_time = time() + 5  # Maximum waiting time (for all processes)
            for pipe, worker in zip(self.worker_pipes, self.worker_processes):
                pipe[0].close()
                worker.join(timeout=max(0.0, end_time - time()))
                self.worker_pipes.remove(pipe)
                self.worker_processes.remove(worker)

            if not self.memories_released.is_set():
                # Requests for all the shared memories to be destroyed
                if self.verbose:
                    print("Releasing shared memories")
                for shared_mem in self._cache_memory_data + self._cache_memory_labels + [self._cache_memory_indices]:
                    shared_mem.unlink()
                self.memories_released.set()


if __name__ == '__main__':
    parser = ArgumentParser("BatchGenerator Test script")
    parser.add_argument("--verbose", "--v", type=int, default=0, help="Verbose level to use")
    args = parser.parse_args()

    verbose = args.verbose

    def test():
        """Function used to run tests on the BatchGenerator"""
        # Prepare mock dataset
        nb_datapoints = 18
        data = np.arange(nb_datapoints)
        labels = np.arange(nb_datapoints) / 10

        # Prepare variables to test against
        workers = [1, 2, 5]
        batch_sizes = [5, 2*nb_datapoints]
        data_preprocessing_fns = [None]
        labels_preprocessing_fns = [None]

        # Put all the variables into a list, then use itertools to get all the possible combinations
        args_lists = [workers, batch_sizes, data_preprocessing_fns, labels_preprocessing_fns]
        for args in product(*args_lists):
            nb_workers, batch_size, data_preprocessing_fn, labels_preprocessing_fn = args

            if verbose:
                print(f"\n\nStarting test with {nb_workers=}, {batch_size=}")

            # Preprocess data and labels here to do it only once
            processed_data = data_preprocessing_fn(data) if data_preprocessing_fn else data
            processed_labels = labels_preprocessing_fn(labels) if labels_preprocessing_fn else labels

            # Prepare some variables used for testing
            step_per_epoch = (nb_datapoints + (batch_size-1)) // batch_size
            last_batch_size = nb_datapoints % batch_size if nb_datapoints % batch_size else batch_size
            global_step = 0

            with BatchGenerator(data, labels, batch_size, data_preprocessing_fn=data_preprocessing_fn,
                                nb_workers=nb_workers, shuffle=True, verbose=verbose) as batch_generator:
                for epoch in range(5):
                    # Variables used to aggregate dataset
                    agg_data = []
                    agg_labels = []

                    for step, (data_batch, labels_batch) in enumerate(batch_generator, start=1):
                        global_step += 1
                        agg_data += list(data_batch)
                        agg_labels += list(labels_batch)

                        if verbose > 1:
                            print(f"{batch_generator.epoch=}, {batch_generator.step=}")
                        if verbose > 2:
                            print(f"{data_batch=}, {labels_batch=}")

                        # Check that variables are what they should be
                        assert global_step == batch_generator.global_step, (
                            f"Global step is {batch_generator.global_step} but should be {global_step}")
                        expected_epoch = (global_step-1) // step_per_epoch
                        assert batch_generator.epoch == expected_epoch, (
                            f"Epoch is {batch_generator.epoch} but should be {expected_epoch}")
                        assert step == batch_generator.step, (
                            f"Step is {batch_generator.step} but should be {step}")

                        # Check that length  of each batch is as expected
                        assert len(data_batch) == len(labels_batch), "Data and labels' shapes are different"
                        if step != step_per_epoch:
                            assert len(data_batch) == batch_size, (
                                f"Batch size is {len(data_batch)} but should be {batch_size}")
                        else:
                            assert len(data_batch) == last_batch_size, (
                                f"Batch size is {len(data_batch)} but should be {last_batch_size}")

                        # Check that labels correspond to datapoints
                        for data_point, label in zip(data_batch, labels_batch):
                            original_index = np.where(processed_data == data_point)[0][0]
                            assert processed_labels[original_index] == label, (
                                f"Expected label for {data_point} to be {processed_labels[original_index]}",
                                f"but got {label}.")

                    # Check that all elements appeared (once) during the epoch
                    assert len(agg_data) == nb_datapoints, (
                        f"{len(agg_data)} elements appeared instead of {nb_datapoints}")
                    assert set(agg_data) == set(processed_data), (
                        f"Data returned are not as expected.\nExpected:\n{processed_data}\nGot:\n{agg_data}"
                        f"\n{list(set(agg_data))}  (set version)")
                    assert set(agg_labels) == set(processed_labels), (
                        f"labels returned are not as expected.\nExpected:\n{processed_labels}\nGot:\n{agg_labels}")

    test()
