import functools
import multiprocessing as mp
import signal
from collections.abc import Callable
from multiprocessing import shared_memory
from pathlib import Path
from time import time
from types import TracebackType
from typing import Final, Self

import numpy as np
import numpy.typing as npt
import torch

T_np_img = np.float64 | np.float16 | np.uint8
T_np_labels = np.float64 | np.int64


class BatchGenerator:
    def __init__(
        self: Self,
        data: npt.NDArray[np.object_ | T_np_img],
        labels: npt.NDArray[T_np_labels],
        batch_size: int,
        nb_workers: int = 1,
        data_preprocessing_fn: Callable[[Path], npt.NDArray[T_np_img]] | None = None,
        labels_preprocessing_fn: Callable[[Path], npt.NDArray[T_np_labels]] | None = None,
        cpu_pipeline: Callable[[npt.NDArray[T_np_img], npt.NDArray[T_np_labels]], tuple[npt.NDArray[T_np_img], npt.NDArray[T_np_labels]]] | None = None,
        gpu_pipeline: Callable[[npt.NDArray[T_np_img], npt.NDArray[T_np_labels]], tuple[torch.Tensor, torch.Tensor]] | None = None,
        shuffle: bool = False,
        seed: int = 0,
        verbose_lvl: int = 0,
    ) -> None:
        """Initialize the batch generator.

        Args:
            data: Numpy array with the data. It can be be paths to the datapoints to load
                  (or any other form of data) if the loading function is given as data_preprocessing_fn.
            labels: Numpy array with the labels, as for data the labels can be only partially processed
            batch_size: The desired batch size.
            nb_workers: Number of workers to use for multiprocessing (>=1).
            data_preprocessing_fn: If not None, data will be passed through this function.
            labels_preprocessing_fn: If not None, labels will be passed through this function.
            cpu_pipeline: Function that takes in data and labels and processes them on cpu
            gpu_pipeline: Function that takes in data and labels and processes them on gpu
                                               Should start by transforming numpy arrays into torch Tensors.
            shuffle: If True, then dataset is shuffled for each epoch.
            seed: Seed to use if shuffling is enabled.
            verbose_lvl: Verbose level, the higher the number, the more debug information will be printed
        """
        # Handles ctrl+c to have a clean exit.
        # self.init_signal_handling(KeyboardInterrupt, signal.SIGINT, self.signal_handler)

        self.data: Final[npt.NDArray[np.generic]] = data
        self.labels: Final[npt.NDArray[np.generic]] = labels
        self.batch_size: Final[int] = batch_size
        self.nb_workers: Final[int] = nb_workers
        self.data_preprocessing_fn = data_preprocessing_fn
        self.labels_preprocessing_fn = labels_preprocessing_fn
        self.cpu_pipeline = cpu_pipeline
        self.gpu_pipeline = gpu_pipeline
        self.shuffle: Final[bool] = shuffle
        self.seed = seed
        self.verbose_lvl: Final[int] = verbose_lvl

        # TODOLIST
        # TODO: Add possibility to drop last batch
        # TODO: Have the prefetch in a worker

        self.nb_datapoints: Final[int] = len(self.data)
        self.epoch = 0
        self.global_step = 0
        self.step = 0

        index_list = np.arange(self.nb_datapoints)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self.epoch)  # Make the shuffle deterministic
            rng.shuffle(index_list)

        # Prepare a batch of data to know its size and shape
        if self.verbose_lvl >= 1:
            print("Preparing the first batch of data.")
        data_batch = np.asarray([data_preprocessing_fn(entry) if data_preprocessing_fn else entry
                                 for entry in data[:batch_size]])
        labels_batch = np.asarray([labels_preprocessing_fn(entry) if labels_preprocessing_fn else entry
                                   for entry in labels[:batch_size]])
        gpu_data_batch: torch.Tensor | None = None
        gpu_labels_batch: torch.Tensor | None = None
        if self.cpu_pipeline:
            data_batch, labels_batch = self.cpu_pipeline(data_batch, labels_batch)
        if self.gpu_pipeline:
            gpu_data_batch, gpu_labels_batch = self.gpu_pipeline(data_batch, labels_batch)

        # The shapes are not used in the BatchGenerator, but they can be accessed by other functions
        self.data_shape = gpu_data_batch.shape[1:] if gpu_data_batch is not None else data_batch.shape[1:]
        self.label_shape = gpu_labels_batch.shape[1:] if gpu_labels_batch is not None else labels_batch.shape[1:]

        self.steps_per_epoch = (self.nb_datapoints + (batch_size-1)) // self.batch_size
        self.last_batch_size = self.nb_datapoints % self.batch_size
        if self.last_batch_size == 0:
            self.last_batch_size = self.batch_size
        self.current_batch_size = self.batch_size if self.steps_per_epoch > 1 else self.last_batch_size

        if self.verbose_lvl >= 1:
            print("Creating the shared memories and mp related objects.")
        # Create shared memories for indices, data and labels.
        self.memories_released = mp.Event()   # TODO: change that to a boolean ?
        # For data and labels, 2 memories / caches are required for prefetch to work.
        # (One for the main process to read from, one for the workers to write in)
        self._current_cache = 0
        # Indices
        self._cache_memory_indices = shared_memory.SharedMemory(create=True, size=index_list.nbytes)
        self._cache_indices: npt.NDArray[np.int64] = np.ndarray(self.nb_datapoints, dtype=int,
                                                                buffer=self._cache_memory_indices.buf)
        self._cache_indices[:] = index_list
        # Data
        self._cache_memory_data = [
            shared_memory.SharedMemory(create=True, size=data_batch.nbytes),
            shared_memory.SharedMemory(create=True, size=data_batch.nbytes)]
        self._cache_data = [np.ndarray(data_batch.shape, dtype=data_batch.dtype, buffer=self._cache_memory_data[0].buf),
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

    def _init_workers(self: Self):
        """Create workers and pipes / events used to communicate with them."""
        self.stop_event = mp.Event()
        self.worker_pipes = [mp.Pipe() for _ in range(self.nb_workers)]
        self.worker_processes: list[mp.Process] = []
        for worker_index in range(self.nb_workers):
            self.worker_processes.append(mp.Process(target=self._worker_fn, args=(worker_index,)))
            self.worker_processes[-1].start()

    def _worker_fn(self: Self, worker_index: int):
        """Function executed by workers, loads and process a mini-batch of data and puts it in the shared memory."""
        self._process_id = f"worker_{worker_index}"
        pipe = self.worker_pipes[worker_index][1]

        # Reinitialize numpy's random because data augmentation should be different between workers.
        # (If this is not done, the same data augmentation might be applied to all the elements of a batch for exemple)
        # Could / Should use:
        # rng: np.random._generator.Generator = np.random.default_rng()
        # But then it would need to be passed to all the functions that need it
        np.random.seed(worker_index)

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
                if self.verbose_lvl > 2:
                    print(f"Worker {worker_index}, Starting to prepare mini-batch of {nb_elts} elements")

                indices_to_process = self._cache_indices[indices_start_index:indices_start_index+nb_elts]

                # Get the data (and process it)
                if self.data_preprocessing_fn:
                    processed_data = self.data_preprocessing_fn(self.data[indices_to_process])
                else:
                    processed_data = self.data[indices_to_process]
                if self.verbose_lvl > 2:
                    print(f"Worker {worker_index}, data processed successfully")

                # Do the same for labels
                if self.labels_preprocessing_fn:
                    processed_labels = self.labels_preprocessing_fn(self.labels[indices_to_process])
                else:
                    processed_labels = self.labels[indices_to_process]
                if self.verbose_lvl > 2:
                    print(f"Worker {worker_index}, labels processed successfully")

                # Do data augmentation if required
                if self.cpu_pipeline:
                    processed_data, processed_labels = self.cpu_pipeline(processed_data, processed_labels)
                    if self.verbose_lvl > 2:
                        print(f"Worker {worker_index}, data augmentation done successfully")

                # Put the mini-batch into the shared memory
                self._cache_data[current_cache][cache_start_index:cache_start_index+nb_elts] = processed_data
                self._cache_labels[current_cache][cache_start_index:cache_start_index+nb_elts] = processed_labels
                if self.verbose_lvl > 2:
                    print(f"Worker {worker_index}, data and labels put to cache successfully")

                # Send signal to the main process to say that everything is ready
                pipe.send(True)
            except (KeyboardInterrupt):  # , ValueError):
                break

    def _prefetch_batch(self: Self):
        """Start sending intructions to workers to load the next batch while the previous one is being used."""
        # Prefetch step is one step ahead of the actual one
        if self.step < self.steps_per_epoch:
            step = (self.step + 1)
        else:
            step = 1
            # Here is the true beginning of the new epoch as far as data preparation is concerned, hence the shuffle
            if self.shuffle:
                rng = np.random.default_rng(self.seed + self.epoch)
                rng.shuffle(self._cache_indices)

        # Prepare arguments for workers and send them
        prefetch_batch_size = self.batch_size if step != self.steps_per_epoch else self.last_batch_size
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

    def next_batch(self: Self) -> tuple[npt.NDArray[T_np_img] | torch.Tensor, npt.NDArray[T_np_labels] | torch.Tensor]:
        """Return the next bach of data.

        Returns a batch of data, goes to the next epoch when the previous one is finished.
        Does not raise a StopIteration, looping using this function means the loop will never stop.
        """
        # Check if the current epoch is finished. If it is then start a new one.
        if self.step >= self.steps_per_epoch:
            self._next_epoch()

        self.global_step += 1
        self.step += 1   # Steps start at 1

        # Wait for every worker to have finished preparing its mini-batch
        for pipe, _ in self.worker_pipes:
            pipe.recv()

        self.current_batch_size = self.batch_size if self.step != self.steps_per_epoch else self.last_batch_size
        self._current_cache = (self._current_cache+1) % 2
        data_batch: npt.NDArray[T_np_img] = self._cache_data[self._current_cache][:self.current_batch_size]
        labels_batch: npt.NDArray[T_np_labels] = self._cache_labels[self._current_cache][:self.current_batch_size]

        # Start prefetching the next batch
        self._prefetch_batch()

        # Transform data to tensor on gpu and do some data augmentation if needed
        if self.gpu_pipeline:
            return self.gpu_pipeline(data_batch, labels_batch)

        return data_batch, labels_batch

    def reset_epoch(self: Self) -> None:
        """Go back to the first step of the current epoch. (data will be shuffled if shuffle is set to True)."""
        self.step = self.steps_per_epoch - 1  # Go to the last step of the epoch
        self.next_batch()  # Take the last batch and ignore it  (to have the prefetch function called)
        self.global_step -= 1  # Do not count the extra step done in nest_batch() in the global counter
        self._next_epoch()
        self.epoch -= 1  # Since the call to _next_epoch increments the counter, substract 1

    @staticmethod
    def init_signal_handling(
        exception_class: type[Exception],
        signal_num: int,
        handler: Callable[[type[Exception], int, object], None],
    ) -> None:
        handler_except = functools.partial(handler, exception_class)
        signal.signal(signal_num, handler_except)
        signal.siginterrupt(signal_num, False)

    def signal_handler(self: Self, exception_class: type[Exception], _signal_num: int, _current_stack_frame: object):
        self.release()
        if self.stop_event.is_set():
            raise exception_class

    def _next_epoch(self: Self) -> None:
        """Prepare variables for the next epoch."""
        self.epoch += 1
        self.step = 0

    def __iter__(self: Self) -> Self:
        return self

    def __next__(self: Self) -> None:
        if self.step < self.steps_per_epoch:
            return self.next_batch()
        self._next_epoch()
        raise StopIteration

    def __del__(self: Self) -> None:
        self.release()

    def __exit__(
        self: Self,
        _exc_type: type[BaseException] | None,
        _exc_value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.release()

    def __enter__(self: Self) -> Self:
        return self

    def __len__(self: Self) -> int:
        return self.nb_datapoints

    def release(self: Self) -> None:
        """Terminate all workers and release all the shared resources."""
        # Terminate cleanly even if there was an error during the initialization
        if not hasattr(self, "_cache_memory_data"):
            return

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
                if self.verbose_lvl:
                    print("Releasing shared memories")
                for shared_mem in self._cache_memory_data + self._cache_memory_labels + [self._cache_memory_indices]:
                    shared_mem.unlink()
                self.memories_released.set()
