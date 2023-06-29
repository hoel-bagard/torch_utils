from itertools import product

import numpy as np
from hbtorch_utils.utils.batch_gnerator import BatchGenerator


def test_batch_generator(verbose_lvl: int) -> None:
    """Function used to run tests on the BatchGenerator."""
    # Prepare mock dataset
    nb_datapoints = 18
    data = np.arange(nb_datapoints, dtype=np.float64)
    labels = np.arange(nb_datapoints) / 10

    # Prepare variables to test against
    workers = [1, 2, 5]
    batch_sizes = [5, 2*nb_datapoints]
    data_preprocessing_fns = [None]
    labels_preprocessing_fns = [None]

    # Put all the variables into a list, then use itertools to get all the possible combinations
    args_lists = [workers, batch_sizes, data_preprocessing_fns, labels_preprocessing_fns]
    for test_args in product(*args_lists):
        nb_workers, batch_size, data_preprocessing_fn, labels_preprocessing_fn = test_args

        if verbose_lvl:
            print(f"\n\nStarting test with {nb_workers=}, {batch_size=}")

        # Preprocess data and labels here to do it only once
        processed_data = data_preprocessing_fn(data) if data_preprocessing_fn else data
        processed_labels = labels_preprocessing_fn(labels) if labels_preprocessing_fn else labels

        # Prepare some variables used for testing
        steps_per_epoch = (nb_datapoints + (batch_size-1)) // batch_size
        last_batch_size = nb_datapoints % batch_size if nb_datapoints % batch_size else batch_size
        global_step = 0

        with BatchGenerator(data, labels, batch_size, data_preprocessing_fn=data_preprocessing_fn,
                            nb_workers=nb_workers, shuffle=True, verbose_lvl=verbose_lvl) as batch_generator:
            for _epoch in range(5):
                # Variables used to aggregate dataset
                agg_data: list[int] = []
                agg_labels: list[int] = []
                for step, (data_batch, labels_batch) in enumerate(batch_generator, start=1):
                    global_step += 1
                    # For Pyright, no GPU test here.
                    assert isinstance(data_batch, np.ndarray)
                    assert isinstance(labels_batch, np.ndarray)
                    agg_data += list(data_batch)
                    agg_labels += list(labels_batch)

                    if verbose_lvl > 1:
                        print(f"{batch_generator.epoch=}, {batch_generator.step=}")
                    if verbose_lvl > 2:
                        print(f"{data_batch=}, {labels_batch=}")

                    # Check that variables are what they should be
                    assert global_step == batch_generator.global_step, (
                        f"Global step is {batch_generator.global_step} but should be {global_step}")
                    expected_epoch = (global_step-1) // steps_per_epoch
                    assert batch_generator.epoch == expected_epoch, (
                        f"Epoch is {batch_generator.epoch} but should be {expected_epoch}")
                    assert step == batch_generator.step, (
                        f"Step is {batch_generator.step} but should be {step}")

                    # Check that length  of each batch is as expected
                    assert len(data_batch) == len(labels_batch), "Data and labels' shapes are different"
                    if step != steps_per_epoch:
                        assert len(data_batch) == batch_size, (
                            f"Batch size is {len(data_batch)} but should be {batch_size}")
                    else:
                        assert len(data_batch) == last_batch_size, (
                            f"Batch size is {len(data_batch)} but should be {last_batch_size}")

                    # Check that labels correspond to datapoints
                    for data_point, label in zip(data_batch, labels_batch, strict=True):
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
