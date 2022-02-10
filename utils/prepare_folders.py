import subprocess
import time
from pathlib import Path
from shutil import copy, rmtree
from typing import Optional


def yes_no_prompt(question: str, default: bool = True) -> bool:
    """Prompts the user for a binary answer.

    Args:
        question (str): The text that will be shown to the user.
        default (bool): Default value to use if the user just presses enter.

    Returns:
        A boolean (True for yes, False for no)
    """
    choices = " [Y/n]: " if default else " [y/N]: "

    answer = input(question + choices).lower().strip()

    if len(answer) == 0:
        return default

    if answer not in ['y', 'n']:
        print("Input invalid, please enter 'y' or 'n'")
        return yes_no_prompt(question)
    return answer == 'y'


def prepare_folders(tb_dir: Optional[Path] = None,
                    checkpoints_dir: Optional[Path] = None,
                    repo_name: str = "train_code",
                    extra_files: Optional[list[Path]] = None):
    """Prepare TensorBoard and checkpoints folders.

    For the given paths, if the folder already exists, then promts the user on what to do.
    If the checkpoints_dir path is given, then the project's files are copied in that folder to facilitate future use.

    Args:
        tb_dir (Path, optional): Path to where the TensorBoard folder should be created.
        checkpoints_dir (Path, optional): Path to where the checkpoints folder should be created.
        repo_name (str): Name of the project (git repo name), used when checkpoints_dir is given.
        extra_files (list[Path], optional): List of files that are in the .gitignore but that should still be copied.
    """
    # If path not None -> promt to remove if exist
    if tb_dir is not None:
        if tb_dir.exists():
            delete = yes_no_prompt(f"TensorBoard folder \"{tb_dir}\" already exists, do you want to delete it ?")
            if delete:
                while tb_dir.exists():
                    rmtree(tb_dir, ignore_errors=True)
                    time.sleep(0.5)
            else:
                print(f"TensorBoard dir {tb_dir} will not be overwritten, exiting.")
                exit()
        tb_dir.mkdir(parents=True, exist_ok=False)

    if checkpoints_dir is not None:
        if checkpoints_dir.exists():
            delete = yes_no_prompt(f"Checkpoints folder \"{checkpoints_dir}\" already exists,"
                                   " do you want to delete it ?")
            if delete:
                while checkpoints_dir.exists():
                    rmtree(checkpoints_dir, ignore_errors=True)
                    time.sleep(0.5)
            else:
                print(f"Checkpoints dir {checkpoints_dir} will not be overwritten, exiting.")
                exit()

        # Makes a copy of all the code (and config) so that the checkpoints are easy to load and use
        # Note: Using git instead of pure python for simplicity.
        files_to_copy = list([Path(p.decode("utf-8")) for p in
                              subprocess.check_output("git ls-files --recurse-submodules", shell=True).splitlines()])
        files_to_copy.extend(extra_files)  # Files that are in the .gitignore
        output_folder = checkpoints_dir / repo_name
        for file_path in files_to_copy:
            destination_path = output_folder / file_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            copy(file_path, destination_path)
