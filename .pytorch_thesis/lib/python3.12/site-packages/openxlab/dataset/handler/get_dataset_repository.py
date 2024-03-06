""" 
get dataset repository totally
"""

import os

from rich import print as rprint
from tqdm import tqdm

from openxlab.dataset.commands.utility import ContextInfo
from openxlab.dataset.constants import FILE_THRESHOLD
from openxlab.dataset.io import downloader
from openxlab.types.command_type import *


def get(dataset_repo: str, target_path=""):
    """
    Get the dataset repository.

    Example:
        openxlab.dataset.get(
            dataset_repo="username/dataset_repo_name",
            target_path="/path/to/local/folder"
        )

    Parameters:
        @dataset_repo String The address of dataset repository.
        @target_path String The target local path to save the dataset repository.
    """
    if not target_path:
        target_path = os.getcwd()
    # else:
    #     destination_path = os.path.normpath(destination_path)

    ctx = ContextInfo()
    client = ctx.get_client()

    # parse dataset_name
    parsed_ds_name = dataset_repo.replace("/", ",")
    parsed_save_path = dataset_repo.replace("/", "___")

    data_dict = client.get_api().get_dataset_files(dataset_name=parsed_ds_name)
    info_dataset_id = data_dict['list'][0]['dataset_id']

    object_info_list = []
    for info in data_dict['list']:
        curr_dict = {}
        curr_dict['size'] = info['size']
        curr_dict['name'] = info['path'][1:]
        object_info_list.append(curr_dict)

    if object_info_list:
        # download check for crawler with one file
        download_check_path = object_info_list[0]['name']
        # download check
        client.get_api().download_check(dataset_id=info_dataset_id, path=download_check_path)

    with tqdm(total=len(object_info_list)) as pbar:
        for idx in range(len(object_info_list)):
            # exist already
            rprint(f" Downloading No.{idx+1} of total {len(object_info_list)} files")
            if os.path.exists(
                (os.path.join(target_path, parsed_save_path, object_info_list[idx]['name']))
            ):
                rprint(f"target already exists, jumping to next !")
                pbar.update(1)
                continue

            # big file download
            if object_info_list[idx]['size'] > FILE_THRESHOLD:
                download_url = client.get_api().get_dataset_download_urls(
                    info_dataset_id, object_info_list[idx]
                )
                downloader.BigFileDownloader(
                    url=download_url,
                    filename=object_info_list[idx]['name'],
                    download_dir=os.path.join(target_path, parsed_save_path),
                    file_size=object_info_list[idx]['size'],
                    blocks_num=8,
                ).start()
                pbar.update(1)
            # small file download
            else:
                download_url = client.get_api().get_dataset_download_urls(
                    info_dataset_id, object_info_list[idx]
                )
                downloader.SmallFileDownload(
                    url=download_url,
                    filename=object_info_list[idx]['name'],
                    download_dir=os.path.join(target_path, parsed_save_path),
                )._single_thread_download()
                pbar.update(1)
    client.get_api().track_download_dataset_files(dataset_name=parsed_ds_name, file_path="")
    rprint(f"\nDownload Completed.")
