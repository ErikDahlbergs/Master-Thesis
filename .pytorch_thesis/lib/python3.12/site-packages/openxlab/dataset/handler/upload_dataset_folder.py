""" 
upload local folder to dataset repository
"""
from openxlab.dataset.commands.utility import ContextInfo
from openxlab.dataset.io.upload import Uploader


def upload_folder(dataset_repo: str, source_path: str, target_path=""):
    """
    Upload folder from local to remote.

    Example:
        openxlab.dataset.upload_folder(
            dataset_repo="username/dataset_repo_name",
            source_path="/path/to/local/folder",
            target_path="/raw/folder"
        )

    Parameters:
        @dataset_repo String The address of dataset repository.
        @source_path String The local path of the folder to upload.
        @target_path String The target path to upload folder.
    """
    ctx = ContextInfo()
    client = ctx.get_client().get_api()
    parsed_ds_name = dataset_repo.replace("/", ",")
    uploader = Uploader(client, parsed_ds_name)
    uploader.upload_folder(source_path, target_path)
