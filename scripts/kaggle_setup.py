#!/usr/bin/env python3
"""Helper used on Kaggle to mirror the raw GazeBaseVR archive as a private
Kaggle dataset, so later sessions can attach it read-only. Not needed for the
local pipeline; documented here for provenance.

    from kaggle_secrets import UserSecretsClient        # in a Kaggle cell
    os.environ['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')

    import json
    meta = {"title": "GazeBaseVR raw mirror",
            "id": "<kaggle-user>/gazebasevr-raw",
            "licenses": [{"name": "CC-BY-4.0"}]}
    json.dump(meta, open("/kaggle/working/gbvr/dataset-metadata.json", "w"))
    # !kaggle datasets create -p /kaggle/working/gbvr
"""
if __name__ == "__main__":
    print(__doc__)
