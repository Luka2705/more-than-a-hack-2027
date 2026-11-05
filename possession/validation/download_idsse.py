"""Download the IDSSE / DFL dataset (Bassek et al. 2025, CC BY 4.0) from figshare into
external/idsse/. About 2.6 GB.

    python -m possession.validation.download_idsse
"""

import json
import os
import urllib.request

from .loaders import EXT

ARTICLE = "https://api.figshare.com/v2/articles/28196177"


def main():
    out = os.path.join(EXT, "idsse")
    os.makedirs(out, exist_ok=True)
    meta = json.load(urllib.request.urlopen(ARTICLE))
    for f in meta["files"]:
        path = os.path.join(out, f["name"].strip())
        if os.path.isfile(path) and os.path.getsize(path) == f["size"]:
            continue
        print("downloading", f["name"], f["size"])
        urllib.request.urlretrieve(f["download_url"], path)


if __name__ == "__main__":
    main()
