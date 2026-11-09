# Pictograph SDK examples

Runnable, self-contained scripts. Each generates its own demo images (via
Pillow, a base dependency), so the only setup is an API key:

```bash
pip install pictograph
export PICTOGRAPH_API_KEY=pk_live_...      # from Settings -> API Keys in the app
python examples/quickstart.py
```

Every script is re-runnable. A second run reuses the dataset, images and exports
the first one created rather than failing on "already exists".

| Example | What it shows | Spends GPU? |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Create a dataset, upload an image, annotate it, export to COCO | No |
| [`annotations.py`](annotations.py) | Every annotation geometry type: bbox, polygon, polyline, keypoint | No |
| [`upload_directory.py`](upload_directory.py) | Bulk-upload a directory, then list what landed | No |
| [`search.py`](search.py) | Search by auto-tag, by visual similarity, and find near-duplicates | No |
| [`predict.py`](predict.py) | Run inference with one of your trained models | A little (one inference pass) |
| [`train_and_deploy.py`](train_and_deploy.py) | Train a model, then stand it up as a live endpoint | Only with `PICTOGRAPH_RUN_TRAINING=1` |
| [`async_quickstart.py`](async_quickstart.py) | The same API, async - concurrent reads with `asyncio.gather` | No |

Resources are addressed by **name** everywhere - there are no ids to look up
first. The examples create datasets named `sdk-example-*`; delete them from the
app, or with `client.datasets.delete("sdk-example-quickstart")`, when you are done.

Full documentation lives at [pictograph.io/docs](https://pictograph.io/docs).
