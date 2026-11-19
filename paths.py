"""Folders the Windows-side scripts read and write.

Set an environment variable to use another folder. The defaults are the folders of the machine the results in
README.md were made on. Scripts that run in the container take --data and --run instead (default /data).

  AUCTION_MODEL_DATA    training files (*.npz) and the model runs (runs\\NAME\\model.pt)
                        default E:\\bridge\\data\\auction-model
  AUCTION_MODEL_BBO     BBO files: data\\bbo_mldf_augmented.parquet, data\\<player>\\*.lin,
                        bidding\\bbo_bt_seat1.parquet
                        default E:\\bridge\\data\\bbo
  AUCTION_MODEL_BRILL   Brill events: field.jsonl, board-table\\dd.json, live\\, runs\\
                        default ..\\data\\brill-replay (beside this folder)
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("AUCTION_MODEL_DATA") or r"E:\bridge\data\auction-model"
BBO = os.environ.get("AUCTION_MODEL_BBO") or r"E:\bridge\data\bbo"
BRILL = os.environ.get("AUCTION_MODEL_BRILL") or os.path.normpath(os.path.join(HERE, "..", "data", "brill-replay"))
