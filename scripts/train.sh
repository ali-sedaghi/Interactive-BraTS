python ./scripts/train.py ./models/brats/segformerB3.py\
  --exp-name=Test\
  --epochs=2\
  --batch-size=2\
  --ngpus=0\
  --cpu\
  --workers=2\
  --datapath=./data/datasets/BraTS\
  --pretrained
