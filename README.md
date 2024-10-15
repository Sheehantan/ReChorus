已经用ReChorus1.0初步实现BSARec,具体可见"src\models\sequential\BSARec.py",使用方法如下
cd src
python main.py --model_name BSARec --emb_size 64 --num_layers 1 --num_heads 1 --lr 1e-4 --l2 1e-3 --history_max 20 --train 1 --num_workers 0 --dataset 'Grocery_and_Gourmet_Food'
