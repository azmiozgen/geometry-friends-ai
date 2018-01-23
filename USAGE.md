## Human play
python main.py -g 10 --setup 2  <!-- Game over after 10 finishes of a setup -->

## AI train
<!-- Load model with -l, save with -s, --lr learning rate, -v for verbose, --setup for different setups -->
python main.py --train -s models/<model_name>.h5 -l models/<model_name>.h5 --lr 1e-5 -g 100 -v
python main.py --train -s models/<model_name>.h5 --lr 1e-5 -g 100 -v		<!-- New model, no load -->

## AI test
python main.py --test -l models/<model_name>.h5 -g 100 --setup 2 -v
