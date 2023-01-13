import tensorflow as tf
import argparse 

parser = argparse.ArgumentParser(
                    prog = 'SpottingDiffusion',
                    description = 'Evaluate the SpottingDiffusion Model')

parser.add_argument('data_path', metavar='datapath', type=str, nargs=1,
                    help='the path to the evaluation data folder.')

parser.add_argument('model_path', metavar='modelpath', type=str, nargs=1,
                    help='the path to the SpottingDiffusion model.')

args = parser.parse_args()

model = tf.saved_model.load(args.modelpath[0])

test_data = tf.keras.utils.image_dataset_from_directory(
    args.datapath[0],
    labels='inferred',
    color_mode='rgb',
    shuffle=True,
    seed=123,
    interpolation='bilinear',
    follow_links=False,
    batch_size=batch_size,
    image_size=(256, 256)
)

results = model.evaluate(test_data)
