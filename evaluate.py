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

model = tf.keras.models.load_model(args.model_path[0])
use_hack = True
if use_hack:
  for variable in model.trainable_weights:
    variable.regularizer = None


test_data = tf.keras.utils.image_dataset_from_directory(
    args.data_path[0],
    labels='inferred',
    color_mode='rgb',
    shuffle=True,
    seed=123,
    interpolation='bilinear',
    follow_links=False,
    batch_size=5,
    image_size=(256, 256)
)

print("Commencing evalutation.")

results = model.evaluate(test_data)

print("")

print("Model Stats on given data:")

print("Loss: " + str(results[0]))
print("Accuracy: " + str(results[1]))
