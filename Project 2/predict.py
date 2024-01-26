# imports
import argparse
import matplotlib.pyplot as plt
import futils


# Take in command line arguments
parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path', help='Path to image')
parser.add_argument('checkpoint_path', default='checkpoint.pth', help='Given checkpoint of a network')
parser.add_argument('--top_k', default=5, help='Return top k most likely classes')
parser.add_argument('--category_names',default='cat_to_name.json',  help='Use a mapping of categories to real names')
parser.add_argument('--gpu', choices=['0', '1'], default='1', help='True for using GPU for training')

args = parser.parse_args()

top_k = int(args.top_k)

# Loading the model from the checkpoint
print()
print("Loading and building model from {}".format(args.checkpoint_path))
model, epochs, learning_rate = futils.load_checkpoint(args.checkpoint_path)

print(model)
print()

# Predicting the output
probs, predict_classes_idx, predict_classes = futils.predict(args.image_path, model, args.gpu, top_k, args.category_names)
print("Prediction results :")
print(probs)
print(predict_classes_idx)
print(predict_classes)

fig = plt.figure(figsize = (5,5))
ax = plt.subplot(2,1,1)
ax.set_title(predict_classes[0])
plt.axis('off')
futils.imshow(futils.process_image(args.image_path), ax, title="lol")
plt.show()