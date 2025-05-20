# --------------------------------------------------------------------------
# Title: Midterm Project - Brain Tumor Classification with FastAI
# Author: Hakan Uca
# Lesson: ADA - 447
# ---------------------------------------------------------------------------

from fastai.vision.all import *
import matplotlib.pyplot as plt
from collections import Counter
import os

def main():
    # Step A.1: Sets the path
    path = Path(r"C:\Users\hakan\PycharmProjects\ADA-447_MIDTERM-PROJECT\Data")

    # Step A.1.1: Inspects the data layout
    print("Inspecting dataset folder structure...")
    for label_folder in os.listdir(path):
        print(f"Label '{label_folder}' has {len(os.listdir(path/label_folder))} images")

    # Step A.1.2: Visualizes label distribution
    print("Visualizing label distribution...")
    labels = [p.parent.name for p in get_image_files(path)]
    label_counts = Counter(labels)
    print("Label Counts:", label_counts)

    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("Label Distribution")
    plt.xlabel("Label")
    plt.ylabel("Number of Images")
    plt.show()

    # Step A.2: Creates the DataBlock and dataloaders with presizing and transforms
    tumor_data_block = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(460),  # Presizing: Larger size before augmentation
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
    )

    dls = tumor_data_block.dataloaders(path, bs=32, num_workers=0)

    # Step A.3.1: Shows batch of images
    print("Showing batch of images...")
    dls.show_batch(max_n=9)

    # Step A.3.2: Checks the labels
    print("Classes in dataset:", dls.vocab)

    # Step A.3.3: Shows DataBlock summary
    learn = vision_learner(dls, resnet34, metrics=accuracy)
    print("Model Summary:")
    learn.summary()

    # Step A.4.1: Trains a benchmark model
    print("Training simple benchmark model for comparison...")
    simple_learner = vision_learner(dls, resnet18, metrics=accuracy)
    simple_learner.fine_tune(1)
    simple_learner.show_results()
    interp_simple = ClassificationInterpretation.from_learner(simple_learner)
    interp_simple.plot_confusion_matrix(title="Benchmark Model Confusion Matrix")
    plt.show()

    # Step B.1: Learning Rate Finder
    print("Running learning rate finder...")
    learn.lr_find()

    # Step B.2: Learning Rate Finder Algorithm Explanations
    # FastAI internally increases the learning rate exponentially to find optimal range
    # It plots the loss as learning rate increases
    # Too small => slow learning, too large => divergence

    # Step B.3: Transfer Learning - Freezes and trains
    print("Training with frozen layers (only head)...")
    learn.freeze()
    # Step B.5: Epoch count selected based on LR Finder result and time constraints
    learn.fit_one_cycle(4, lr_max=1e-3)

    # Step B.4: Discriminative Learning Rates - Unfreezes and trains all layers
    print("Unfreezing and training all layers with discriminative learning rates...")
    learn.unfreeze()
    learn.fit_one_cycle(4, lr_max=slice(1e-6, 1e-4))

    # Step A.4.2 - A.4.3: Shows results and confusion matrix
    print("Model Results and Confusion Matrix:")
    learn.show_results()
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(title="Final Model Confusion Matrix")
    plt.show()

    # Model
    learn.export("tumormodelfinal.pkl")

    # Step B.6: Tries higher capacity model (ResNet50) with reduced batch size
    print("Training a larger model (ResNet50)...")
    dls_large = tumor_data_block.dataloaders(path, bs=16, num_workers=0)
    learn_large = vision_learner(dls_large, resnet50, metrics=accuracy)
    learn_large.fine_tune(2)  # Fewer epochs to reduce training time
    learn_large.show_results()

    # Step B.7: Weight Initialization (commentary)
    # PyTorch/fastai use Kaiming initialization by default for ReLU networks.
    # For pretrained models, weights are already well initialized, so not critical.

# Runs
if __name__ == '__main__':
    main()
