"""
plot training and validation loss for usegeo fine-tuning.
similar to figure a1 in the m4depth paper.

usage:
    python plot_usegeo_training.py --history_file=ckpt_usegeo/training_history.json
"""

import json
import matplotlib.pyplot as plt
import argparse
import os


def plot_training_history(history_file, output_file='usegeo_training_loss.png'):
    """
    plot training and validation loss curves.
    generates a figure similar to figure a1 in the paper.
    """
    with open(history_file) as f:
        history = json.load(f)

    epochs = history['epochs']
    train_loss = history['train_loss']
    val_loss = history['val_loss']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('UseGeo Fine-tuning: Training and Validation Loss', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # set reasonable y-axis limits
    max_loss = max(max(train_loss), max(val_loss))
    min_loss = min(min(train_loss), min(val_loss))
    margin = (max_loss - min_loss) * 0.1
    ax.set_ylim(min_loss - margin, max_loss + margin)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"saved: {output_file}")

    # also print summary statistics
    print("\ntraining summary:")
    print(f"  initial train loss: {train_loss[0]:.4f}")
    print(f"  final train loss: {train_loss[-1]:.4f}")
    print(f"  initial val loss: {val_loss[0]:.4f}")
    print(f"  final val loss: {val_loss[-1]:.4f}")
    print(f"  best val loss: {min(val_loss):.4f} (epoch {val_loss.index(min(val_loss)) + 1})")


def main():
    parser = argparse.ArgumentParser(description='Plot UseGeo training history')
    parser.add_argument('--history_file', type=str, default='ckpt_usegeo/training_history.json',
                        help='path to training history json file')
    parser.add_argument('--output_file', type=str, default='usegeo_training_loss.png',
                        help='output image file')
    args = parser.parse_args()

    if not os.path.exists(args.history_file):
        print(f"error: history file not found: {args.history_file}")
        print("run training first: python train_usegeo.py --from_midair_ckpt --epochs=50")
        return

    plot_training_history(args.history_file, args.output_file)


if __name__ == '__main__':
    main()
