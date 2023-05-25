import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def decay_learning_rate(optimizer, decay_rate):
    # learning rate annealing
    state_dict = optimizer.state_dict()
    lr = state_dict['param_groups'][0]['lr']
    lr *= decay_rate
    for param_group in state_dict['param_groups']:
        param_group['lr'] = lr
    optimizer.load_state_dict(state_dict)
    return optimizer

def save_checkpoint(epoch, model, validation_loss, optimizer, directory, \
                    filename='best.pt'):
    checkpoint=({'epoch': epoch+1,
    'model': model.state_dict(),
    'validation_loss': validation_loss,
    'optimizer' : optimizer.state_dict()
    })
    try:
        torch.save(checkpoint, os.path.join(directory, filename))
        
    except:
        os.mkdir(directory)
        torch.save(checkpoint, os.path.join(directory, filename))


def plot_stroke(strokes, save_name=0):

    n = len(strokes)
    f = plt.figure(figsize=(17 * n, 5))
    width_ratios = []

    for stroke in strokes:
        x = np.cumsum(stroke[:, 1])

        size_x = x.max() - x.min() + 1.

        width_ratios.append(size_x)

    gs = gridspec.GridSpec(1, n, width_ratios=width_ratios, wspace=0.7)

    for i, stroke in enumerate(strokes):
        ax = plt.subplot(gs[i])
        x = np.cumsum(stroke[:, 1])
        y = np.cumsum(stroke[:, 2])

        cuts = np.where(stroke[:, 0] == 1)[0]
        start = 0

        for cut_value in cuts:
            ax.plot(x[start:cut_value], y[start:cut_value],
                    'k-', linewidth=15)
            start = cut_value + 1
        ax.axis('off')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    save_name = "plots/" + str(save_name) + ".png"

    try:
        plt.savefig(
            save_name,
            bbox_inches='tight',
            pad_inches=0.5)
    except Exception:
        print("Error building image!: " + save_name)

    plt.show()
    plt.close()


def plot_concat():
    os.chdir("plots")
    image_files = os.listdir('.')
    image_files.sort(key=lambda f: int(f.split('.')[0]))

    images = [Image.open(img) for img in image_files]

    max_width = max(img.size[0] for img in images)

    images_padded = []
    for img in images:
        width, height = img.size
        new_img = Image.new('RGBA', (max_width, height), (255, 255, 255, 255))  # Создаем новое белое изображение
        new_img.paste(img, (0, 0))  # Вставляем исходное изображение в верхний левый угол
        images_padded.append(new_img)

    total_height = sum(img.size[1] for img in images_padded)
    combined = Image.new('RGBA', (max_width, total_height))  # Создаем новое изображение
    y_offset = 0
    for img in images_padded:
        combined.paste(img, (0, y_offset))
        y_offset += img.size[1]

    os.chdir("..")
    combined.save('combined.png')

    Image.open('combined.png').show()
