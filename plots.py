import matplotlib.pyplot as plt
import os

txt_file = '/home/wcampos/tests/s3bir/train_output_vitb16_sketchy.txt'
txt_name = os.path.splitext(os.path.basename(txt_file))[0]

with open(txt_file) as f:
    lines = f.readlines()

loss = []
mean_ap = []
map_at_5 = []
for line in lines:
    line = line.strip()
    if line[:9] == 'Tiempo en':
        loss.append(float(line.split('loss: ')[1]))
    elif line[:10] == 'nueva mAP@':
        map_at_5.append(float(line.split('modelo: ')[1]))
    elif line[:10] == 'nueva mAP ':
        mean_ap.append(float(line.split('modelo: ')[1]))
    elif line[:7] == 'mAP@5 a':
        map_at_5.append(float(line.split(' ')[4]))
    elif line[:7] == 'mAP act':
        mean_ap.append(float(line.split(' ')[4]))

title = 'ViT-B-16 + BYOL (w/ Adam)\non Sketchy dataset'
# Loss
plt.figure()
plt.title(title)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(loss)
plt.savefig(f'/home/wcampos/tests/s3bir/plots/{txt_name}_loss.jpg')
plt.close()

# mAP
plt.figure()
plt.title(title)
plt.xlabel('Epochs')
plt.plot(mean_ap, label='mAP')
plt.plot(map_at_5, label='mAP@5')
plt.legend()
plt.savefig(f'/home/wcampos/tests/s3bir/plots/{txt_name}_map.jpg')
plt.close()
