import os
from parse_annotation import parse_annotation

class_labels = ['RBC', 'WBC', 'Platelets']
directory = '../dataset/Testing/Annotations/'
class_data = []
for ann_file in os.listdir(directory):
    ann_dir = directory + ann_file
    ground_truths, labels = parse_annotation(ann_dir, class_labels)
    # print(ann_file, ground_truths, labels)
    for gtobject in ground_truths:
        for entry in gtobject['object']:
            w = entry['xmax'] - entry['xmin']
            h = entry['ymax'] - entry['ymin']
            r = max(w, h) / 2
            df_row = {'type':'groundTruth','file_name':ann_file, 'label':entry['name'], 'center_x':entry['center_x'], 'center_y':entry['center_y'], 'radius': r, 'confidence': 1.0}
            class_data.append(df_row)

import pandas as pd
df = pd.DataFrame(class_data)
df.to_csv('../output/ground_truth.csv', index=False)
