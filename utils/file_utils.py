import os
import csv



def write_to_csv(csv_rows, filename, file_path, seed=None):
    if not seed is None:
        file_path = f'{file_path}{seed}/' + filename
    else:
        file_path = f'{file_path}' + filename
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
        for row in csv_rows:
            writer.writerow(row)
            csvfile.flush()


def make_dir(path, seed = None):
    if not os.path.exists(path):
        os.mkdir(path)
    if not seed is None:
        if not os.path.exists(f'{path}/{seed}'):
            os.mkdir(f'{path}/{seed}')

