from torch.utils.data import Dataset
from os import path
import cv2
import glob


class MOBendCnnDataset(Dataset):
    def __init__(self, directory):
        self.coordinates_dictionary = path.join(directory, 'coordinates')
        self.files = glob.glob(path.join(directory, '*.png'))
        self.coordinates_dictionary = self.coordinates_dictionary.split(';')
        self.coordinates = {}
        for line in self.coordinates_dictionary:
            key_word = line.split()
            self.coordinates[key_word[0]] = key_word[1]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        name = self.files[item]
        img = cv2.imread(name)
        coord = self.coordinates[name]
        return img, coord
