from . import InputExample
import csv
import os


class KwTitleDataReader(object):
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def load_dataset(self, filename,encoding='utf-8'):
        data_set = []
        with open(filename, "r", encoding=encoding) as fp:
            for line in fp.readlines():
                items = line.strip().split('\t')
                # [sen_a, sen_b, label] = items 
                data_set.append(items)
        return data_set

    def get_examples(self, filename, max_examples=0):
        """
        data_splits specified which data split to use (train, dev, test).
        """
        dataset = self.load_dataset(filename)
        examples = []
        id = 0
        for sentence_a, sentence_b, label in dataset:
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=self.map_label(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"1": 0, "2": 1, "3": 2}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]