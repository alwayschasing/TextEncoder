from . import InputExample
import csv
import os


class SimTextDataReader(object):
    def __init__(self,):
        pass

    def load_dataset(self, filename):
        data_set = []
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp.readlines():
                items = line.rstrip('\n').split('\t')
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
        check_limit = 10
        for idx, (sentence_a, sentence_b, label) in enumerate(dataset):
            guid = "%s-%d" % (filename, id)
            id += 1
            examples.append(InputExample(guid=guid, texts=[sentence_a, sentence_b], label=float(label)))

            if idx < check_limit:
                print("[check example]%s\t%s\t%f"%(sentence_a.encode("utf-8").decode("unicode_escape"), sentence_b.encode("utf-8").decode("unicode_escape"),float(label)))

            if 0 < max_examples <= len(examples):
                break

        return examples

    @staticmethod
    def get_labels():
        return {"0": 0.0, "1": 1.0}

    def get_num_labels(self):
        return len(self.get_labels())

    def map_label(self, label):
        return self.get_labels()[label.strip().lower()]
