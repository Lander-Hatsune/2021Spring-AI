import torch

class isear(torch.utils.data.Dataset):

    def __init__(self, part):
        super(isear).__init__()

        self.sentences = []
        self.labels = []
        self.lengths = []

        if part == "train":
            sentences_path = "ISEAR/ISEAR ID_train"
            labels_path = "ISEAR/ISEAR_train"
        elif part == "test":
            sentences_path = "ISEAR/ISEAR ID_test"
            labels_path = "ISEAR/ISEAR_test"
        elif part == "validation":
            sentences_path = "ISEAR/ISEAR ID_validation"
            labels_path = "ISEAR/ISEAR_validation"
        else:
            pass
            
        with open(sentences_path) as f:
            self.dic = get_all_words()
            self.sentences = []
            for line in f.readlines():
                numbered = list(map(lambda s: self.dic[s],
                                    line.strip().split()[9:]))
                self.lengths.append(len(numbered))
                self.sentences.append(numbered + [0] * (200 - len(numbered)))
            
            self.sentences = torch.Tensor(self.sentences).to(torch.int32)
            self.lengths = torch.Tensor(self.lengths).to(torch.int32)
            
        with open(labels_path) as f:
            for line in f.readlines():
                self.labels.append(eval(line)[1:])
            self.labels = torch.Tensor(self.labels)

        assert len(self.sentences) == len(self.labels)

    def __getitem__(self, idx):

        return (self.sentences[idx], self.labels[idx], self.lengths[idx])

    def __len__(self):

        return len(self.sentences)

def get_all_words():
    ls = []
    dic = {}
    with open("ISEAR/ISEAR ID") as f:
        for line in f.readlines():
            for word in line.strip().split()[9:]:
                if (word not in ls):
                    ls.append(word)
    for idx, word in enumerate(ls, 0):
        dic[word] = idx
    #dic[''] = 0
    return dic

