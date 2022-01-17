'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang 
About: An utility function for getting data for my repository: text-matching-explained
'''

from random import seed, sample, shuffle

def get_quora_data():
    data = open('quora_duplicate_questions.tsv', 'r').readlines()
    corpus = []

    # we do not want the header to be included 
    for line in data[1:]:
        line = line.split('\t')
        try:
            # If this cannot be done, there is a problem and we do not want to save this example
            text_a, text_b, label = line[-3], line[-2], line[-1].strip()
            int(label) # just a test, to make sure that the label is convertible to int
            corpus.append([text_a, text_b, label])
        except:
            pass
    
    matched = [c for c in corpus if c[-1] == "1"]
    unmatched = [c for c in corpus if c[-1] == "0"]

    seed(32)
    part1 = sample(matched, 2500)
    part2 = sample(unmatched, 2500)

    train = part1[:1500] + part2[:1500]
    dev = part1[1500:2000] + part2[1500:2000]
    test = part1[2000:] + part2[2000:]


    shuffle(train)
    shuffle(dev)
    shuffle(test)


    def save(dataset, fpath):
        dataset = ['\t'.join(d) for d in dataset]
        with open(fpath, 'w') as f:
            f.write('\n'.join(dataset))
            f.close()
            print(f"{fpath} has been saved!")

    save(train, "train.txt")
    save(dev, "dev.txt")
    save(test, "test.txt")


if __name__=='__main__':    
    do()
