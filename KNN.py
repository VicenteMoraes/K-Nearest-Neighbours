import numpy as np
import scipy.io as sio


def knn(testset, testsetlabels, trainset, trainsetlabels, K):
    testerror = 0
    hit = 0
    for index, node in enumerate(testset):
        testnode = np.full_like(trainset, node)
        distance = (testnode - trainset) ** 2
        while len(distance.shape) > 1:
            distance = distance.sum(axis=1)
        distance = distance ** 0.5
        ksmallest = [trainsetlabels[x] for x in np.argsort(distance)[0:K]]
        value, counts = np.unique(ksmallest, return_counts=True)
        guess = value[np.argmax(counts)]
        if guess != testsetlabels[index]:
            testerror += 1
        else:
            hit += 1
    with open("testset.txt", "a") as wf:
        wf.write(f"TEST FOR K={K}\n")
        wf.write(f"ERRORS={testerror}\n")
        wf.write(f"ACCURACY={hit/(hit + testerror)}%\n\n\n")


if __name__ == "__main__":
    data = sio.loadmat("knn_data.mat")
    knn(np.transpose(data["TestSet"]) / 255,
        data["TestSetLabels"],
        np.transpose(data["TrainingSet"]) / 255,
        data["TrainingSetLabels"],
        3)
