import numpy as np
import scipy.io as sio


def knn(testset, testsetlabels, trainset, trainsetlabels, K):
    testerror = 0
    for index, node in enumerate(testset):
        testnode = np.full_like(trainset, node)
        distance = (testnode - trainset) ** 2
        while len(distance.shape) > 1:
            distance = distance.sum(axis=1)
        ksmallest = [trainsetlabels[x] for x in np.argsort(distance)[0:K]]
        value, counts = np.unique(ksmallest, return_counts=True)
        guess = value[np.argmax(counts)]
        if guess != testsetlabels[index]:
            testerror += 1
    with open("testset.txt", "a") as wf:
        wf.write(f"TEST FOR K={K}\n")
        wf.write(f"ERRORS={testerror}\n")
        wf.write(f"ACCURACY={(10**4-testerror)/(10**4)}%\n\n\n")


if __name__ == "__main__":
    data = sio.loadmat("knn_data.mat")
    for k in range(50, 550, 50):
        knn(np.transpose(data["TestSet"]),
            data["TestSetLabels"],
            np.transpose(data["TrainingSet"]),
            data["TrainingSetLabels"],
            k)

