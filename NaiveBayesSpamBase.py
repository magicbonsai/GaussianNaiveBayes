# Alex Zhu
# This was written for Python 3.4.2 to practice Gaussian Naive Bayes on the UCI SpamBase Dataset
# This implementation of Gaussian Naive Bayes does not use numpy, only the math libraries
from __future__ import division
import math

# 57 features, 4601 items for reference

class SpamBayes:

    # init
    # All variables
    def __init__(self, features, entries):
        self.entries = entries
        self.features = features
        self.pSpam = 0
        self.pNotSpam = 0
        self.means = [0.0 for x in range(features)]
        self.stdevs = [0.0 for x in range(features)]
        self.meansN = [0.0 for x in range(features)]
        self.stdevsN = [0.0 for x in range(features)]
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.accuracy = 0

    # For the functions stDev_Means and GaussianBayes x refers to the class defined as 'Spam' and y
    # refers to the class known as 'Not Spam'

    # This could definitely be cleaner.
    # Manual calculating of mean and stdev without libraries;
    # I ran into an issue where I could not figure out where python was
    # returning results as ints rather than floats this is the result
    def stDev_Means(self, trainset):
        for i in range(self.features):
            x = 0.0
            y = 0.0
            for j in range(len(trainset)):
                if(int(trainset[j][-1]) == 1 ):
                    x += float(trainset[j][i])
                else:
                    y += float(trainset[j][i])
            self.means[i] = x / float(len(trainset))
            self.meansN[i] = y / float(len(trainset))

        for i in range(self.features):
            x = 0.0
            y = 0.0
            for j in range(len(trainset)):
                if (int(trainset[j][-1]) == 1):
                    x += abs(float(trainset[j][i]) - self.means[i]) ** 2
                else:
                    y += abs(float(trainset[j][i]) - self.meansN[i]) ** 2
            x = x / float(len(trainset) - 1)
            y = y / float(len(trainset) - 1)
            self.stdevs[i] = math.sqrt(x)
            self.stdevsN[i] = math.sqrt(y)

    # Runs Gaussian Naive Bayes on one set of features
    # I decided to run the Gaussian expression as a sum of absolutes, so the resulting argmax is reversed:
    # so it's not an argmax expression anymore, but an argmin expression
    # It is functionally the same argmax expression since the products of all 58 Gaussian operations
    # results in a very small decimal number, which translate to only negative numbers as logs.
    # I decided to stick with this since it still delivers the appropriate results compared to the same
    # if I did not apply absolute values to all 58 arguments.
    def GaussianBayes(self, featureset):
        # math.exp for e
        # math.log for natural log

        const = math.sqrt(2*math.pi)

        # Priors for the classes Spam and Not Spam
        x = math.log(self.pSpam)
        y = math.log(self.pNotSpam)
        for i in range(self.features):
            # This section tries to account purely for zero stdev and zero/negative logarithmic computation
            a = self.stdevs[i]
            b = self.stdevsN[i]
            stdP = a if a > 0 else 0.0001
            stdN = b if b > 0 else 0.0001
            # Gaussian expression
            gB = ((1.0 / (const * stdP)) * math.exp(-1*(((float(featureset[i]) - self.means[i])**2) / (2 * (stdP**2)))))
            # Expression to account for zero/negative Gaussian Algorithm results
            gB = 0.0001 if gB <= 0 else gB
            # log of the Gaussian result
            gB = math.log(gB)

            # Repeated for the opposite case
            gBn = ((1.0 / (const * stdN)) * math.exp(-1*(((float(featureset[i]) - self.meansN[i])**2) / (2 * (stdN**2)))))
            gBn = 0.0001 if gBn <= 0 else gBn
            gBn = math.log(gBn)

            # Sum of the Absolute Gaussian results
            x += abs(gB)
            y += abs(gBn)


        if x < y:
            # This is classified as spam
            if int(featureset[-1] == 1):
                self.tp += 1
                self.accuracy += 1
            else:
                self.fp += 1

        else:
            # This is not spam
            if int(featureset[-1]) == 0:
                self.tn += 1
                self.accuracy += 1
            else:
                self.fn += 1


# main loop code
if __name__ == '__main__':

    # init a new SpamBayes class object
    sb = SpamBayes(57, 4601)

    # data holds the raw spambase numbers

    data = []
    count  = 0.0
	# Parse spambase data.  This implys the file 'spambase.data' exists within the same directory as the
    # '.py' file

    with open('spambase.data', 'r') as filestream:
        for line in filestream:
            curr = []
            currentline = line.split(",")
            for feature in currentline:
                # Makes sure parsed data is in float, not as string.
                curr.append(float(feature))
            data.append(curr)
            count += float(currentline[-1])


    # This is the priors
    sb.pSpam = count / float(sb.entries)
    sb.pNotSpam = 1.0 - sb.pSpam

    # print('Prior for if Spam: ', sb.pSpam)
    # print('Prior if not Spam: ', sb.pNotSpam)


    # split data into four parts to make 40 60 splits and then concatenate into 2 arrays for training and testing
    # I am hyper aware there is probably a better method to do this

    data1 = data[:919]
    data2 = data[920:1839]
    data3 = data[1840:3219]
    data4 = data[3220:4600]

    # Creates the new arrays for train and test, both with 40%~ roughly as Spam entries
    train = data1 + data3
    test = data2 + data4

    # Get the stdevs and means
    sb.stDev_Means(train)

    # Run the Gaussian Naive Bayes on the test section of the spambase data
    for line in test:
        sb.GaussianBayes(line)

    # Results
    print('True Positive Rate: ', sb.tp)
    print('False Positive Rate: ', sb.fp)
    print('True Negative Rate: ', sb.tn)
    print('False Negative Rate: ', sb.fn)
    print('Accuracy: ', sb.accuracy / len(test))