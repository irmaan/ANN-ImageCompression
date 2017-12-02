from PIL import Image
import math
import random
from scipy.special import expit
from pathlib import Path
import csv
from os import listdir


v=[] # first layer weights
w=[] # second layer weights
hiddenLayerBias=[]
outputLayerBias=[]
y_in=[]
z_in=[]
DeltaV=[]
DeltaW=[]
DeltaB2=[] # bias second layer
DeltaB1=[] # bias first layer
alpha=0.2

N = 8*8     # number of inputs
P = 9       # number of hidden layer neurons
M = N       # number of ouput neurons

trainTSE=[]  # total squared errors of all neurons for each training sample
trainMSE=[]  # mean squared errors on each  epoch for all training samples
testTSE=[] # total squared errors of all neurons for each test sample
testMSE=[]  # mean squared errors on each  epoch for all test samples
psnr=[]

pictureDimension=256

lenOfTrainClass=10
lenOfTestClass=5

def loadDataSet(set):
    global sizeTuple
    imagesSet=[]
    files=listdir(set)
    for fileName in files:
        img = Image.open(set+'/'+fileName).convert('L')  # convert image to 8-bit grayscale
        WIDTH, HEIGHT = img.size

        rawData = list(img.getdata()) # convert image data to a list of integers
        # convert that to 2D list (list of lists of integers)
        data = [rawData[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]

        allSquares=[]
        for i in range(0,len(data),8):
            for j in range(0,len(data),8):
                square=[]
                for k in range (i,i+8):
                    for l in range (j,j+8):
                        square.append(data[k][l]/255) # dividing by 255 to normalize data in range of (0,1)
                allSquares.append(square)

        imagesSet.append(allSquares)

    return imagesSet



def readWeights(weightFileContent):
    weights=[]
    for weight in weightFileContent:
        rowW=[]
        for i in range(len(weight)):
            if i==0:
                weight[i]=str(weight[i]).replace("[","")
            if i == len(weight) - 1:
                weight[i]=str(weight[i]).replace("]","")
            rowW.append(float(weight[i]))
        weights.append(rowW)

    return weights


def readWeightFiles(weight):
    weightFile = Path(weight +".txt")
    weightList = []
    if weightFile.is_file():
        weightFile = open(weight +".txt", 'r', newline="\n")
        reader = csv.reader(weightFile, delimiter=',')
        weightList = list(reader)
    return weightList


def initWeights():
    global  v,w
    firstLayerWeights=[]
    secondLayerWeights=[]
    vFile = Path("v.txt")
    wFile = Path("w.txt")
    if vFile.is_file() and wFile.is_file():
        vList=readWeightFiles("v")
        wList = readWeightFiles("w")

        w = readWeights(wList)
        v = readWeights(vList)

    else:
        for i in range(N):
            xRowV=[]
            for j in range(P):
                xRowV.append(random.uniform(-0.2,0.2))
            firstLayerWeights.append(xRowV)

        for k in range(P):
            xRowW = []
            for l in range(M):
                xRowW.append(random.uniform(-0.2, 0.2))
            secondLayerWeights.append(xRowW)

        v=firstLayerWeights
        w=secondLayerWeights


def initBias():
    global hiddenLayerBias,outputLayerBias
    bHFile = Path("bH.txt")
    bOFile = Path("bO.txt")

    if bHFile.is_file() and bOFile.is_file():
        bHList = readWeightFiles("bH")
        bOList = readWeightFiles("bO")

        hiddenLayerBias = readWeights(bHList)
        outputLayerBias = readWeights(bOList)

        hTemp=hiddenLayerBias[0]
        hiddenLayerBias=hTemp
        oTemp=outputLayerBias[0]
        outputLayerBias=oTemp


    else:
        for j in range(M):
            outputLayerBias.append(random.uniform(-0.2, 0.2))
        for i in range(P):
            hiddenLayerBias.append(random.uniform(-0.2, 0.2))


def activationFunction(input): # Bipolar Sigmoid
    #return (2/(1+math.exp(-1*input)))-1
    #return (2*(expit(input)))-1
    return expit(input)


def derivationOfActivationFunction(input):
   # return 0.5 *(1+activationFunction(input))*(1-activationFunction(input))
    return activationFunction(input) *(1-activationFunction(input))



def feedForward(X):
    global Z,Y,y_in,z_in
    tempZ=[]
    tempY=[]
    y_in=[]
    z_in=[]
    for j in range(P):
        sumZ=0
        for i in range(N):
            sumZ+=X[i]* v[i][j]
        z_in.append(hiddenLayerBias[j]+sumZ)
        tempZ.append(activationFunction(z_in[j]))
    Z=tempZ

    for k in range(M):
        sumY=0
        for j in range(P):
            sumY+=Z[j]* w[j][k]
        y_in.append(outputLayerBias[k]+sumY)
        tempY.append(activationFunction(y_in[k]))
    Y=tempY

    return Y,Z


def backPropagation(X,Y,Z,target):
    #step 6
    deltaOut=[]
    for k in range(M):
        deltaOut.append((target[k]-Y[k])* derivationOfActivationFunction(y_in[k]))

    for j in range(P):
        deltaRow=[]
        for k in range(M):
            deltaRow.append(alpha* deltaOut[k]*Z[j])
        DeltaW.append(deltaRow)

    for k in range(M):
        DeltaB2.append(alpha * deltaOut[k])

    #step 7
    deltaHidden=[]
    delta_in = []
    for j in range(P):
        sumDelta=0
        for k in range(M):
            sumDelta+=deltaOut[k]*w[j][k]
        delta_in.append(sumDelta)
        deltaHidden.append(delta_in[j] * derivationOfActivationFunction(z_in[j]))

    for i in range(N):
        deltaRow=[]
        for j in range(P):
            deltaRow.append(alpha* deltaHidden[j]*X[i])
        DeltaV.append(deltaRow)

    for j in range(P):
        DeltaB1.append(alpha * deltaHidden[j])


    return deltaOut,deltaHidden

def updateWeights():
    #step 8
    for j in range(P):
        for k in range(M):
            w[j][k]=w[j][k]+DeltaW[j][k]

    for k in range(M) :
            outputLayerBias[k]=outputLayerBias[k]+DeltaB2[k]

    for i in range(N):
        for j in range(P):
            v[i][j]=v[i][j]+DeltaV[i][j]

    for j in range(P):
        hiddenLayerBias[j] = hiddenLayerBias[j] + DeltaB1[j]


def calculateError(Y,target,TSE):
    sumSquaredError = 0
    for i in range(len(Y)):
        sumSquaredError += pow(target[i] - Y[i], 2)
    TSE.append(sumSquaredError)


def calculateMSE(MSE,TSE):
    sumTSEs = 0
    for value in TSE:
        sumTSEs += value
    mse = sumTSEs/len(TSE)
    MSE.append(mse)


def calculatePSNR(MSE,numOfImages):
    global psnr
    for i in range(numOfImages):
        value=((pictureDimension-1)*(pictureDimension-1))/MSE[i]
        psnr.append(10 * math.log10(value))


def stoppingCondition():
    print("Train MSE :%s" % str(trainMSE[-1]))
    if len(trainMSE)>=5:
        #if abs(trainMSE[-1]-trainMSE[len(trainMSE)-2])<=0.001:
        return True
    else :
        return False


def saveWeights(epoch):
    wFile = open('w.txt', 'w')
    wFile.truncate()
    for wRow in w:
        wFile.write("%s\n" % wRow)

    vFile = open('v.txt', 'w')
    vFile.truncate()
    for vRow in v:
        vFile.write("%s\n" % vRow)

    bHFile = open('bH.txt', 'w')
    bHFile.truncate()
    bHFile.write("%s" % hiddenLayerBias)

    bOFile = open('bO.txt', 'w')
    bOFile.truncate()
    bOFile.write("%s" % outputLayerBias)

    epochFile = open('e.txt', 'w')
    epochFile.write("%s\n" % str(epoch))

    validMSEFile = open('validMSE.txt', 'w')
    validMSEFile.truncate()
    for row in testMSE:
        validMSEFile.write("%s\n" % row)

    trainMSEFile = open('trainMSE.txt', 'w')
    trainMSEFile.truncate()
    for row in trainMSE:
        trainMSEFile.write("%s\n" % row)

    psnrFile=open('psnr.txt','w')
    psnrFile.write('%s'% psnr)

def clearSampleVariables():
    y_in.clear()
    z_in.clear()
    DeltaW.clear()
    DeltaV.clear()
    DeltaB1.clear()
    DeltaB2.clear()


def testNetwork(testData,lenOfTest):
    compressedImages=[]
    testTSE.clear()
    for i in range(lenOfTest):
        squares=[]
        for j in range(len(testData[0])):
            X=testData[i][j]
            Y,Z=feedForward(X)
            squares.append(Y)
            if Y!=X:
                calculateError(Y, X, testTSE)

        calculateMSE(testMSE, testTSE)
        compressedImages.append(squares)

    calculatePSNR(testMSE,lenOfTest)
    return  compressedImages


def trainNetwork():
    global alpha
    epoch=0
    while True:
        #if epoch%10==0 and alpha >=0.02:
            #alpha=alpha/2
        trainTSE.clear()
        for i in range(lenOfTrainClass):
            Y = []
            for j in range(len(trainImages[0])):
                    clearSampleVariables()
                    X = trainImages[i][j]
                    Y, Z = feedForward(X)
                    backPropagation(X,Y, Z, X) # second X is target
                    updateWeights()
                    calculateError(Y, X,trainTSE)
        calculateMSE(trainMSE,trainTSE)
        epoch += 1
        saveWeights(epoch)
        if stoppingCondition():
                break

    return epoch

def extractCompressedImages(compressedImages):
    images=[]
    for squares in compressedImages:
        data=[]
        for i in range(0,len(squares),32):
            for k in range(0,len(squares[0]),8):
                for j in range(i, i + 32):
                    for l in range(k,k+8):
                         data.append(int(squares[j][l]*255))
        images.append(data)

    return images


def saveResults(compressedImages):
    for i in range(len(compressedImages)):
        img=compressedImages[i]
        imageOut = Image.new("L", (256, 256))
        imageOut.putdata(img)
        imageOut.save('result'+str(i)+'.jpg')

    sum=0
    for value in psnr:
        sum+=value
    pnsrAvg=sum/len(psnr)

    psnrFile=open('psnr.txt','w')
    psnrFile.write("%s" %psnr)

    return pnsrAvg

trainDataSet=loadDataSet('TrainSet')
trainImages=trainDataSet

testImages=loadDataSet('TestSet')

initWeights()
initBias()

errorRates=[]
finalEpoch=trainNetwork()
print(finalEpoch)
compressedImages=testNetwork(testImages,lenOfTestClass)
finalResultImages=extractCompressedImages(compressedImages)

psnrAvg=saveResults(finalResultImages)

print("FINAL PSNR:")
print(psnrAvg)


