import numpy as np
import time
import os
import glob
from PIL import Image, ImageDraw,ImageFont
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import linear_model
import pickle
import os.path
from pathlib import Path
from sklearn.neural_network import MLPRegressor

def expandMatrix(grayMatrix,photo_name):
    height,width = grayMatrix.shape
    photoName = os.path.splitext(os.path.basename(photo_name))[0]
    open("data/data_"+photoName+".csv","w").close()
    with open("data/data_"+photoName+".csv", "a") as resultsFile:
        for i in range(height):
            for j in range(width):
                if(i - 1 == -1 or j - 1 == -1 or i + 1 == height or j + 1 == width):
                    g1 = 0
                    g2 = 0
                    g3 = 0
                    g4 = 0
                    g5 = 0
                    g6 = 0
                    g7 = 0
                    g8 = 0
                    g9 = 0
                else:
                    g1 = grayMatrix[i - 1][j - 1]
                    g2 = grayMatrix[i - 1][j]
                    g3 = grayMatrix[i - 1][j + 1]
                    g4 = grayMatrix[i][j - 1]
                    g5 = grayMatrix[i][j]
                    g6 = grayMatrix[i][j + 1]
                    g7 = grayMatrix[i + 1][j - 1]
                    g8 = grayMatrix[i + 1][j]
                    g9 = grayMatrix[i + 1][j + 1]
                resultsFile.write(str(g1) + "," + str(g2) + "," + str(g3) + "," + str(g4) + "," + str(g5) + "," + str(g6) + "," + str(g7) + "," + str(g8) + "," + str(g9) + "\n")
    expandedMatrix = np.genfromtxt("data/data_"+photoName+".csv", dtype='i', delimiter=",")
    return expandedMatrix

def createTrainingData(image, grayMatrix,photo_name):
    height,width = grayMatrix.shape
    photoName = os.path.splitext(os.path.basename(photo_name))[0]
    print("Creating input for "+photo_name+"...")
    open("train/input_"+photoName+".csv","w").close()
    with open("train/input_"+photoName+".csv", "a") as resultsFile:
        for i in range(height):
            for j in range(width):
                if(i - 1 == -1 or j - 1 == -1 or i + 1 == height or j + 1 == width):
                    g1 = 0
                    g2 = 0
                    g3 = 0
                    g4 = 0
                    g5 = 0
                    g6 = 0
                    g7 = 0
                    g8 = 0
                    g9 = 0
                else:
                    g1 = grayMatrix[i - 1][j - 1]
                    g2 = grayMatrix[i - 1][j]
                    g3 = grayMatrix[i - 1][j + 1]
                    g4 = grayMatrix[i][j - 1]
                    g5 = grayMatrix[i][j]
                    g6 = grayMatrix[i][j + 1]
                    g7 = grayMatrix[i + 1][j - 1]
                    g8 = grayMatrix[i + 1][j]
                    g9 = grayMatrix[i + 1][j + 1]
                resultsFile.write(str(g1) + "," + str(g2) + "," + str(g3) + "," + str(g4) + "," + str(g5) + "," + str(g6) + "," + str(g7) + "," + str(g8) + "," + str(g9) + "\n")
    print("Input generated to train/input_"+photoName+".csv.")
    width,height = image.size
    pixelList = list(image.getdata())
    print("Creating colors for "+photo_name+"...")
    pixelMatrix = np.asarray(pixelList,dtype='i,i,i')
    pixelMatrix = np.reshape(pixelMatrix,(height*width,1))
    h, w = pixelMatrix.shape
    open("train/color_"+photoName+".csv","w").close()
    with open("train/color_"+photoName+".csv", "a") as resultsFile:
        for i in range(h):
            for j in range(w):
                resultsFile.write(str(pixelMatrix[i][j][0]) + "," + str(pixelMatrix[i][j][1]) + "," + str(pixelMatrix[i][j][2]) + "\n")
    print("Colors generated to train/color_"+photoName+".csv.")

def visualizeInput1(inputImage):
    numPatches,patchSize = inputImage.shape

    image = Image.new('L', (597, 798))

    centerGray = np.zeros((numPatches,1),'uint8')
    for i in range(numPatches):
        centerGray[i] = inputImage[i][5]
    centerGray = np.reshape(centerGray, (597, 798))
    rgbGray = np.zeros((597, 798, 3), 'uint8')
    rgbGray[...,0] = centerGray
    rgbGray[...,1] = centerGray
    rgbGray[...,2] = centerGray
    image = Image.fromarray(rgbGray)
    return image, rgbGray

def visualizeDataFile(inputImage):
    numPatches,patchSize = inputImage.shape
    image = Image.new('L', (361,641))

    centerGray = np.zeros((numPatches,1),'uint8')
    for i in range(numPatches):
        centerGray[i] = inputImage[i][5]
    centerGray = np.reshape(centerGray,(361,641))
    rgbGray = np.zeros((361,641,3),'uint8')
    rgbGray[...,0] = centerGray
    rgbGray[...,1] = centerGray
    rgbGray[...,2] = centerGray
    image = Image.fromarray(rgbGray)
    return image, rgbGray

def visualizeInputFile(inputImage):
    numPatches,patchSize = inputImage.shape
    image = Image.new('L', (174, 281))
    centerGray = np.zeros((numPatches,1),'uint8')
    for i in range(numPatches):
        centerGray[i] = inputImage[i][5]
    centerGray = np.reshape(centerGray,(174, 281))
    image = Image.fromarray(centerGray)
    return image, centerGray

def grayscale(color_image):
    bw = color_image.convert('L')
    return bw

def getGrayscaleMatrix(grayscale_image):
    width,height = grayscale_image.size
    grayList = list(grayscale_image.getdata())
    grayscaleMatrix = np.asarray(grayList,dtype='i')
    grayscaleMatrix = np.reshape(grayscaleMatrix,(height,width))
    return grayscaleMatrix

def colorize(grayscale_image, colorLabels, resultofprediction, numberOfClusters,numberofgrayscalecolors):
    grayMatrix = getGrayscaleMatrix(grayscale_image)
    height,width = grayMatrix.shape
    image = Image.new('RGB',(height,width))
    colorList = []
    for i in range(height):
        for j in range(width):
            colorList.append((grayMatrix[i][j],grayMatrix[i][j],grayMatrix[i][j]))
    colorMatrix = np.asarray(colorList,dtype='i,i,i')
    colorMatrix = np.reshape(colorMatrix,(height,width))
    colors = np.zeros((height,width,3),'uint8')
    if(method == 'H' or method == 'h'):
        for i in range(height):
            for j in range(width):
                # DO COLORIZATION BASED ON CLASSIFICATION HERE; sample below
                if grayMatrix[i][j] < 75:
                    colorMatrix[i][j] = (91, 97, 23)
                elif grayMatrix[i][j] < 100:
                    colorMatrix[i][j] = (51, 79, 162)
                elif grayMatrix[i][j] < 200:
                    colorMatrix[i][j] = (70, 131, 201)
                else:
                    colorMatrix[i][j] = (255,245,246)
        for i in range(height):
            for j in range(width):
                colors[i][j] = tuple((colorMatrix[i][j][0],colorMatrix[i][j][1],colorMatrix[i][j][2])) 
        image = Image.fromarray(colors)
        open("hardcoded_color_output.csv","w").close()
        with open('hardcoded_color_output.csv', "a") as resultsFile:
            for i in range(height):
                for j in range(width):
                    resultsFile.write(str(colors[i][j][0]) + "," + str(colors[i][j][1]) + "," + str(colors[i][j][2]) + "\n")
        print("Hardcoded color results generated to hardcoded_color_output.csv.")
    else:
        resultofprediction = np.reshape(resultofprediction,(height,width))
    #            print("Results of Prediction:")
    #            print(resultofprediction)
        for i in range(height):
            for j in range(width):
                for k in range(len(colorLabels)):
                    if(resultofprediction[i][j] == colorLabels[k][0]):
                        colorMatrix[i][j] = (colorLabels[k][1],colorLabels[k][2],colorLabels[k][3])
        for i in range(height):
            for j in range(width):
                colors[i][j] = tuple((colorMatrix[i][j][0],colorMatrix[i][j][1],colorMatrix[i][j][2])) 
    #            print("Colors:")
    #            print(colors)
        image = Image.fromarray(colors)
        open("logistic_color_output.csv","w").close()
        with open('logistic_color_output.csv', "a") as resultsFile:
            for i in range(height):
                for j in range(width):
                    resultsFile.write(str(colors[i][j][0]) + "," + str(colors[i][j][1]) + "," + str(colors[i][j][2]) + "\n")
        print("Logistic color results generated to logistic_color_output.csv.")
    return image

def colorizeWithNN(result_r, height, width):
    numPixels,dim = result_r.shape
    image = Image.new('RGB', (width,height))
#    print("Height: " + str(height) +", width: " +str(width))
#    print("Num pixels: " + str(int(numPixels/3)) +", dimensions: " +str(dim))
    result_r = np.reshape(result_r,(int(numPixels/3),3))
#    print(result_r)
#    result_r_r = result_r[:,0]
#    result_r_g = result_r[:,1]
#    result_r_b = result_r[:,2]
#    r = np.reshape(result_r_r,(height,width))
#    print(r)
#    g = np.reshape(result_r_g,(height,width))
#    print(g)
#    b = np.reshape(result_r_b,(height,width))
#    print(b)
#    rgbGray = np.zeros((height,width,3),'uint8')
#    rgbGray[...,0] = r
#    rgbGray[...,1] = g
#    rgbGray[...,2] = b
    open("neural_color_output.csv","w").close()
    with open('neural_color_output.csv', "a") as resultsFile:
        for i in range(int(numPixels/3)):
                resultsFile.write(str(result_r[i][0]) + "," +str(result_r[i][1]) + "," +str(result_r[i][2]) + "\n")
    print("Neural network color results generated to neural_color_output.csv.")
    print("Coloring image...")
    colors = np.genfromtxt('neural_color_output.csv', dtype='i,i,i', delimiter=",")
    colors = np.reshape(colors,(height,width))
    colorMatrix = np.zeros((height,width,3),'uint8')
    for i in range(height):
        for j in range(width):
            colorMatrix[i][j] = tuple((colors[i][j][0],colors[i][j][1],colors[i][j][2]))
    image = Image.fromarray(colorMatrix)
    return image

def concatenateWithoutOriginal(grayscale, colorized):
    widths, heights = zip(*(grayscale.size for i in range(0,2)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height + int(max_height/8)))
    x_offset = 0

    new_im.paste(grayscale, (x_offset,0))
    draw = ImageDraw.Draw(new_im)
    fnt = ImageFont.truetype("arial.ttf",int(max_height/10))
    draw.text(((total_width/4)-50,max_height), "Grayscale", font = fnt, fill=(255,255,255,128))

    x_offset += grayscale.size[0]

    new_im.paste(colorized, (x_offset,0))
    draw.text(((3*total_width/4)-50,max_height), "Colorized", font = fnt, fill=(255,255,255,128))

    return new_im

def concatenateWithOriginal(image, grayscale, colorized):
    widths, heights = zip(*(image.size for i in range(0,3)))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height + int(max_height/8)))
    x_offset = 0

    new_im.paste(image, (x_offset,0))
    draw = ImageDraw.Draw(new_im)
    fnt = ImageFont.truetype("arial.ttf",int(max_height/10))
    draw.text(((total_width/6)-50,max_height), "Original", font = fnt, fill=(255,255,255,128))

    x_offset += image.size[0]

    new_im.paste(grayscale, (x_offset,0))
    draw.text(((total_width/2)-50,max_height), "Grayscale", font = fnt, fill=(255,255,255,128))

    x_offset += image.size[0]
    new_im.paste(colorized, (x_offset,0))
    draw.text(((5*total_width/6)-50,max_height), "Colorized", font = fnt, fill=(255,255,255,128))

    return new_im

def generateResults(image, gray, colorLabels, resultofprediction, numberOfClusters,numberofgrayscalecolors):
    gray_image = grayscale(image)
    colorized_image = colorize(gray_image, colorLabels, resultofprediction, numberOfClusters,numberofgrayscalecolors)
    if(gray == 'C' or gray == 'c'):
        new_im = concatenateWithOriginal(image, gray_image, colorized_image)
    else:
        if (option == 'D' or option == 'd'):
            original = Image.open('original_data.jpg')
            width,height = image.size
            original = original.resize((width,height))
            new_im = concatenateWithOriginal(original, image, colorized_image)
        else:
            new_im = concatenateWithoutOriginal(image, colorized_image)
    return new_im

def generateResultsWithNN(image, gray, resultofprediction):
    gray_image = grayscale(image)
    width, height = gray_image.size
    colorized_image = colorizeWithNN(resultofprediction, height, width)
    if(gray == 'C' or gray == 'c'):
        new_im = concatenateWithOriginal(image, gray_image, colorized_image)
    else:
        if (option == 'D' or option == 'd'):
            original = Image.open('original_data.jpg')
            width,height = image.size
            original = original.resize((width,height))
            new_im = concatenateWithOriginal(original, image, colorized_image)
        else:
            new_im = concatenateWithoutOriginal(image, colorized_image)
    return new_im

def plotRGB(rgbdata):
    # DRAW RGB OF COLOR TO GUESS NUMBER OF CLUSTER
    paleta=list(zip(*rgbdata))
    fig = plt.figure(figsize=(600,500))
    ax = Axes3D(fig)
    ax.scatter(paleta[0],paleta[1],paleta[2], c=[(r[0] / 255., r[1] / 255., r[2] / 255.) for r in rgbdata])
    ax.grid(False)
    ax.set_title('grid on')
    plt.interactive(False)
    plt.show()

def clusterColor(rgbdata,colorFile,numberofcluster):
    if(colorFile == 'color.csv'):
        filename = "models/colors_kmeans_"+str(numberofcluster)+"clusters_model.sav"
        rgbfilename = "models/colorwith"+str(numberofcluster)+"clusters.csv"
    else:
        colorFileName = os.path.splitext(os.path.basename(inputFile))[0]
        filename = "models/"+colorFileName+"_kmeans_"+str(numberofcluster)+"clusters_model.sav"
        rgbfilename = "models/"+colorFileName+"_with"+str(numberofcluster)+"clusters.csv"
    myfile = Path(filename)
    rgbfile = Path(rgbfilename)
    if (myfile.is_file() and rgbfile.is_file()):
        rgbwithlabels = np.genfromtxt(rgbfilename, dtype='i', delimiter=",")
        model = pickle.load(open(filename, 'rb'))
    elif(myfile.is_file()):
        model = pickle.load(open(filename, 'rb'))
        rgbwithlabels = np.concatenate((rgbdata, model.labels_[:, None]), axis=1)
        cols = len(rgbdata[0])
        rows = len(rgbdata)
        rgbdatacenter = np.zeros(shape=(rows, cols))
        uniquelabels = np.unique(model.labels_)
        for label in uniquelabels:
            listofindex, = np.where(model.labels_ == label)
            rgbdatacenter[listofindex] = model.cluster_centers_[label]
        rgbwithlabels = np.concatenate((rgbwithlabels, rgbdatacenter), axis=1)
        pickle.dump(model, open(filename, 'wb'))
        np.savetxt(rgbfilename, rgbwithlabels.astype(np.int), fmt='%d', delimiter=",")
    elif(rgbfile.is_file()):
        rgbwithlabels = np.genfromtxt(rgbfilename, dtype='i', delimiter=",")
        model = KMeans(n_clusters=numberofcluster, random_state=0).fit(rgbdata)
        pickle.dump(model, open(filename, 'wb'))
    else:
        model = KMeans(n_clusters=numberofcluster, random_state=0).fit(rgbdata)
        rgbwithlabels = np.concatenate((rgbdata, model.labels_[:, None]), axis=1)
        cols = len(rgbdata[0])
        rows = len(rgbdata)
        rgbdatacenter = np.zeros(shape=(rows, cols))
        uniquelabels = np.unique(model.labels_)
        for label in uniquelabels:
            listofindex, = np.where(model.labels_ == label)
            rgbdatacenter[listofindex] = model.cluster_centers_[label]
        rgbwithlabels = np.concatenate((rgbwithlabels, rgbdatacenter), axis=1)
        pickle.dump(model, open(filename, 'wb'))
        np.savetxt(rgbfilename, rgbwithlabels.astype(np.int), fmt='%d', delimiter=",")
    return rgbwithlabels, model

def clusterGrayScale(grayscaledata,inputFile,numberofcluster):
    if (inputFile == 'input.csv'):
        filename="models/grayscale_kmeans_"+str(numberofcluster)+"clusters_model.sav"
        grayfilename="models/inputwith"+str(numberofcluster)+"clusters.csv"
    else:
        inputFileName = os.path.splitext(os.path.basename(inputFile))[0]
        filename="models/"+inputFileName+"_grayscale_kmeans_"+str(numberofcluster)+"clusters_model.sav"
        grayfilename="models/"+inputFileName+"_with"+str(numberofcluster)+"clusters.csv"
    myfile=Path(filename)
    grayfile=Path(grayfilename)
    if(myfile.is_file() and grayfile.is_file()):
        grayscalewithlabels=np.genfromtxt(grayfilename, dtype='i', delimiter=",")
        model = pickle.load(open(filename, 'rb'))
    elif(myfile.is_file()):
        model = pickle.load(open(filename, 'rb'))
        grayscalewithlabels = np.concatenate((grayscaledata, model.labels_[:, None]), axis=1)
        cols=len(grayscaledata[0])
        rows=len(grayscaledata)
        grayscalecenter=np.zeros(shape=(rows, cols))
        uniquelabels=np.unique(model.labels_)
        for label in uniquelabels:
            listofindex, = np.where(model.labels_ == label)
            grayscalecenter[listofindex]=model.cluster_centers_[label]
        grayscalewithlabels = np.concatenate((grayscalewithlabels, grayscalecenter), axis=1)
        np.savetxt(grayfilename, grayscalewithlabels.astype(np.int), fmt='%d', delimiter=",")
        #print(type(kmeans.cluster_centers_))
        #paleta2 = list(zip(*kmeans.cluster_centers_))
    elif (grayfile.is_file()):
        grayscalewithlabels=np.genfromtxt(grayfilename, dtype='i', delimiter=",")
        model = KMeans(n_clusters=numberofcluster, random_state=0).fit(grayscaledata)
        pickle.dump(model, open(filename, 'wb'))
    else:
        model = KMeans(n_clusters=numberofcluster, random_state=0).fit(grayscaledata)
        grayscalewithlabels = np.concatenate((grayscaledata, model.labels_[:, None]), axis=1)
        cols=len(grayscaledata[0])
        rows=len(grayscaledata)
        grayscalecenter=np.zeros(shape=(rows, cols))
        uniquelabels=np.unique(model.labels_)
        for label in uniquelabels:
            listofindex, = np.where(model.labels_ == label)
            grayscalecenter[listofindex]=model.cluster_centers_[label]
        grayscalewithlabels = np.concatenate((grayscalewithlabels, grayscalecenter), axis=1)
        np.savetxt(grayfilename, grayscalewithlabels.astype(np.int), fmt='%d', delimiter=",")
        pickle.dump(model, open(filename, 'wb'))
    return grayscalewithlabels,model

def clustercolorandgrayscale(inputData,rgbdata, numberOfCluster,numberofgrayscalecolor):
    #if(numberOfCluster > 100):
    #    numberOfCluster = 100
    if(numberofgrayscalecolor > 500):
        numberofgrayscalecolor = 500
    print("Clustering grayscale...")
    grayscalewithlabels,modelgrayscale = clusterGrayScale(inputData,inputFile,numberofgrayscalecolor)
    print("Clustering colors...")
    colorwithlabels, modelcolor = clusterColor(rgbdata, colorFile, numberOfCluster)
    return colorwithlabels, grayscalewithlabels,modelcolor,modelgrayscale,numberOfCluster,numberofgrayscalecolor


def ClassificationWithLogisticRegression(grayscaleimage,colorlabels,newgrayscaleimage,modelgrayscale,numberOfCluster,numberofgrayscalecolor):
    filename = "models/logistic_regression_color_"+str(numberOfCluster)+"_gray_"+str(numberofgrayscalecolor)+"_model.sav"
    myfile = Path(filename)
    if (myfile.is_file()):
        logreg = pickle.load(open(filename, 'rb'))
    else:
        logreg = linear_model.LogisticRegression(solver='lbfgs', C=1e2, multi_class="multinomial")

        # we create an instance of Neighbours Classifier and fit the data.
        grayscaleimage = grayscaleimage.reshape(-1, 1)
        # colorlabels.reshape(-1, 1)
        logreg.fit(grayscaleimage, colorlabels)
        pickle.dump(logreg, open(filename, 'wb'))

    newgrayscalecluster = modelgrayscale.predict(newgrayscaleimage)
    newgrayscalecluster=newgrayscalecluster.reshape(-1, 1)
    return logreg.predict(newgrayscalecluster)

def ClassificationWithLogisticRegression2(grayscaleimage,colorlabels,newgrayscaleimage,numberOfCluster,numberofgrayscalecolor):
    filename = "models/logistic_regression2_color_"+str(numberOfCluster)+"_gray_"+str(numberofgrayscalecolor)+"_model.sav"
    myfile = Path(filename)
    if (myfile.is_file()):
        logreg = pickle.load(open(filename, 'rb'))
    else:
        logreg = linear_model.LogisticRegression(solver='lbfgs', C=1e2, multi_class="multinomial")

        logreg.fit(grayscaleimage, colorlabels)
        pickle.dump(logreg, open(filename, 'wb'))


    return logreg.predict(newgrayscaleimage)

def ClassificationWithNN(inputFile,colorFile,graypatches, rgbdata, newgraypatches):
    if(inputFile == 'input.csv'):
        filename_r = "neural_network_model_r.sav"
    else:
        inputFileName = os.path.splitext(os.path.basename(inputFile))[0]
        filename_r = "models/"+ inputFileName +"_neural_network_model_r.sav"
    myfile_r = Path(filename_r)
    if (myfile_r.is_file()):
        clf_r = pickle.load(open(filename_r, 'rb'))
    else:
        clf_r = MLPRegressor(solver='sgd', alpha=1e-5, hidden_layer_sizes=(20,10,5,), random_state=1, activation='logistic', max_iter=100000, tol=1e-6)


        clf_r.fit(graypatches/255, rgbdata/255)
        pickle.dump(clf_r, open(filename_r, 'wb'))
        
    result_r = clf_r.predict(newgraypatches/255)
    result_r = result_r*255;
    return result_r

if __name__ == '__main__':
#    print("Visualizing input.csv and saving into inputImage.jpg...")
#    inputFile = np.genfromtxt("input.csv",dtype='i',delimiter=",")
#    inputImage, inputMatrix = visualizeInputFile(inputFile)
#    inputImage.save('inputImage.jpg')
    clusterfiles = input("Default training data (Y or N)? ")
    if (clusterfiles == 'Y'):
        inputFile = 'input.csv'
        colorFile = 'color.csv'
    else:
        inputFile = input("Enter input training filepath: ")
        colorFile = input("Enter color training filepath: ")
    print("Loading training data...")
    start = time.time()
    inputData = np.genfromtxt(inputFile, dtype='i', delimiter=",")
    rgbData = np.genfromtxt(colorFile, dtype='i', delimiter=",")
    end = time.time() - start
    print("Time to load training data: " + str(end) + " seconds")
    print("Creating clusters based on "+ inputFile + " and " + colorFile + " for logistic colorization...")
    numberOfCluster = int(input("Input the number of color clusters: "))
    numberofgrayscalecolor = int(input("Input the number of grayscale clusters: "))
    start = time.time()
    clusterColors, clusterGrayScales, modelcolor, modelgrayscale, numberOfCluster, numberofgrayscalecolor = clustercolorandgrayscale(inputData,rgbData, numberOfCluster,numberofgrayscalecolor)
    end = time.time() - start
    print("Time to create "+str(numberOfCluster)+" color clusters and "+str(numberofgrayscalecolor)+" grayscale clusters: " + str(end)+ " seconds")
    option = input("Create training data (T), test using data.csv (D), test usind maui - input1.csv (M) , colorize your own (C), or exit program (E)? ")
    while (option != 'E' or option != 'e'):
        if(option == 'C' or option == 'c'):
             method = input("Hard-coded (H), neural network (N) or logistic regression (L)? ")
             in_path = input("Default input path (Y or N)? ")
             if in_path == 'Y':
                 pathname = 'images/*.jpg'
                 for filename in glob.glob(pathname):
                     photo_name = os.path.basename(filename)
                     photoName = os.path.splitext(os.path.basename(photo_name))[0]
                     #print(photo_name)
                     im = Image.open(filename)
                     im = im.resize((641,361))
                     print("Predicting colors based on "+photo_name+"...")
                     start = time.time()
                     gray_image = grayscale(im)
                     gray_matrix = getGrayscaleMatrix(gray_image)
                     expanded_matrix = expandMatrix(gray_matrix, photo_name)
                     if(method == 'N' or method == 'n'):
                         resultofprediction = ClassificationWithNN(inputFile,colorFile,inputData, rgbData, expanded_matrix)
                     else:
                         #resultofprediction = ClassificationWithLogisticRegression(clusterGrayScales[:, 9], clusterColors[:, 3], expanded_matrix, modelgrayscale,numberOfCluster,numberofgrayscalecolor)
                         resultofprediction = ClassificationWithLogisticRegression2(clusterGrayScales[:, 0:9], clusterColors[:, 3], expanded_matrix, numberOfCluster, numberofgrayscalecolor)
                         labelValue, labelIndex = np.unique(clusterColors[:,3],return_index="True")
                         colorlabelIndex = clusterColors[labelIndex,3:7]
                         plotRGB(colorlabelIndex[:,1:4])
                     resultofprediction = resultofprediction.reshape(-1, 1)
                     end = time.time() - start
                     print("Time to predict colors: " + str(end) + " seconds")
                     gray = 'C'
                     print("Colorizing " + photo_name + "...")
                     start = time.time()
                     if (method == 'N' or method == 'n'):
                         new_im = generateResultsWithNN(im,gray,resultofprediction)
                     else:
                         new_im = generateResults(im, gray, colorlabelIndex, resultofprediction, numberOfCluster,numberofgrayscalecolor)
                     if(method == 'H' or method == 'h'):
                         image_file = 'results/hardcoded_result_' + photo_name
                     elif(method == 'L' or method == 'l'):
                         image_file = 'results/logistic_result_' + photoName + '.jpg'
                     else:
                         image_file = 'results/neural_result_' + photoName + '.jpg'
                     new_im.save(image_file)
                     end = time.time() - start
                     print("Time to colorize image(s): " + str(end) + " seconds")
                     print("Result saved to " + image_file + ".")
             else:
                 pathname = input("Enter path to image file: ")
                 gray = input("Is the input image grayscale (G) or in color (C)? ")
                 start = time.time()
                 image = Image.open(pathname)
                 image = image.resize((641,361))
                 photo_name = os.path.basename(pathname)
                 photoName = os.path.splitext(os.path.basename(photo_name))[0]
                 print("Predicting colors based on "+photo_name+"...")
                 start = time.time()
                 gray_image = grayscale(image)
                 gray_matrix = getGrayscaleMatrix(gray_image)
                 expanded_matrix = expandMatrix(gray_matrix, photo_name)
                 if(method == 'N' or method == 'n'):
                     resultofprediction = ClassificationWithNN(inputFile,colorFile,inputData, rgbData, expanded_matrix)
                 else:
                     #resultofprediction = ClassificationWithLogisticRegression(clusterGrayScales[:, 9], clusterColors[:, 3], expanded_matrix, modelgrayscale,numberOfCluster,numberofgrayscalecolor)
                     resultofprediction = ClassificationWithLogisticRegression2(clusterGrayScales[:, 0:9], clusterColors[:, 3], expanded_matrix, numberOfCluster, numberofgrayscalecolor)
                     labelValue, labelIndex = np.unique(clusterColors[:,3],return_index="True")
                     colorlabelIndex = clusterColors[labelIndex,3:7]
                     plotRGB(colorlabelIndex[:,1:4])
                 resultofprediction = resultofprediction.reshape(-1, 1)
                 end = time.time() - start
                 print("Time to predict colors: " + str(end) + " seconds")
                 print("Colorizing " + photo_name + "...")
                 start = time.time()
                 if(method == 'N' or method == 'n'):
                     new_im = generateResultsWithNN(image,gray,resultofprediction)
                 else:
                   new_im = generateResults(image, gray, colorlabelIndex, resultofprediction, numberOfCluster,numberofgrayscalecolor)  
                 if(method == 'H' or method == 'h'):
                     image_file = 'results/hardcoded_result_' + photo_name
                 elif(method == 'L' or method == 'l'):
                         image_file = 'results/logistic_result_' + photoName + '.jpg'
                 else:
                     image_file = 'results/neural_result_' + photoName +'.jpg'
                 new_im.save(image_file)
                 end = time.time() - start
                 print("Time to colorize image(s): " + str(end) + " seconds")
                 print("Result saved to " + image_file + ".")
                 visualize = input("Visualize here (Y or N)? ")
                 if (visualize == 'Y' or visualize == 'y'):
                    imgplot = plt.imshow(new_im, aspect = 'equal')
                    plt.show()
        elif (option == 'D' or option == 'd'):
            method = input(
                "Hard-coded (H), neural network (N) or logistic regression (L) or Gray Scale Cluster to Color Cluster Classificaiton (CC)? ")
            colors = np.genfromtxt("color.csv", dtype='i', delimiter=",")
            dataImage = np.genfromtxt("data.csv", dtype='i', delimiter=",")

            image, grayMatrix = visualizeDataFile(dataImage)
            start = time.time()
            gray = 'G'
            print("Predicting colors based on data.csv...")
            start = time.time()
            newdata = np.genfromtxt("data.csv", dtype='i', delimiter=",")

            if (method == 'N' or method == 'n'):
                resultofprediction = ClassificationWithNN(inputFile, colorFile, inputData, rgbData, newdata)
            elif (method == 'L' or method == 'l'):
                resultofprediction = ClassificationWithLogisticRegression2(clusterGrayScales[:, 0:9],
                                                                           clusterColors[:, 3], newdata,
                                                                           numberOfCluster, numberofgrayscalecolor)
                labelValue, labelIndex = np.unique(clusterColors[:, 3], return_index="True")
                colorlabelIndex = clusterColors[labelIndex, 3:7]

            else:
                resultofprediction = ClassificationWithLogisticRegression(clusterGrayScales[:, 9], clusterColors[:, 3],
                                                                          newdata, modelgrayscale, numberOfCluster,
                                                                          numberofgrayscalecolor)
                labelValue, labelIndex = np.unique(clusterColors[:, 3], return_index="True")
                colorlabelIndex = clusterColors[labelIndex, 3:7]

            resultofprediction = resultofprediction.reshape(-1, 1)
            end = time.time() - start
            print("Time to predict colors: " + str(end) + " seconds")
            print("Colorizing data.csv image...")
            if(method == 'N' or method == 'n'):
                result = generateResultsWithNN(image,gray,resultofprediction)
            else:
                result = generateResults(image, gray, colorlabelIndex, resultofprediction, numberOfCluster,numberofgrayscalecolor)
            if(method == 'H' or method == 'h'):
                result_file = 'results/hardcoded_result_data.jpg'
            elif(method == 'L' or method == 'l'):
                result_file = 'results/logistic_result_data.jpg'
            elif (method == 'CC' or method == 'c'):
                result_file = 'results/graycluster_colorcluster_result_data.jpg'
            else:
                result_file = 'results/neural_result_data.jpg'
            result.save(result_file)
            end = time.time() - start
            print("Time to colorize data.csv image: " + str(end) + " seconds")
            print("Result saved to " + result_file + ".")
            visualize = input("Visualize here (Y or N)? ")
            if (visualize == 'Y' or visualize == 'y'):
                imgplot = plt.imshow(result, aspect = 'equal')
                plt.show()
        elif (option == 'M' or option == 'm'):
            method = input(
                "Hard-coded (H), neural network (N) or logistic regression (L) or Gray Scale Cluster to Color Cluster Classificaiton (CC)? ")

            colors = np.genfromtxt("color1.csv", dtype='i', delimiter=",")
            dataImage = np.genfromtxt("input1.csv", dtype='i', delimiter=",")
            image, grayMatrix = visualizeInput1(dataImage)
            start = time.time()
            gray = 'G'
            print("Predicting colors based on input1.csv...")
            start = time.time()

            newdata = np.genfromtxt("input1.csv", dtype='i', delimiter=",")

            if (method == 'N' or method == 'n'):
                resultofprediction = ClassificationWithNN(inputFile, colorFile, inputData, rgbData, newdata)
            elif (method == 'L' or method == 'l'):
                resultofprediction = ClassificationWithLogisticRegression2(clusterGrayScales[:, 0:9],
                                                                           clusterColors[:, 3], newdata,
                                                                           numberOfCluster, numberofgrayscalecolor)
                labelValue, labelIndex = np.unique(clusterColors[:, 3], return_index="True")
                colorlabelIndex = clusterColors[labelIndex, 3:7]

            else:
                resultofprediction = ClassificationWithLogisticRegression(clusterGrayScales[:, 9], clusterColors[:, 3],
                                                                          newdata, modelgrayscale, numberOfCluster,
                                                                          numberofgrayscalecolor)
                labelValue, labelIndex = np.unique(clusterColors[:, 3], return_index="True")
                colorlabelIndex = clusterColors[labelIndex, 3:7]

            resultofprediction = resultofprediction.reshape(-1, 1)
            end = time.time() - start
            print("Time to predict colors: " + str(end) + " seconds")
            print("Colorizing data.csv image...")
            if(method == 'N' or method == 'n'):
                result = generateResultsWithNN(image,gray,resultofprediction)
            else:
                result = generateResults(image, gray, colorlabelIndex, resultofprediction, numberOfCluster,numberofgrayscalecolor)
            if(method == 'H' or method == 'h'):
                result_file = 'results/hardcoded_result_data.jpg'
            elif(method == 'L' or method == 'l'):
                result_file = 'results/logistic_result_data.jpg'
            elif (method == 'CC' or method == 'c'):
                result_file = 'results/graycluster_colorcluster_result_data.jpg'
            else:
                result_file = 'results/neural_result_data.jpg'
            result.save(result_file)
            end = time.time() - start
            print("Time to colorize data.csv image: " + str(end) + " seconds")
            print("Result saved to " + result_file + ".")
            visualize = input("Visualize here (Y or N)? ")
            if (visualize == 'Y' or visualize == 'y'):
                imgplot = plt.imshow(result, aspect = 'equal')
                plt.show()
        elif (option == 'T' or option == 't'):
            filename = input("Enter path of which image to create training data from: ")
            photo_name = os.path.basename(filename)
            image = Image.open(filename)
            gray_image = grayscale(image)
            grayMatrix = getGrayscaleMatrix(gray_image)
            start = time.time()
            createTrainingData(image,grayMatrix,photo_name)
            end = time.time() - start
            print("Time to get training data for "+photo_name+": "+str(end)+" seconds")
        else:
            break
        option = input("Create training data (T), test using data.csv (D), colorize your own (C), or exit program (E)? ")