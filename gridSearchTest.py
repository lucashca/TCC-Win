from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,max_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from multiprocessing import Process, Lock ,Manager
from mpl_toolkits.mplot3d import Axes3D  

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getMin(dataSet,column):
    minimo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d < minimo):
            minimo = d
    return minimo    

def getMax(dataSet,column):
    maximo = dataSet[0][column]
    for i in range(len(dataSet)):
        d = dataSet[i][column]
        if(d > maximo):
            maximo = d
    return maximo    

def normalizeColumn(dataSet,column):
    minimo = getMin(dataSet,column)
    maximo = getMax(dataSet,column)

    for i in range(len(dataSet)):
        d = dataSet[i][column]
        dataSet[i][column] = (d - minimo)/(maximo - minimo)


def convertCSV(arr):
    #Corrige o dataset, convertendo strig para float e também coverte a unidade dos nutriente de mg/l para cmolc/dm3
    for data in arr:
        data[0] = float(data[0])
        data[1] = float(data[1])
        data[2] = round(float(data[2])/200.4,2) #Ca
        data[3] = round(float(data[3])/121.56,2) #Mg
        data[4] = round(float(data[4])/230,2) #Na
        data[5] = round(float(data[5])/391,2) #K
        data[6] = round(float(data[6])/506.47,2) #Cl
        
    return arr



'''
 grid = GridSearchCV(model, para_grids, scoring=metricas, verbose=0, refit='r2', return_train_score=False,cv=3,iid=True,n_jobs = -1)
        grid.fit(x_train, y_train)
        bestScore = grid.best_score_
        print("Best Score %.5f"%bestScore)
        if(bestScore< 0):
            bestScore = math.sqrt(bestScore*-1) 
            print("Squared Best Score %.5f"%bestScore)
'''



def findBestModelByMedia(regressor,dataSet,tname,shared_list,k,yColunm):
  
    melhorMediaR2 = 0
    melhorMediaMSE = 0
    bestConfigR2 = []
    bestConfigMSE = []
    totalTestData = len(k)

    cont = 0
    for params,model in regressor:
        mediaR2 = 0
        mediaMSE = 0

        for i in k:
            x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=i)
            
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
        
            mediaR2 += r2
            mediaMSE += mse
        
        mediaMSE = mediaMSE /totalTestData
        mediaR2 = mediaR2 /totalTestData

        if(cont == 0):
            melhorMediaMSE = mediaMSE
            melhorMediaR2 = mediaR2
            bestConfigMSE = params
            bestConfigR2 = params
            
        else:
            if(mediaMSE < melhorMediaMSE):
                melhorMediaMSE = mediaMSE
                bestConfigMSE = params
                
            if(mediaR2 > melhorMediaR2): 
                melhorMediaR2 = mediaR2
                bestConfigR2 = params

        cont +=1
        print("%s : %d/%d - Melhor Media MSE: %.3f Melhor Media R2: %.3f"%(tname,cont,len(regressor),melhorMediaMSE,melhorMediaR2))
    print("********* %s ***********"%(tname))
    print("Melhor média MSE %.5f"%(melhorMediaMSE))
    print("Melhor Configuração MSE ")
    print(bestConfigMSE)
    print("Melhor média R2 %.5f"%(melhorMediaR2))
    print("Melhor Configuração R2 ")
    print(bestConfigR2)
    print("@@@@@@@@@ %s @@@@@@@@@@"%(tname))
    shared_list.append([melhorMediaMSE,melhorMediaR2,bestConfigMSE,bestConfigR2])
    printList(shared_list)


def findBestModelByBestRes(regressor,dataSet,tname,shared_list,k,yColunm):
  
    melhorR2 = 0
    melhorMSE = 0
    bestConfigR2 = []
    bestConfigMSE = []
    faixa = k

    cont = 0
    for params,model in regressor:
        
        for i in faixa:
            x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=i)
            
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            r2 = r2_score(y_test,y_pred)
            mse = mean_squared_error(y_test,y_pred)
    
            if(cont == 0):
                melhorMSE = mse
                melhorR2 = r2
                bestConfigMSE = params
                bestConfigMSE.append(i)
                bestConfigR2 = params
                bestConfigR2.append(i)

            else:
                if(mse < melhorMSE):
                    melhorMSE = mse
                    bestConfigMSE = params
                    bestConfigMSE.append(i)
                    
                if(r2 > melhorR2): 
                    melhorR2 = r2
                    bestConfigR2 = params
                    bestConfigR2.append(i)

        cont +=1
        print("%s : %d/%d - Melhor MSE: %.3f Melhor R2: %.3f"%(tname,cont,len(regressor),melhorMSE,melhorR2))
    print("********* %s ***********"%(tname))
    print("Melhor MSE %.5f"%(melhorMSE))
    print("Melhor Configuração MSE ")
    print(bestConfigMSE)
    print("Melhor média R2 %.5f"%(melhorR2))
    print("Melhor Configuração R2 ")
    print(bestConfigR2)
    print("@@@@@@@@@ %s @@@@@@@@@@"%(tname))
    shared_list.append([melhorMSE,melhorR2,bestConfigMSE,bestConfigR2])
    printList(shared_list)





def printList(shared_list):
    for l in shared_list:
        print(l)

def testConfiguration(d,s,m,b,r,i):
    global yColunm
    model = RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=r)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=i)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    return mse,r2

def getMeanByConfiguration(cont,d,s,m,b,r):
    r2 = 0
    mse = 0
    for i in range(cont):
        a,b = testConfiguration(d,s,m,b,r,i)
        mse +=a
        r2 +=b
    print("Media MSE: %.5f"%(mse/cont))
    print("Media R2: %.5f"%(r2/cont))




def splitArray(totalSize,workers):
    work = []
    if(totalSize%workers == 0):
        carga = int(totalSize/workers)
    else:
        carga = int(totalSize/workers) + 1
    for i in range(workers):
        inicio = i*carga
        fim = i*carga + carga
        if fim > totalSize:
            fim = totalSize
        r = range(inicio,fim)
        work.append(r)
    
    return work


def createProcessByBestRes(qtdWorker,regressor,dataSet,k,yColunm):
    totalSize = len(regressor)
    work = splitArray(totalSize,qtdWorker)
    shared_list = Manager().list([])
    process = []
    
    for i in range(qtdWorker):
       
        start = work[i][0]
        end = work[i][-1] + 1
        
        #t = Process(target=findBestModelByMedia,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        t = Process(target=findBestModelByBestRes,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        
        process.append(t)
        process[i].start()
    for i in range(qtdWorker):
        process[i].join()        


def createProcessByBestMean(qtdWorker,regressor,dataSet,k,yColunm):
    totalSize = len(regressor)
    work = splitArray(totalSize,qtdWorker)
    shared_list = Manager().list([])
    process = []
    
    for i in range(qtdWorker):
       
        start = work[i][0]
        end = work[i][-1] + 1
        
        t = Process(target=findBestModelByMedia,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        #t = Process(target=findBestModelByBestRes,args=(regressor[start:end],dataSet,"Thread"+str(i),shared_list,k,yColunm))
        
        process.append(t)
        process[i].start()
    for i in range(qtdWorker):
        process[i].join()     



#csvFile = pd.read_csv("dataSetOriginal1.csv",usecols=[1,2,3,4,5,6,7])
csvFile = pd.read_csv("mainDataSet1.csv",usecols=[1,2,3,4,5,6,7])

csvFile = convertCSV(csvFile.values)
dataSet = np.array(csvFile)


model = RandomForestRegressor()
#Falta de 80 a 100 coluna 2



metricas = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']

def createRegressor(para_grids):
    regressor = []
    for d in para_grids["max_depth"]:
        for e in para_grids["n_estimators"]:
            for m in para_grids["max_features"]:
                for b in para_grids["bootstrap"]:
                    for r in para_grids["random_state"]:
                        regressor.append([[d,e,m,b,r],RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=r)])
    return regressor




def testAndPlot(d,e,m,b,r,rd,yColunm,dataSet):
    model = RandomForestRegressor(max_depth=d,n_estimators=e,max_features=m,bootstrap=b,random_state=r)
    x_train,x_test,y_train,y_test = train_test_split(dataSet[:,:2],dataSet[:,yColunm],test_size=0.3,random_state=rd)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)

    fig = plt.subplot(211)
    fig.scatter(x_test[:,0],y_test,c=y_test ,cmap="coolwarm")
    fig.set_title("Dados de Teste R2: %.5f MSE: %.5f"%(r2,mse))
    fig.set_ylabel('Ca')
    fig.set_xlabel('Latitude')

    fig = plt.subplot(212)
    fig.scatter(x_test[:,0],y_pred,c=y_pred ,cmap="coolwarm")
    fig.set_title("Dados de Pred")
    fig.set_ylabel('Ca')
    fig.set_xlabel('Latitude')

    plt.show()

d = [i for i in range(10,20)]
d.append(None)
e = range(1,100)

para_grids = {
            "max_depth"    : d,
            "n_estimators" : e,
            "max_features" : ["log2","auto"],
            "bootstrap"    : [True, False],
            "random_state" : range(1)
        }

regressor = createRegressor(para_grids)
yColunm = 2

normalizeColumn(dataSet,0)
normalizeColumn(dataSet,1)
normalizeColumn(dataSet,yColunm)

if __name__ == '__main__':
    createProcessByBestRes(4,regressor,dataSet,range(0,100),yColunm)
    #createProcessByBestMean(4,regressor,dataSet,range(64,74),yColunm)


#testAndPlot(None, 5, 'log2', False, 0, 950, 2,dataSet)
#testAndPlot(9,3,'log2',False,0,64,2,dataSet)

''' Experimento 1:
    Melhor resultado sem média
    coluna 2

    [0.01612144600417462, 0.49073156154753483, [4, 16, 'log2', False, 0, 33, 48], [4, 16, 'log2', False, 0, 33, 48]]
    [0.01297716867014392, 0.6148791341723359, [6, 89, 'log2', False, 0, 48], [7, 1, 'auto', False, 0, 87]]
    [0.012283426467016516, 0.6807662078296097, [8, 19, 'log2', True, 0, 48], [9, 3, 'log2', False, 0, 64]]
    [0.019113212192804315, 0.4492221897808114, [3, 3, 'auto', True, 0, 990], [3, 1, 'log2', True, 0, 917]]
[0.014469156505887683, 0.7005457304010185, [8, 3, 'log2', False, 0, 930], [7, 1, 'log2', True, 0, 950, 950]]
[0.016403440980570134, 0.553050536238382, [5, 3, 'auto', True, 0, 990], [5, 1, 'log2', True, 0, 917]]
[0.012256050922095392, 0.7495031992705566, [None, 5, 'log2', False, 0, 950, 950], [None, 5, 'log2', False, 0, 950, 950]]
[0.01671618967066996, 0.454347582044505, [3, 4, 'log2', True, 0, 729], [3, 1, 'auto', True, 0, 519]]
[0.00945707666456554, 0.6645085207900854, [8, 3, 'auto', False, 0, 295], [7, 1, 'auto', False, 0, 179, 295]]
[0.015085046469587395, 0.5583756408726119, [5, 4, 'log2', False, 0, 729], [5, 8, 'auto', True, 0, 662]]
[0.009473278344713117, 0.654271196173569, [8, 5, 'auto', False, 0, 102, 160, 169, 169, 179, 179, 237, 295], [9, 3, 'auto', False, 0, 179]]

    [0.01671618967066996, 0.454347582044505, [3, 4, 'log2', True, 0, 729], [3, 1, 'auto', True, 0, 519]]
[0.00945707666456554, 0.6645085207900854, [8, 3, 'auto', False, 0, 295], [7, 1, 'auto', False, 0, 179, 295]]
[0.015085046469587395, 0.5583756408726119, [5, 4, 'log2', False, 0, 729], [5, 8, 'auto', True, 0, 662]]

    [0.012888415118027007, 0.6065124470802861, [90, 8, 'log2', True, 0, 62], [90, 50, 'log2', False, 0, 33]]
    [0.012888415118027007, 0.6065124470802861, [93, 8, 'log2', True, 0, 62], [93, 50, 'log2', False, 0, 33]]
    [0.012888415118027007, 0.6065124470802861, [96, 8, 'log2', True, 0, 62], [96, 50, 'log2', False, 0, 33]]
    [0.012888415118027007, 0.6065124470802861, [99, 8, 'log2', True, 0, 62], [98, 50, 'log2', False, 0, 33]]




coluna 3

[0.0038540164051903796, 0.6011196688597428, [3, 1, 'log2', True, 0, 998], [3, 1, 'log2', False, 0, 976]]
[0.003968774499277207, 0.5464588739613511, [6, 5, 'auto', True, 0, 998], [6, 1, 'auto', False, 0, 968, 976, 998]]
[0.0034914650172396464, 0.5903406666121734, [4, 1, 'log2', False, 0, 985, 998], [5, 1, 'log2', False, 0, 976]]
[0.003743979757781881, 0.6542838239956197, [9, 6, 'log2', False, 0, 970], [9, 3, 'log2', False, 0, 950]]



    Melhor resultado média 

    [0.03648712538957544, 0.20275187341719797, [3, 71, 'log2', True, 0], [3, 50, 'log2', True, 0]]
    [0.0304587752263902, 0.3250611469002779, [6, 50, 'log2', True, 0], [6, 50, 'log2', True, 0]]
    [0.029203767643761102, 0.352135503924569, [8, 33, 'log2', True, 0], [8, 99, 'log2', True, 0]]
    [0.028854493648590385, 0.3600698659991327, [10, 69, 'log2', True, 0], [10, 69, 'log2', True, 0]]
'''


'''
# Melhor resultado sem média
yColunm = 2
[0.019315787984205093, 0.463564570239994, [3, 2, 'log2', False, 0, 48], [3, 1, 'log2', False, 0, 33]]
[0.014487208636780467, 0.6148791341723359, [6, 5, 'log2', False, 0, 48], [7, 1, 'auto', False, 0, 87]]
[0.014698020407797977, 0.6075293174186629, [5, 9, 'log2', False, 0, 48], [5, 7, 'auto', True, 0, 64]]
[0.012887263854384084, 0.6807662078296097, [None, 8, 'log2', True, 0, 62], [9, 3, 'log2', False, 0, 64]]

yColunm = 3
[0.0037491328399349936, 0.6878704611580888, [3, 1, 'log2', False, 0, 33, 33], [3, 1, 'log2', False, 0, 33, 33]]
[0.0038186306997247295, 0.6820845005499163, [8, 4, 'log2', False, 0, 33, 33], [8, 4, 'log2', False, 0, 33, 33]]
[0.0042481934009474176, 0.646321775258313, [5, 5, 'log2', False, 0, 33, 33], [5, 5, 'log2', False, 0, 33, 33]]
[0.004272415173115143, 0.6443052207910733, [8, 5, 'sqrt', False, 0, 33, 33], [8, 5, 'sqrt', False, 0, 33, 33]]

yColunm = 4
[0.011930993970140804, 0.40321938707451155, [2, 2, 'log2', True, 0, 84, 84], [2, 2, 'log2', True, 0, 84, 84]]
[0.00812742417774943, 0.582429314072503, [6, 4, 'auto', True, 0, 72, 72], [6, 4, 'auto', True, 0, 72, 72]]
[0.008839769959769974, 0.5458304224297825, [5, 5, 'auto', True, 0, 72, 72], [5, 5, 'auto', True, 0, 72, 72]]
[0.009826199177730203, 0.5215989378177964, [8, 8, 'auto', True, 0, 72, 72], [None, 9, 'auto', True, 0, 16]]


yColunm = 5
[0.0022952657260199775, 0.3862732836938054, [3, 5, 'log2', True, 0, 17, 84], [3, 2, 'auto', True, 0, 20]]
[0.0016015085069081023, 0.5506345213476707, [6, 6, 'auto', True, 0, 64, 64], [7, 4, 'log2', True, 0, 99]]
[0.001421000225512235, 0.49899536910531683, [5, 5, 'log2', False, 0, 64, 64], [5, 5, 'log2', False, 0, 64, 64]]
[0.001609973656202759, 0.4371876745236217, [None, 4, 'auto', True, 0, 64], [9, 1, 'log2', True, 0, 73]]

# Melhores média
yColunm = 2

[0.026100574708490875, 0.33427725560444116, [6, 8, 'log2', True, 0], [6, 8, 'log2', True, 0]]
[0.02749755588430262, 0.3044808358277409, [26, 9, 'log2', True, 0], [26, 9, 'log2', True, 0]]
[0.02749755588430262, 0.3044808358277409, [51, 9, 'log2', True, 0], [51, 9, 'log2', True, 0]]
[0.02749755588430262, 0.3044808358277409, [76, 9, 'log2', True, 0], [76, 9, 'log2', True, 0]]

yColunm = 3
'''
