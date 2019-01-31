import xlrd
from datetime import datetime
import matplotlib as mplot
import matplotlib.pyplot as plt
import operator
import math
from sklearn import preprocessing
import numpy as np
from sklearn import pipeline
import scipy.stats

class Record(object):
    def __init__(self, date, close, volume, open, high, low):
        self.date = date
        self.close = close
        self.volume = volume
        self.open = open
        self.high = high
        self.low = low
    
#    @property
#    def date(self):
#        return self.date
#
#    @date.setter
#    def date(self, value):
#        # you likely want to it here
#        self.date = value
#    
#    @property
#    def close(self):
#        return self.close
#
#    @close.setter
#    def close(self, value):
#        # you likely want to it here
#        self.close = value
#    
#    @property
#    def volume(self):
#        return self.volume
#
#    @volume.setter
#    def volume(self, value):
#        # you likely want to it here
#        self.volume = value
#    
#    @property
#    def open(self):
#        return self.open
#
#    @open.setter
#    def open(self, value):
#        # you likely want to it here
#        self.open = value
#    
#    @property
#    def high(self):
#        return self.high
#
#    @high.setter
#    def high(self, value):
#        # you likely want to it here
#        self.high = value
#
#    @property
#    def low(self):
#        return self.low
#
#    @low.setter
#    def low(self, value):
#        # you likely want to it here
#        self.low = value
#

    def __str__(self):
        return("Current Record:\n"
            "  date = {0}\n"
            "  closing_price = {1}\n"
            "  volume = {2}\n"
            "  openning_price = {3}\n"
            "  highest_price = {4} \n"
            "  lowest_price = {5}"
            .format(self.date, self.close, self.volume,
                self.open, self.high, self.low))


def pred1(cl, op, vol, high, low):

    return

def pred2(cl, op, vol, high, low):

    return

def pred3(cl, op, vol, high, low):

    return

def trainingEM(sampleL, centroidL):
    #lst = list(map(operator.sub, closeL, openL))
    mean = sum(sampleL)/len(sampleL)
    errLst = [(a - mean)**2 for a in sampleL]
    sumErr = sum(errLst)
    std = math.sqrt(sumErr/len(errLst))

        
    for i in range(1, 100):
        prLst = [0 for x in range(len(centroidL))]
        valLst = [0 for x in rnage(len(centroidL))]

        for sample in sampleL:
            a = [(1 / math.sqrt(2 * math.pi * (std**2)) * math.exp(0 - (((centroid - sample)**2) / (2 * (std**2))))) for centroid in centroidL]
            
            prhold = []
            valhold = []

            for prn, prp, prt in zip(a, prLst, valLst):
                prpn = prp + prn
                prtn = prt + prn*sample
                prhold.append(prpn)
                valhold.append(prtn)

            prLst = prhold
            valLst = valhold
        
        ncentL = []
        for prcent, valcent in zip(prLst, valLst):
            ncentL.append(valcent / prcent)

        centroidL = ncentL 

    return centroidL


loc = "C:/Users/User/Desktop/592/HistoricalQuotes.xlsx"
data = xlrd.open_workbook(loc)


sheet = data.sheet_by_index(0)

#print(sheet.nrows)

items = []
for row in range(1, sheet.nrows):
    params = []
    for col in  range(sheet.ncols):
        cell = sheet.cell(row,col).value
        params.append(cell)
    item = Record(*params)
    items.append(item)

dateL =[]
volumeL = []
closeL =[]
openL = []
highL = []
lowL = []
trendL = []
ttrendL = []
		
for item in items:
#    item.date = datetime.datetime(1900, 1, 1) + datetime.timedelta(days=item[0]);
    #print(item)
    #print(item.close)
    #print(item.open)
    try:
        dt = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(item.date))
        dto = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(item.date)).replace(hour=9, minute=30)
        dtc = datetime.fromordinal(datetime(1900, 1, 1).toordinal() + int(item.date)).replace(hour=16, minute=0)

        trendL.append(float(item.open))
        trendL.append(float(item.close))
        ttrendL.append(dto)
        ttrendL.append(dtc)
        
        #print(dt)
        dateL.append(dt)
        #print(item.date)
        volumeL.append(float(item.volume))
        closeL.append(float(item.close))
        openL.append(float(item.open))
        highL.append(float(item.high))
        lowL.append(float(item.low))
    except ValueError:
        print("reach null")
        # dateL.remove("")
#print(list(map(operator.sub, closeL, openL)))
lst = list(map(operator.sub, closeL, openL))
mean = sum(lst)/len(lst)
errLst = [(a - mean)**2 for a in lst]
sumErr = sum(errLst)
std = math.sqrt(sumErr/len(errLst))

new_lst, new_dateL = [a for a in lst if a <= std*3 + mean and a >= mean - std*3], [b for a,b in zip(lst, dateL) if  a <= std*3 + mean and a >= mean - std*3]

outlier_lst, outlier_dateL = [a for a in lst if a > std*3 + mean or a < mean - std*3], [b for a,b in zip(lst, dateL) if  a > std*3 + mean or a < mean - std*3]


#print(len(new_lst) == len(lst))
#print(len(new_lst) == len(new_dateL))
plt.figure(1)
plt.plot(new_dateL, new_lst)

plt.figure(2)
plt.plot(dateL, lst)

plt.figure(3)
plt.plot(dateL,openL)



plt.figure(4)
plt.plot(ttrendL, trendL)
statesL = [0,0,0,0,0,0,0,0,0,0]
stvL = [0,0,0,0,0,0,0,0,0,0]
stcL = [0,0,0,0,0,0,0,0,0,0]

hmmL = []
ratioL = []
dtL = []
emL = []
hurstL = []
prev = 0
end = 0
trainingL = []
tdL = []
mapper = []
perr = []
hd = []
fact = 1
coef = []

emission = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
transition = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]

prevstate = -1


intercept = 0
counter = 0
h = 0
training = 1
#hd.append(openL[len(openL)-1])
hd.append(openL[len(openL)-1])
trainingL.append(openL[len(openL)-1])
emL.append(openL[len(openL)-1])
hmmL.append(openL[len(openL)-1])
tdL.append((dateL[len(dateL)-1] - min(dateL)).days)
hurstL.append(0.99)
pr = len(openL)-1;
nr = 0;
for j in range(len(openL)-2, -1, -1):
    if j==0:
        print('reached this')
    hold = 365
    inc = 0
    show = 0
    
    if dateL[j].month == 1 and dateL[j + 1].month == 12:
        if dateL[j].year % 4 == 0:
            hold = 366
        fact *= 1.02
        h += 1
        
        #print(h)
        #print(fact)
    if j == 0 or (dateL[j - 1].month == 1 and dateL[j].month == 12):
        show = 1
        inc = 1

    
    #if len(openL) < prev + hold:
    #
    #    hold = len(openL)-prev
    #
    #end = prev + hold
    #myL = openL[prev:end]
    #prev = end
    

    #for i in range(hold-1):
    #print(dateL[j])
    nom = j + 1 if j <= len(openL)-2 else len(openL)-1
    sampleL = openL[nom:j:-1]
    mean = sum(sampleL)/len(sampleL)
    #print(mean)
    trainingL.append(mean)
    tdL.append((dateL[j] - min(dateL)).days)
    #print((dateL[j] - min(dateL)).days)
    #print(len(trainingL))
    #print(h)
    #coef *= 1.03
    if h > 1:
        #if training == 1:
        #    training = 0
        #    hd.append(openL[j])
        v = 0
        
        for cfi in range(len(coef[len(coef)-1])):
            v += coef[len(coef)-1][cfi] * (((dateL[j] - min(dateL)).days) ** (len(coef[len(coef)-1])-cfi-1)) - coef[len(coef)-1][cfi] * (((dateL[j+1] - min(dateL)).days) ** (len(coef[len(coef)-1])-cfi-1))
        
        #print(openL[j+1] - closeL[j+1])
        
        v += openL[j+1]
        hd.append(v)
        
        mapper.append(openL[j] - v)
        perr.append((openL[j] - v)/openL[j])

        #nr = j
        yhat = np.array(openL[j+100:j-1:-1])

        

        lagg = range(2,20)
        tau = [np.sqrt(np.std(np.subtract(yhat[lag:], yhat[:-lag]))) for lag in lagg]

        lagCoe = np.polyfit(np.log(lagg), np.log(tau), 1)

        hurstCoe = lagCoe[0]*2
        hurstL.append(hurstCoe)
        #print('hurst coefficient: ')
        #print(hurstCoe)
        #print(dateL[j])
        #print()
        #pr = j

        dtL.append((dateL[j]-dateL[j+1]).days)

        diff = (openL[j+1]-openL[j+2])/openL[j+2]/((dateL[j+1]-dateL[j+2]).days)

        orgidx = min(range(len(statesL)), key=lambda idx: abs(stvL[idx] - diff))

        maxEm = transition[orgidx].index(max(transition[orgidx]))

        idxhm = emission[maxEm].index(max(emission[maxEm]))


        
        hmmpred_org = openL[j+1] + openL[j+1] * statesL[idxhm] / stcL[idxhm]*((dateL[j]-dateL[j+1]).days)  
        
        hmmPred = openL[j+1] + openL[j+1] * statesL[stcL.index(max(stcL))]/max(stcL)*((dateL[j]-dateL[j+1]).days) - statesL[stcL.index(max(stcL))]/max(stcL)*((dateL[j]-dateL[j+1]).days)* len(openL)*0

        
            
        emL.append(hmmPred)
        hmmL.append(hmmpred_org)

        rat = (openL[j]-openL[j+1])/openL[j+1]/((dateL[j]-dateL[j+1]).days)

        for drt in range((dateL[j]-dateL[j+1]).days):

            ratioL.append(rat)
            
        stidx = min(range(len(statesL)), key=lambda idx: abs(stvL[idx]-(rat)))
        #print(stidx)
        statesL[stidx] += rat*((dateL[j]-dateL[j+1]).days)

        stcL[stidx] += 1*((dateL[j]-dateL[j+1]).days)
        emission[maxEm][stidx] += 1

        if abs(openL[j] - v)>std*3:
            print(highL[j+1] - lowL[j+1])
            print(coef[len(coef)-1][0])
            #print(v)
            #print(openL[j-1])
            # recalculate coefficient and intercept using trainingL
            #regParam = preprocessing.PolynomialFeatures(1)
            #piper = pipeline.Pipeline([('lin',regParam)])
            #piper.fit(npy.array(range(1, len(trainingL)+1),trainingL).reshape(1,-1))
            #regParam.fit_transform(npy.array(trainingL).reshape(-1,1), range(1, len(trainingL)+1))

           
            arrY = np.array(trainingL[len(openL) - j-1 -35: len(openL) - j-1])


            arrX = np.array(tdL[len(openL) - j-1 -35: len(openL) - j-1])
           # nr = j
           # yhat = np.array(openL[pr:nr:-1])

            

           # lagg = range(2,20)
           # tau = [np.sqrt(np.std(np.subtract(yhat[lag:], yhat[:-lag]))) for lag in lagg]

           # lagCoe = np.polyfit(np.log(lagg), np.log(tau), 1)

           # hurstCoe = lagCoe[0]*2
           # print('hurst coefficient: ')
           # print(hurstCoe)
           # print(dateL[j])
           # print()
           # pr = j
            #arrX = np.array(range(1,len(trainingL)+1))
            #arrY = np.array(trainingL)
            
            valh = []
            resh = []
            val0, res0, _, _, _ = np.polyfit(arrX, arrY, 1, full = True)
            valh.append(val0)
            resh.append(res0)
            val = val0
            res = res0

            print('res: ' + str(res))

            for holdv in range(1,2):
                #print(holdv+1)
                val0, res0, _, _, _ = np.polyfit(arrX, arrY, holdv+1, full = True)
                valh.append(val0)
                resh.append(res0)
                fv = (resh[len(resh)-2] - resh[len(resh)-1])/(len(resh)-1 - len(resh)+2)/(resh[len(resh)-1]/(len(arrX)-len(resh)+1))
                if fv > scipy.stats.f.ppf(1-0.05, holdv + 1, holdv):
                    val = val0
                    res = res0



            #val = valh[0]
            #res = resh[0]
            std = math.sqrt(res/len(arrX))
            #print('std:' + str(std))
            coef.append(val)
            #print(coef)
            #print(intercept)


            
            



            #implement hurst exponent for short-term changes
            counter+=1
    else:
        hurstL.append(0.5)
        hd.append(openL[j])
        dtL.append((dateL[j]-dateL[j+1]).days)
        for i in range((dateL[j] - dateL[j+1]).days):
            ratioL.append((openL[j]-openL[j+1])/openL[j]/((dateL[j] - dateL[j+1]).days))

        emL.append(openL[j])
        hmmL.append(openL[j])

    if h == 1 and inc == 1:
        # calculate linear coefficient and intercept using trainingL
        #regParam = preprocessing.PolynomialFeatures(1)
        #regParam.fit(npy.array(trainingL).reshape(-1,1), range(1, len(trainingL)+1))
        arrX = np.array(tdL)
        arrY = np.array(trainingL)
        valh = []
        resh = []
        val0, res0, _, _, _ = np.polyfit(arrX, arrY, 1, full = True)
        valh.append(val0)
        resh.append(res0)
        val = val0
        res = res0

        for holdv in range(1,2):
            #print(holdv+1)
            val0, res0, _, _, _ = np.polyfit(arrX, arrY, holdv+1, full = True)
            valh.append(val0)
            resh.append(res0)
            fv = (resh[len(resh)-2] - resh[len(resh)-1])/(len(resh)-1 - len(resh)+2)/(resh[len(resh)-1]/(len(arrX)-len(resh)+1))
            if fv > scipy.stats.f.ppf(1-0.05, holdv + 1, holdv):
                val = val0
                res = res0

        std = math.sqrt(res/len(arrX))
        #print('std: ' + str(std))
        coef.append(val)
        #intercept = val[1]

        maxR = max(ratioL)
        minR = min(ratioL)
        step = (maxR - minR)/10
        cll = minR + 0
        for vidx in range(len(stvL)):
            cll+= step
            stvL[vidx] = cll - step/2

        for rt in ratioL:

            kh = rt - minR
            if prevstate == -1:
                prevstate = int(kh/step - 0.001)
            else:
                transition[prevstate][int(kh/step - 0.001)] += 1
                prevstate = int(kh/step - 0.001)

            
            stcL[int(kh/step - 0.001)] += 1
            statesL[int(kh/step - 0.001)] += kh

        prevstate = -1
        for rt in ratioL:

            kh = rt - minR
            idxh = int(kh/step - 0.001)
            if prevstate == -1:
                prevstate = idxh
            else:
                maxer = transition[prevstate].index(max(transition[prevstate]))
                emission[maxer][idxh] += 1
        
        
    if show == 1:
        print('it: ')
        print(counter/len(openL))
        print((len(lst)-len(new_lst))/len(lst))
        #print(coef)
        #print(intercept)


print('num readjust: ' + str(counter / len(openL)))

signal = 0
temp = 0
v0 = 0

state1L = []
state2L = []
state3L = []
state4L = []

HL0L = []
AN0L = []
V0L = []

HL1L = []
AN1L = []
V1L = []

HL2L = []
AN2L = []
V2L = []

HL3L = []
AN3L = []
V3L = []

stat_co_110 = 0
stat_co_010 = 0
stat_co_100 = 0
stat_co_000 = 0

stat_vd_110 = 0
stat_vd_010 = 0
stat_vd_100 = 0
stat_vd_000 = 0

vpred = 0

std = math.sqrt(sumErr/len(errLst))
for da, vo, op, cl, hi, lo in  zip(dateL, volumeL, openL, closeL, highL, lowL):
    
    if signal == 1 and op > cl:
        state1L.append(Record(da, cl, vo, op, hi, lo))
        HL0L.append(hi - lo)
        V0L.append(v0-vo)
        meanV = sum(V0L)/len(V0L)
        errLstV = [(a - meanV)**2 for a in V0L]
        sumErrV = sum(errLstV)
        std = math.sqrt(sumErrV/len(errLstV))

        if vo > 3*std + vpred or vo < vpred - 3*std:
            AN0L.append(1)
        else:
            AN0L.append(0)

    elif signal == 0 and op > cl:
        state2L.append(Record(da, cl, vo, op, hi, lo))
        HL1L.append(hi - lo)
        V1L.append(v0-vo)
        meanV = sum(V1L)/len(V1L)
        errLstV = [(a - meanV)**2 for a in V1L]
        sumErrV = sum(errLstV)
        std = math.sqrt(sumErrV/len(errLstV))

        if vo > 3*std + vpred or vo < vpred - 3*std:
            AN1L.append(1)
        else:
            AN1L.append(0)

    elif signal == 1 and op <= cl:
        state3L.append(Record(da, cl, vo, op, hi, lo))
        HL2L.append(hi - lo)
        V2L.append(v0-vo)
        meanV = sum(V2L)/len(V2L)
        errLstV = [(a - meanV)**2 for a in V2L]
        sumErrV = sum(errLstV)
        std = math.sqrt(sumErrV/len(errLstV))

        if vo > 3*std + vpred or vo < vpred - 3*std:
            AN2L.append(1)
        else:
            AN2L.append(0)

    else:
        state4L.append(Record(da, cl, vo, op, hi, lo))
        HL3L.append(hi - lo)
        V3L.append(v0-vo)
        meanV = sum(V3L)/len(V3L)
        errLstV = [(a - meanV)**2 for a in V3L]
        sumErrV = sum(errLstV)
        std = math.sqrt(sumErrV/len(errLstV))

        if vo > 3*std + vpred or vo < vpred - 3*std:
            AN3L.append(1)
        else:
            AN3L.append(0)
    
    diff = hi - lo
    clldiff = cl - lo
    clhdiff = cl - hi
    opldiff = op - lo
    ophdiff = op - hi
    vdiff = v0 - vo

    
    #print('volume: ' + str(vo))
    #print('open: ' + str(op))
    #print('close: ' + str(cl))
    #print('hi: ' + str(hi))
    #print('low: ' + str(lo))
    #print('date: ' + da.strftime("%Y-%m-%d %H:%M:%S"))
    #print('oc diff: ' + str((cl-op)))


    #print('high-low diff: ' + str(diff))
    #print('closing low diff: ' + str(clldiff))
    #print('closing high diff: ' + str(clhdiff))
    #print('open low diff: ' + str(opldiff))
    #print('open high diff: ' + str(ophdiff))
    #print('volume diff: ' + str(vdiff))
    #print('______________________________')
    if temp >= op:
        signal = 1
    else:
        signal = 0
    temp = op
    v0 = vo
    
    if cl - op > 10:
        stat_co_110 += 1
        if vdiff > 50000:
            stat_vd_110 += 1
        elif vdiff > 0:
            stat_vd_100 += 1
    elif cl - op < -10:
        stat_co_010 += 1
        if vdiff < -50000:
            stat_vd_010 += 1
        elif vdiff < 0:
            stat_vd_000 += 1
    elif cl - op < 0:
        stat_co_000 += 1
    else:
        stat_co_100 += 1

print('___________________________________________________')
print(stat_co_110)
print(stat_co_100)
print(stat_co_000)
print(stat_co_010)

print(stat_vd_110)
print(stat_vd_100)
print(stat_vd_000)
print(stat_vd_010)

#print([obj.date for obj in state1L])
#print([obj.date for obj in state2L])
#print([obj.date for obj in state3L])
#print([obj.date for obj in state4L])
#opsrcL = openL
#opsrcL.reverse()

plt.figure(5)

plt.plot(tdL[900:],hd[900:],'b-', tdL[900:], openL[-900:-len(openL): -1], 'r-')

plt.figure(6)
plt.plot(tdL,hurstL)

plt.figure(7)
plt.plot(dateL, volumeL)

plt.figure(8)

plt.plot(tdL,hd)

plt.figure(9)

plt.plot(tdL[900:], emL[900:], 'b-', tdL[900:], openL[-900 + 1:-len(openL)+1:-1], 'r-')

plt.figure(10)

plt.plot(tdL[900:], hmmL[900:], 'b-', tdL[900:], openL[-900 + 1:-len(openL)+1:-1], 'r-')


plt.show()
